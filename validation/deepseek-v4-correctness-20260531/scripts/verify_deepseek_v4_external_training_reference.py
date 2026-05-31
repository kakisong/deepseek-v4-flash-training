#!/usr/bin/env python3
"""DeepSeek-V4 external training-reference one-step parity.

This verifier is the first external-training-reference gate after the official
inference parity work.  It does not compare two Miles attention backends.  It
builds a one-layer Megatron/Miles DeepSeek-V4 ``TransformerBlock`` as the system
under test, then compares it with an explicit PyTorch reference for the same
training-time math:

* block HyperConnection expand/head
* layer HyperConnection pre/post for attention and MLP
* RMSNorms
* non-compressed or deterministic compressed DeepSeek-V4 attention with dense
  masked reference attention
* Q/KV LoRA projections, RoPE, KV QAT, output projection
* standard GELU MLP
* backward gradients and one manual SGD update

The reference currently covers ``compress_ratio=0`` and a deterministic
``compress_ratio=128`` compressed-KV path with a non-MoE MLP so that each gate
is mathematically tight.  The ``compress_ratio=4`` indexer path, routed MoE, and
loaded mini-checkpoint SFT are separate extensions after these gates are stable.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.deepseek_v4.deepseek_v4 import get_dsv4_spec
from miles_plugins.models.deepseek_v4.ops.kernel.sinkhorn import hc_split_sinkhorn
from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat
from miles_plugins.models.deepseek_v4.ops.rope import apply_rotary_emb


SEED = 20260531


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_distributed() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=1,
            init_method=f"tcp://127.0.0.1:{_free_port()}",
            device_id=torch.device("cuda:0"),
        )
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(SEED)


def _destroy_distributed() -> None:
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _init_method(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.02)


def _build_config(compress_ratio: int) -> TransformerConfig:
    if compress_ratio not in (0, 128):
        raise ValueError(f"supported external-reference compress ratios are 0 and 128, got {compress_ratio}")
    config = TransformerConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        perform_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=8,
        ffn_hidden_size=1024,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        init_method=_init_method,
        output_layer_init_method=_init_method,
        experimental_attention_variant="dsv4",
        dsv4_hc_mult=4,
        dsv4_hc_sinkhorn_iters=2,
        dsv4_hc_eps=1e-6,
        dsv4_compress_ratios=[compress_ratio],
        dsv4_compress_rope_theta=160000,
        dsv4_o_groups=8,
        dsv4_o_lora_rank=1024,
        dsv4_window_size=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
    )
    for key, value in {
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_pos_emb_head_dim": 64,
        "rotary_base": 10000,
        "rotary_scaling_factor": 4,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    }.items():
        setattr(config, key, value)
    return config


def _build_block(config: TransformerConfig) -> TransformerBlock:
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp", "pp"])
    spec = get_dsv4_spec(None, config, None)
    return TransformerBlock(
        config,
        spec=spec,
        pre_process=True,
        post_process=True,
        pg_collection=pg_collection,
    ).cuda()


def _initialize_synthetic_state(block: TransformerBlock) -> None:
    with torch.no_grad():
        for name, param in block.named_parameters():
            if "hc_" in name and ("scale" in name or "base" in name):
                param.zero_()
            elif "norm" in name and param.ndim == 1:
                param.fill_(1.0)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)


def _clone_param_dict(block: TransformerBlock) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().clone().requires_grad_(True)
        for name, param in block.named_parameters()
    }


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(dtype)


def _hc_pre(
    x_sbhc: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_bshc = x_sbhc.permute(1, 0, 2, 3).contiguous()
    shape, dtype = x_bshc.size(), x_bshc.dtype
    x_flat = x_bshc.flatten(2).float()
    with torch.no_grad():
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale.float(), hc_base.float(), hc_mult, sinkhorn_iters, eps)
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
    return y.to(dtype).permute(1, 0, 2).contiguous(), post, comb


def _hc_post(
    x_sbd: torch.Tensor,
    residual_sbhc: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    x_bsd = x_sbd.permute(1, 0, 2).contiguous()
    residual_bshc = residual_sbhc.permute(1, 0, 2, 3).contiguous()
    term1 = post.unsqueeze(-1) * x_bsd.unsqueeze(-2)
    term2 = torch.sum(comb.unsqueeze(-1) * residual_bshc.unsqueeze(-2), dim=2)
    y = (term1 + term2).type_as(x_sbd)
    return y.permute(1, 0, 2, 3).contiguous()


def _hc_head(
    x_sbhc: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    *,
    norm_eps: float,
    eps: float,
) -> torch.Tensor:
    x_bshc = x_sbhc.permute(1, 0, 2, 3).contiguous()
    shape, dtype = x_bshc.size(), x_bshc.dtype
    x_flat = x_bshc.flatten(2).float()
    with torch.no_grad():
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + norm_eps)
        mixes = F.linear(x_flat, hc_fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + eps
    y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
    return y.to(dtype).permute(1, 0, 2).contiguous()


def _window_topk(batch: int, seqlen: int, window_size: int, device: torch.device) -> torch.Tensor:
    topk = torch.full((batch, seqlen, window_size), -1, device=device, dtype=torch.int32)
    for pos in range(seqlen):
        start = max(0, pos - window_size + 1)
        idx = torch.arange(start, pos + 1, device=device, dtype=torch.int32)
        topk[:, pos, : idx.numel()] = idx
    return topk


def _compress_topk(batch: int, seqlen: int, ratio: int, offset: int, device: torch.device) -> torch.Tensor:
    groups = seqlen // ratio
    matrix = torch.arange(groups, device=device, dtype=torch.int32).repeat(seqlen, 1)
    invalid = matrix >= (torch.arange(1, seqlen + 1, device=device, dtype=torch.int32).unsqueeze(1) // ratio)
    matrix = torch.where(invalid, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(batch, -1, -1)


def _compressor_reference(
    x_sbd: torch.Tensor,
    params: dict[str, torch.Tensor],
    freqs_cis: torch.Tensor,
    *,
    config: TransformerConfig,
    ratio: int,
) -> torch.Tensor:
    x = x_sbd.permute(1, 0, 2).contiguous()
    bsz, seqlen, _ = x.shape
    if seqlen < ratio or seqlen % ratio != 0:
        raise ValueError(
            "compressed reference requires seqlen >= ratio and divisible by ratio: "
            f"seqlen={seqlen} ratio={ratio}"
        )

    prefix = "layers.0.self_attention.compressor."
    head_dim = config.kv_lora_rank
    rd = config.qk_pos_emb_head_dim
    nope_dim = head_dim - rd

    x_fp32 = x.float()
    kv = F.linear(x_fp32, params[prefix + "wkv.weight"].float())
    score = F.linear(x_fp32, params[prefix + "wgate.weight"].float())
    kv = kv.unflatten(1, (-1, ratio))
    score = score.unflatten(1, (-1, ratio)) + params[prefix + "ape"].float()
    kv = (kv * score.softmax(dim=2)).sum(dim=2)
    kv = _rmsnorm(kv.to(x_sbd.dtype), params[prefix + "norm.weight"], config.layernorm_epsilon)
    kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], freqs_cis[:seqlen:ratio])], dim=-1)
    if os.environ.get("MEGATRON_USE_KV_QAT", "0") == "1":
        kv = torch.cat([fp8_simulate_qat(kv[..., :nope_dim].contiguous(), 64), kv[..., nope_dim:]], dim=-1)
    return kv


def _dense_attention_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    bsz, seqlen, n_heads, head_dim = q.shape
    kv_len = kv.shape[1]
    attn_mask = torch.zeros(bsz, seqlen, kv_len, device=q.device, dtype=torch.bool)
    batch_idx = torch.arange(bsz, device=q.device).view(bsz, 1, 1).expand_as(topk_idxs)
    seq_idx = torch.arange(seqlen, device=q.device).view(1, seqlen, 1).expand_as(topk_idxs)
    valid = topk_idxs != -1
    attn_mask[batch_idx[valid], seq_idx[valid], topk_idxs[valid].long()] = True

    scores = torch.einsum("bmhd,bnd->bmhn", q, kv).float() * sm_scale
    scores = scores.masked_fill(~attn_mask.unsqueeze(2), float("-inf"))
    scores_max = scores.max(dim=-1, keepdim=True).values
    scores_max = torch.maximum(scores_max, attn_sink.view(1, 1, n_heads, 1)).clamp(min=-1e30)
    exp_scores = torch.exp(scores - scores_max)
    numerator = torch.einsum("bmhn,bnd->bmhd", exp_scores, kv.float())
    denominator = exp_scores.sum(dim=-1) + torch.exp(attn_sink.view(1, 1, n_heads) - scores_max.squeeze(-1))
    return (numerator / denominator.unsqueeze(-1)).to(q.dtype)


def _attention_reference(
    x_sbd: torch.Tensor,
    params: dict[str, torch.Tensor],
    attention_freqs_cis: torch.Tensor,
    compressor_freqs_cis: torch.Tensor | None,
    *,
    config: TransformerConfig,
) -> torch.Tensor:
    x = x_sbd.permute(1, 0, 2).contiguous()
    bsz, seqlen, _ = x.shape
    rd = config.qk_pos_emb_head_dim
    n_heads = config.num_attention_heads
    head_dim = config.kv_lora_rank
    nope_dim = head_dim - rd

    q = F.linear(x, params["layers.0.self_attention.wq_a.weight"])
    q = _rmsnorm(q, params["layers.0.self_attention.q_norm.weight"], config.layernorm_epsilon)
    q = F.linear(q, params["layers.0.self_attention.wq_b.weight"])
    q = q.unflatten(-1, (n_heads, head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + config.layernorm_epsilon)
    q = torch.cat([q[..., :-rd], apply_rotary_emb(q[..., -rd:], attention_freqs_cis[:seqlen])], dim=-1)

    kv = F.linear(x, params["layers.0.self_attention.wkv.weight"])
    kv = _rmsnorm(kv, params["layers.0.self_attention.kv_norm.weight"], config.layernorm_epsilon)
    kv = torch.cat([kv[..., :-rd], apply_rotary_emb(kv[..., -rd:], attention_freqs_cis[:seqlen])], dim=-1)
    if os.environ.get("MEGATRON_USE_KV_QAT", "0") == "1":
        kv = torch.cat([fp8_simulate_qat(kv[..., :nope_dim].contiguous(), 64), kv[..., nope_dim:]], dim=-1)

    topk = _window_topk(bsz, seqlen, config.dsv4_window_size, x.device)
    ratio = int(config.dsv4_compress_ratios[0]) if config.dsv4_compress_ratios else 0
    if ratio:
        if compressor_freqs_cis is None:
            raise ValueError("compressor freqs_cis is required for compressed reference")
        kv_compress = _compressor_reference(x_sbd, params, compressor_freqs_cis, config=config, ratio=ratio)
        topk = torch.cat([topk, _compress_topk(bsz, seqlen, ratio, offset=seqlen, device=x.device)], dim=-1)
        kv = torch.cat([kv, kv_compress], dim=1)
    o = _dense_attention_reference(
        q,
        kv,
        params["layers.0.self_attention.attn_sink"].float(),
        topk,
        head_dim**-0.5,
    )
    o = torch.cat([o[..., :-rd], apply_rotary_emb(o[..., -rd:], attention_freqs_cis[:seqlen], inverse=True)], dim=-1)
    o = o.view(bsz, seqlen, config.dsv4_o_groups, -1)
    wo_a = params["layers.0.self_attention.wo_a.weight"].view(config.dsv4_o_groups, config.dsv4_o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    out = F.linear(o.flatten(2), params["layers.0.self_attention.wo_b.weight"])
    return out.permute(1, 0, 2).contiguous()


def _mlp_reference(x_sbd: torch.Tensor, params: dict[str, torch.Tensor], *, config: TransformerConfig) -> torch.Tensor:
    x = _rmsnorm(x_sbd, params["layers.0.mlp.linear_fc1.layer_norm_weight"], config.layernorm_epsilon)
    x = F.linear(x, params["layers.0.mlp.linear_fc1.weight"])
    x = F.gelu(x)
    return F.linear(x, params["layers.0.mlp.linear_fc2.weight"])


def _block_reference(
    hidden_states: torch.Tensor,
    params: dict[str, torch.Tensor],
    attention_freqs_cis: torch.Tensor,
    compressor_freqs_cis: torch.Tensor | None,
    *,
    config: TransformerConfig,
) -> torch.Tensor:
    hc_mult = config.dsv4_hc_mult
    x = hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()

    residual = x
    x_attn, attn_post, attn_comb = _hc_pre(
        x,
        params["layers.0.hc_attn_fn"],
        params["layers.0.hc_attn_scale"],
        params["layers.0.hc_attn_base"],
        hc_mult=hc_mult,
        sinkhorn_iters=config.dsv4_hc_sinkhorn_iters,
        eps=config.dsv4_hc_eps,
        norm_eps=config.layernorm_epsilon,
    )
    x_attn = _rmsnorm(x_attn, params["layers.0.input_layernorm.weight"], config.layernorm_epsilon)
    attn_out = _attention_reference(x_attn, params, attention_freqs_cis, compressor_freqs_cis, config=config)
    x = _hc_post(attn_out, residual, attn_post, attn_comb)

    residual = x
    x_mlp, ffn_post, ffn_comb = _hc_pre(
        x,
        params["layers.0.hc_ffn_fn"],
        params["layers.0.hc_ffn_scale"],
        params["layers.0.hc_ffn_base"],
        hc_mult=hc_mult,
        sinkhorn_iters=config.dsv4_hc_sinkhorn_iters,
        eps=config.dsv4_hc_eps,
        norm_eps=config.layernorm_epsilon,
    )
    mlp_out = _mlp_reference(x_mlp, params, config=config)
    x = _hc_post(mlp_out, residual, ffn_post, ffn_comb)

    x = _hc_head(
        x,
        params["hc_head_params.hc_head_fn"],
        params["hc_head_params.hc_head_scale"],
        params["hc_head_params.hc_head_base"],
        norm_eps=config.layernorm_epsilon,
        eps=config.dsv4_hc_eps,
    )
    return _rmsnorm(x, params["final_layernorm.weight"], config.layernorm_epsilon)


def _tensor_rel_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = (xf.square().sum() + yf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (xf * yf).sum().item() / denom)


def _summarize_pair(x: torch.Tensor, y: torch.Tensor) -> dict[str, Any]:
    diff = (x.detach().float() - y.detach().float()).abs()
    flat = diff.flatten()
    if flat.numel():
        kth = max(1, int(0.99 * flat.numel()))
        p99 = float(flat.kthvalue(kth).values.item())
    else:
        p99 = 0.0
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "p99_abs": p99,
        "relative_l2_gap": _tensor_rel_gap(x, y),
        "exact_equal": bool(torch.equal(x.detach().cpu(), y.detach().cpu())),
    }


def _manual_sgd_update(params: dict[str, torch.Tensor], lr: float) -> dict[str, torch.Tensor]:
    updated = {}
    with torch.no_grad():
        for name, param in params.items():
            grad = param.grad
            if grad is None:
                updated[name] = param.detach().float().cpu()
            else:
                param.add_(grad.to(param.dtype), alpha=-lr)
                updated[name] = param.detach().float().cpu()
    return updated


def _run_megatron(
    block: TransformerBlock,
    hidden_states: torch.Tensor,
    upstream_grad: torch.Tensor,
    *,
    lr: float,
) -> dict[str, Any]:
    block.train()
    x = hidden_states.detach().clone().requires_grad_(True)
    output = block(x, None)
    loss = (output.float() * upstream_grad.float()).mean()
    loss.backward()

    grads = {}
    updated = {}
    with torch.no_grad():
        for name, param in block.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().float().cpu()
                param.add_(param.grad.to(param.dtype), alpha=-lr)
            updated[name] = param.detach().float().cpu()
    return {
        "output": output.detach().cpu(),
        "loss": float(loss.item()),
        "input_grad": x.grad.detach().cpu(),
        "grads": grads,
        "updated": updated,
    }


def _run_reference(
    params: dict[str, torch.Tensor],
    attention_freqs_cis: torch.Tensor,
    compressor_freqs_cis: torch.Tensor | None,
    hidden_states: torch.Tensor,
    upstream_grad: torch.Tensor,
    *,
    config: TransformerConfig,
    lr: float,
) -> dict[str, Any]:
    x = hidden_states.detach().clone().requires_grad_(True)
    output = _block_reference(x, params, attention_freqs_cis, compressor_freqs_cis, config=config)
    loss = (output.float() * upstream_grad.float()).mean()
    loss.backward()
    grads = {
        name: param.grad.detach().float().cpu()
        for name, param in params.items()
        if param.grad is not None
    }
    updated = _manual_sgd_update(params, lr)
    return {
        "output": output.detach().cpu(),
        "loss": float(loss.item()),
        "input_grad": x.grad.detach().cpu(),
        "grads": grads,
        "updated": updated,
    }


def _compare_results(megatron: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    common_grad_names = sorted(set(megatron["grads"]) & set(reference["grads"]))
    common_state_names = sorted(set(megatron["updated"]) & set(reference["updated"]))
    grad_max = {"name": "", "max_abs": 0.0, "relative_l2_gap": 0.0}
    state_max = {"name": "", "max_abs": 0.0, "relative_l2_gap": 0.0}
    for name in common_grad_names:
        summary = _summarize_pair(megatron["grads"][name], reference["grads"][name])
        if summary["max_abs"] > grad_max["max_abs"]:
            grad_max = {
                "name": name,
                "max_abs": summary["max_abs"],
                "relative_l2_gap": summary["relative_l2_gap"],
            }
    for name in common_state_names:
        summary = _summarize_pair(megatron["updated"][name], reference["updated"][name])
        if summary["max_abs"] > state_max["max_abs"]:
            state_max = {
                "name": name,
                "max_abs": summary["max_abs"],
                "relative_l2_gap": summary["relative_l2_gap"],
            }
    return {
        "loss_abs": abs(megatron["loss"] - reference["loss"]),
        "output": _summarize_pair(megatron["output"], reference["output"]),
        "input_grad": _summarize_pair(megatron["input_grad"], reference["input_grad"]),
        "grad": grad_max,
        "state_after_step": state_max,
        "num_megatron_grad_tensors": len(megatron["grads"]),
        "num_reference_grad_tensors": len(reference["grads"]),
        "num_common_grad_tensors": len(common_grad_names),
        "num_common_state_tensors": len(common_state_names),
        "missing_reference_grad_tensors": sorted(set(megatron["grads"]) - set(reference["grads"])),
        "extra_reference_grad_tensors": sorted(set(reference["grads"]) - set(megatron["grads"])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--seqlen", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compress-ratio", type=int, choices=[0, 128], default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-loss-abs", type=float, default=None)
    parser.add_argument("--max-output-abs", type=float, default=None)
    parser.add_argument("--max-input-grad-abs", type=float, default=None)
    parser.add_argument("--max-grad-abs", type=float, default=None)
    parser.add_argument("--max-state-abs", type=float, default=None)
    args = parser.parse_args()

    if args.max_loss_abs is None:
        args.max_loss_abs = 0.0
    if args.max_output_abs is None:
        args.max_output_abs = 0.0
    if args.max_input_grad_abs is None:
        args.max_input_grad_abs = 0.0 if args.compress_ratio == 0 else 1e-7
    if args.max_grad_abs is None:
        args.max_grad_abs = 1e-7 if args.compress_ratio == 0 else 5e-7
    if args.max_state_abs is None:
        args.max_state_abs = 0.0 if args.compress_ratio == 0 else 1e-9

    _init_distributed()
    try:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = "dense"
        if args.compress_ratio and (args.seqlen < args.compress_ratio or args.seqlen % args.compress_ratio != 0):
            raise ValueError(
                f"--seqlen must be >= --compress-ratio and divisible by it: "
                f"seqlen={args.seqlen} compress_ratio={args.compress_ratio}"
            )
        config = _build_config(args.compress_ratio)
        block = _build_block(config)
        _initialize_synthetic_state(block)
        reference_params = _clone_param_dict(block)
        attention_freqs_cis = block.layers[0].self_attention.freqs_cis.detach()
        compressor_freqs_cis = (
            block.layers[0].self_attention.compressor.freqs_cis.detach()
            if args.compress_ratio
            else None
        )
        num_parameters = sum(param.numel() for param in block.parameters())

        hidden_states = torch.randn(
            args.seqlen,
            args.batch_size,
            config.hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        upstream_grad = torch.randn_like(hidden_states)

        megatron = _run_megatron(block, hidden_states, upstream_grad, lr=args.lr)
        reference = _run_reference(
            reference_params,
            attention_freqs_cis,
            compressor_freqs_cis,
            hidden_states,
            upstream_grad,
            config=config,
            lr=args.lr,
        )
        comparison = _compare_results(megatron, reference)

        checks = {
            "loss_abs": comparison["loss_abs"] <= args.max_loss_abs,
            "output_max_abs": comparison["output"]["max_abs"] <= args.max_output_abs,
            "input_grad_max_abs": comparison["input_grad"]["max_abs"] <= args.max_input_grad_abs,
            "grad_max_abs": comparison["grad"]["max_abs"] <= args.max_grad_abs,
            "state_after_step_max_abs": comparison["state_after_step"]["max_abs"] <= args.max_state_abs,
            "grad_tensor_sets_match": not comparison["missing_reference_grad_tensors"]
            and not comparison["extra_reference_grad_tensors"],
        }
        failures = [name for name, passed in checks.items() if not passed]
        payload = {
            "date": "2026-05-31",
            "seed": SEED,
            "status": "PASS" if not failures else "FAIL",
            "scope": (
                "external PyTorch training reference vs Megatron/Miles DeepSeek-V4 "
                f"one-layer compress_ratio={args.compress_ratio} TransformerBlock"
            ),
            "reference": "explicit PyTorch formula reference outside the Megatron module forward graph",
            "config": {
                "num_layers": 1,
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "ffn_hidden_size": config.ffn_hidden_size,
                "compress_ratio": args.compress_ratio,
                "dsv4_hc_mult": config.dsv4_hc_mult,
                "dsv4_hc_sinkhorn_iters": config.dsv4_hc_sinkhorn_iters,
                "normalization": config.normalization,
                "mlp": "standard GELU MLP",
            },
            "seqlen": args.seqlen,
            "batch_size": args.batch_size,
            "num_parameters": num_parameters,
            "manual_update": {"rule": "sgd", "lr": args.lr},
            "thresholds": {
                "max_loss_abs": args.max_loss_abs,
                "max_output_abs": args.max_output_abs,
                "max_input_grad_abs": args.max_input_grad_abs,
                "max_grad_abs": args.max_grad_abs,
                "max_state_abs": args.max_state_abs,
            },
            "checks": checks,
            "comparison": comparison,
            "failures": failures,
            "boundary": (
                "This closes an external training-reference gate for a one-layer "
                "DeepSeek-V4 training block. Routed MoE and loaded mini-checkpoint SFT "
                "are intentionally left as follow-up extensions."
            ),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {args.json_output}")
        return 0 if not failures else 1
    finally:
        _destroy_distributed()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")
    os.environ.setdefault("MILES_DSV4_CKPT_VERSION", "0415")
    raise SystemExit(main())

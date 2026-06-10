#!/usr/bin/env python3
"""DeepSeek-V4 已加载 mini checkpoint 的外部全模型 reference。

本校验器加载 4 层 mini checkpoint 与固定的 SFT batch，然后在同一份模型
参数上运行两条 forward/loss 路径：

* 真实的 Miles/Megatron GPTModel 路径；
* 一条显式 PyTorch reference 路径，覆盖 embedding、四个 DeepSeek-V4 层、
  dense reference attention、EP=8 哈希路由 / 分数路由的 MoE forward、
  最终 norm、输出 head 以及 SFT loss。

reference 路径对模型主体不调用 Megatron 模块的 ``forward`` 方法。为了让
已加载的 checkpoint 在内存上可行，它与 Miles 路径共享同一批参数 tensor，
而不是克隆完整的 4 层 EP=8 模型。显式 reference 使用 BF16/FP32 的 PyTorch
数学计算；本校验器不模拟 FP8/blockwise 量化，因此 FP8 的 Miles 运行只
记录为诊断信息，而不作为严格 PASS 的证据。backward 的检查方式是对本地
loss 差值做反向传播，并测量选定的非 expert 参数的变化量。完整的 EP
expert backward/update 有意交由专门的真实 EP=8 MoELayer reference 覆盖，
因为真实的 all-to-all backward 会从所有源 rank 累积本地 expert 梯度，
而复制的非 expert 权重得到的是本地 rank 的梯度。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_model

import miles_plugins.mbridge  # noqa: F401
from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state, get_packed_seq_params
from miles.backends.training_utils.data import DataIterator, get_batch
from miles.backends.training_utils.loss import loss_function
from miles.utils.logging_utils import configure_logger

try:
    from verification.verify_deepseek_v4_external_training_reference import (
        SEED,
        _attention_reference,
        _hc_head,
        _hc_post,
        _hc_pre,
        _rmsnorm,
        _swiglu_reference,
        _tensor_rel_gap,
    )
    from verification.verify_deepseek_v4_mini_train_step_parity import (
        _is_selected_param,
        _manual_update_param,
        _prepare_manual_grad_buffers,
        _zero_manual_grad_buffers,
    )
    from verification.verify_deepseek_v4_sft_loss_reference import (
        _all_reduce_float,
        _all_reduce_int,
        _explicit_sft_reference,
        _init_distributed,
        _load_rollout_data,
        _move_rollout_to_device,
        _summarize_log,
    )
except ModuleNotFoundError:
    from verify_deepseek_v4_external_training_reference import (  # type: ignore
        SEED,
        _attention_reference,
        _hc_head,
        _hc_post,
        _hc_pre,
        _rmsnorm,
        _swiglu_reference,
        _tensor_rel_gap,
    )
    from verify_deepseek_v4_mini_train_step_parity import (  # type: ignore
        _is_selected_param,
        _manual_update_param,
        _prepare_manual_grad_buffers,
        _zero_manual_grad_buffers,
    )
    from verify_deepseek_v4_sft_loss_reference import (  # type: ignore
        _all_reduce_float,
        _all_reduce_int,
        _explicit_sft_reference,
        _init_distributed,
        _load_rollout_data,
        _move_rollout_to_device,
        _summarize_log,
    )


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--impl", choices=["dense"], default="dense")
    parser.add_argument("--manual-update-rule", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--manual-sgd-lr", type=float, default=1e-7)
    parser.add_argument("--manual-adamw-lr", type=float, default=None)
    parser.add_argument("--manual-adamw-beta1", type=float, default=0.9)
    parser.add_argument("--manual-adamw-beta2", type=float, default=0.98)
    parser.add_argument("--manual-adamw-eps", type=float, default=1e-8)
    parser.add_argument("--manual-adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--manual-update-selected-only", action="store_true")
    parser.add_argument("--max-selected-numel", type=int, default=5_000_000)
    parser.add_argument("--qkv-format", choices=["thd", "bshd"], default="thd")
    parser.add_argument("--data-pad-size-multiplier", type=int, default=128)
    parser.add_argument("--log-probs-chunk-size", type=int, default=-1)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--allgather-cp", action="store_true")
    parser.add_argument("--true-on-policy-mode", action="store_true")
    parser.add_argument("--use-rollout-entropy", action="store_true")
    parser.add_argument("--use-dynamic-batch-size", action="store_true")
    parser.add_argument("--max-logit-abs", type=float, default=2e-1)
    parser.add_argument("--max-logit-mean-abs", type=float, default=4e-2)
    parser.add_argument("--max-logit-p99-abs", type=float, default=1.5e-1)
    parser.add_argument("--max-logit-rel-gap", type=float, default=5e-6)
    parser.add_argument("--max-loss-abs", type=float, default=2e-3)
    parser.add_argument("--max-loss-abs-per-token", type=float, default=-1.0)
    parser.add_argument("--max-token-count-abs", type=float, default=0.0)
    parser.add_argument("--max-selected-grad-abs", type=float, default=1e-2)
    parser.add_argument("--max-selected-grad-rel-gap", type=float, default=1e-2)
    parser.add_argument("--max-selected-state-abs", type=float, default=2e-5)
    parser.add_argument("--no-offload-reference-saved-tensors", action="store_true")
    parser.add_argument("--skip-backward-check", action="store_true")
    parser.add_argument("--replay-miles-routing", action="store_true")
    parser.add_argument("--tolerance-profile", type=str, default="strict_external_reference")
    parser.add_argument("--debug-layer-gaps", action="store_true")
    parser.add_argument("--debug-first-layer-submodules", action="store_true")
    parser.add_argument("--debug-all-layer-submodules", action="store_true")
    return parser


def _parse_args():
    args = parse_args(extra_args_provider=add_args)
    args = set_default_megatron_args(args)
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = True
    args.loss_type = "sft_loss"
    args.calculate_per_token_loss = True
    args.disable_compute_advantages_and_returns = True
    args.__dict__.setdefault("custom_megatron_before_log_prob_hook_path", None)
    args.__dict__.setdefault("custom_megatron_before_train_step_hook_path", None)
    args.__dict__.setdefault("custom_loss_function_path", None)
    args.__dict__.setdefault("recompute_loss_function", False)
    args.__dict__.setdefault("enable_mtp_training", False)
    args.__dict__.setdefault("use_rollout_logprobs", False)
    args.__dict__.setdefault("use_opsm", False)
    args.__dict__.setdefault("use_tis", False)
    args.__dict__.setdefault("get_mismatch_metrics", False)
    args.__dict__.setdefault("custom_pg_loss_reducer_function_path", None)
    args.save_interval = getattr(args, "save_interval", None) or 1
    args.micro_batch_size = getattr(args, "micro_batch_size", None) or args.max_samples
    args.global_batch_size = getattr(args, "global_batch_size", None) or args.micro_batch_size
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    validate_args(args)
    args.variable_seq_lengths = True
    return args


def _param_dict(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    params = dict(module.named_parameters())
    for name, value in list(params.items()):
        if name.startswith("module."):
            params.setdefault(name[len("module.") :], value)
    return params


def _buffer_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    buffers = dict(module.named_buffers())
    for name, value in list(buffers.items()):
        if name.startswith("module."):
            buffers.setdefault(name[len("module.") :], value)
    return buffers


def _unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "module", module)


def _model_vocab_size(module: torch.nn.Module, args: argparse.Namespace) -> int:
    raw_model = _unwrap_model(module)
    config = getattr(raw_model, "config", None)
    for source in (raw_model, config, args):
        if source is None:
            continue
        for attr in ("vocab_size", "padded_vocab_size"):
            value = getattr(source, attr, None)
            if value is not None:
                return int(value)
    return int(_get(_param_dict(module), "output_layer.weight").shape[0])


def _get(mapping: dict[str, Any], name: str) -> Any:
    if name not in mapping:
        similar = [key for key in mapping if key.endswith(name.split(".")[-1])][:20]
        raise KeyError(f"missing {name}; similar suffixes={similar}")
    return mapping[name]


def _layer_view(mapping: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    prefix = f"decoder.layers.{layer_idx}."
    view: dict[str, Any] = {}
    for name, value in mapping.items():
        if name.startswith(prefix):
            view["layers.0." + name[len(prefix) :]] = value
    return view


def _combined_grad(param: torch.nn.Parameter) -> torch.Tensor | None:
    grads = []
    main_grad = getattr(param, "main_grad", None)
    if main_grad is not None:
        grads.append(main_grad.detach().float())
    if param.grad is not None:
        grads.append(param.grad.detach().float())
    if not grads:
        return None
    out = grads[0].clone()
    for grad in grads[1:]:
        out = out + grad
    return out


def _tensor_gap(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    diff = (left.detach().float() - right.detach().float()).abs().flatten()
    if diff.numel():
        kth = max(1, int(0.99 * diff.numel()))
        p99 = float(diff.kthvalue(kth).values.item())
    else:
        p99 = 0.0
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "p99_abs": p99,
        "relative_l2_gap": _tensor_rel_gap(left, right),
        "numel": int(diff.numel()),
        "nonzero_abs_count": int((diff != 0).sum().item()) if diff.numel() else 0,
    }


def _capture_decoder_layer_outputs(
    model: torch.nn.Module,
    *,
    first_layer_submodules: bool,
    all_layer_submodules: bool,
    capture_router: bool,
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    raw_model = _unwrap_model(model)
    outputs: dict[str, torch.Tensor] = {}
    hooks = []

    def _first_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, (tuple, list)):
            return _first_tensor(value[0])
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"expected tensor output, got {type(value)}")
        return value

    for layer_idx, layer in enumerate(raw_model.decoder.layers):
        def _hook(_module, _inputs, output, *, layer_idx=layer_idx):
            outputs[f"layer_{layer_idx}"] = _first_tensor(output).detach()

        hooks.append(layer.register_forward_hook(_hook))

    if raw_model.decoder.layers and (first_layer_submodules or all_layer_submodules):
        layer_indices = range(len(raw_model.decoder.layers)) if all_layer_submodules else range(1)
        for sub_layer_idx in layer_indices:
            layer = raw_model.decoder.layers[sub_layer_idx]
            submodules = {
                f"layer_{sub_layer_idx}.input_layernorm": layer.input_layernorm,
                f"layer_{sub_layer_idx}.self_attention": layer.self_attention,
                f"layer_{sub_layer_idx}.pre_mlp_layernorm": layer.pre_mlp_layernorm,
                f"layer_{sub_layer_idx}.mlp": layer.mlp,
            }
            for name, module in submodules.items():
                def _submodule_hook(_module, _inputs, output, *, name=name):
                    outputs[name] = _first_tensor(output).detach()

                hooks.append(module.register_forward_hook(_submodule_hook))
    if raw_model.decoder.layers and capture_router:
        for sub_layer_idx, layer in enumerate(raw_model.decoder.layers):
            if hasattr(layer.mlp, "router"):
                def _router_hook(_module, _inputs, output, *, sub_layer_idx=sub_layer_idx):
                    probs, routing_map = output
                    outputs[f"layer_{sub_layer_idx}.router_probs"] = probs.detach()
                    outputs[f"layer_{sub_layer_idx}.router_map"] = routing_map.detach().float()

                hooks.append(layer.mlp.router.register_forward_hook(_router_hook))
    return outputs, hooks


def _global_tensor_gap(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {
            "status": "SHAPE_MISMATCH",
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    gap = _tensor_gap(left, right)
    return {
        "status": "OK",
        "max_abs": _all_reduce_float(gap["max_abs"], dist.ReduceOp.MAX),
        "mean_abs": _all_reduce_float(gap["mean_abs"], dist.ReduceOp.MAX),
        "p99_abs": _all_reduce_float(gap["p99_abs"], dist.ReduceOp.MAX),
        "relative_l2_gap": _all_reduce_float(gap["relative_l2_gap"], dist.ReduceOp.MAX),
    }


def _local_expert_tensors(params: dict[str, Any], prefix: str) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    fc1: list[tuple[int, torch.Tensor]] = []
    fc2: list[tuple[int, torch.Tensor]] = []
    for name, param in params.items():
        if name.startswith(prefix + "experts.linear_fc1.weight"):
            fc1.append((int(name.rsplit("weight", 1)[1]), param))
        elif name.startswith(prefix + "experts.linear_fc2.weight"):
            fc2.append((int(name.rsplit("weight", 1)[1]), param))
    fc1 = sorted(fc1)
    fc2 = sorted(fc2)
    if not fc1 or len(fc1) != len(fc2):
        raise ValueError(f"invalid local expert tensors for {prefix}: fc1={len(fc1)} fc2={len(fc2)}")
    return [item[1] for item in fc1], [item[1] for item in fc2]


def _expert_tensor(local_tensors: list[torch.Tensor], expert_id: int) -> torch.Tensor:
    rank = dist.get_rank()
    local_experts = len(local_tensors)
    owner = expert_id // local_experts
    local_idx = expert_id % local_experts
    if owner == rank:
        send = local_tensors[local_idx].detach().contiguous()
    else:
        send = torch.empty_like(local_tensors[0], memory_format=torch.contiguous_format)
    dist.broadcast(send, src=owner)
    return local_tensors[local_idx] if owner == rank else send


def _moe_ep_reference(
    x_sbd: torch.Tensor,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    *,
    config: Any,
    input_ids: torch.Tensor,
    layer_idx: int,
    trace: dict[str, torch.Tensor] | None = None,
    replay_trace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    prefix = "layers.0.mlp."
    x_flat = x_sbd.contiguous().view(-1, x_sbd.shape[-1])
    router_weight = _get(params, prefix + "router.weight")
    router_dtype = x_flat.dtype
    if config.moe_router_dtype == "fp32":
        router_dtype = torch.float32
    elif config.moe_router_dtype == "fp64":
        router_dtype = torch.float64
    logits = F.linear(x_flat.to(router_dtype), router_weight.to(router_dtype))
    if config.moe_router_score_function != "sqrtsoftplus":
        raise ValueError(f"unsupported router score function: {config.moe_router_score_function}")
    scores = F.softplus(logits.float()).sqrt().to(logits.dtype)
    replay_map = None
    if replay_trace is not None:
        replay_map = replay_trace.get(f"layer_{layer_idx}.router_map")
    if replay_map is not None:
        routing_map = replay_map.to(device=x_flat.device).bool()
        if tuple(routing_map.shape) != tuple(logits.shape):
            raise ValueError(
                f"routing replay shape mismatch for layer {layer_idx}: "
                f"replay={tuple(routing_map.shape)} logits={tuple(logits.shape)}"
            )
        selected_scores = scores * routing_map.to(scores.dtype)
        top_probs_full = selected_scores / (selected_scores.sum(dim=-1, keepdim=True) + 1e-20)
        if config.moe_router_topk_scaling_factor:
            top_probs_full = top_probs_full * config.moe_router_topk_scaling_factor
        routing_probs = top_probs_full.to(logits.dtype)
    else:
        tid2eid = params.get(prefix + "router.tid2eid")
        if tid2eid is not None:
            input_ids_flat = input_ids.reshape(-1)
            if input_ids_flat.numel() != x_flat.shape[0]:
                raise ValueError(
                    "hash-routed MoE reference expects one input id per routed token; "
                    f"got input_ids={tuple(input_ids.shape)} flat={input_ids_flat.numel()} "
                    f"tokens={x_flat.shape[0]}"
                )
            top_indices = tid2eid[input_ids_flat].to(device=x_flat.device, dtype=torch.long)
            if bool((top_indices < 0).any().item()):
                raise ValueError("hash-routed MoE tid2eid contains unresolved expert ids")
        else:
            expert_bias = buffers.get(prefix + "router.expert_bias")
            if expert_bias is None:
                expert_bias = torch.zeros(config.num_moe_experts, device=x_flat.device, dtype=torch.float32)
            _, top_indices = torch.topk(scores + expert_bias.to(scores.dtype), k=config.moe_router_topk, dim=1)
        top_scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        top_probs = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
        if config.moe_router_topk_scaling_factor:
            top_probs = top_probs * config.moe_router_topk_scaling_factor
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, top_probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    if trace is not None:
        trace[f"layer_{layer_idx}.router_probs"] = routing_probs.detach()
        trace[f"layer_{layer_idx}.router_map"] = routing_map.detach().float()

    local_fc1, local_fc2 = _local_expert_tensors(params, prefix)
    local_experts = len(local_fc1)
    if local_experts * dist.get_world_size() != config.num_moe_experts:
        raise ValueError(
            f"expected {config.num_moe_experts} experts, got local={local_experts} world={dist.get_world_size()}"
        )

    routed_output = torch.zeros_like(x_flat)
    clamp = float(config.activation_func_clamp_value) if config.activation_func_clamp_value is not None else None
    # 所有 EP rank 必须以相同顺序执行 broadcast，即使各自本地回放的
    # routing map 选择了不同的 expert 子集。
    for expert_id in range(int(config.num_moe_experts)):
        fc1_weight = _expert_tensor(local_fc1, expert_id)
        fc2_weight = _expert_tensor(local_fc2, expert_id)
        selected = routing_map[:, expert_id]
        if not bool(selected.any().item()):
            continue
        token_idx = selected.nonzero(as_tuple=True)[0]
        expert_input = x_flat[token_idx]
        probs = routing_probs[token_idx, expert_id].unsqueeze(-1)
        fc1 = F.linear(expert_input, fc1_weight)
        intermediate = _swiglu_reference(fc1, clamp=clamp)
        intermediate = intermediate * probs.to(intermediate.dtype)
        routed_output.index_add_(0, token_idx, F.linear(intermediate, fc2_weight))

    shared_fc1 = F.linear(x_flat, _get(params, prefix + "shared_experts.linear_fc1.weight"))
    shared = _swiglu_reference(shared_fc1, clamp=None)
    shared = F.linear(shared, _get(params, prefix + "shared_experts.linear_fc2.weight"))
    return (routed_output + shared).view_as(x_sbd)


def _layer_reference(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    layer_params: dict[str, torch.Tensor],
    layer_buffers: dict[str, torch.Tensor],
    layer_module: Any,
    *,
    config: Any,
    layer_idx: int,
    trace: dict[str, torch.Tensor] | None = None,
    replay_trace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    hc_mult = config.dsv4_hc_mult
    residual = hidden_states
    x_attn, attn_post, attn_comb = _hc_pre(
        hidden_states,
        _get(layer_params, "layers.0.hc_attn_fn"),
        _get(layer_params, "layers.0.hc_attn_scale"),
        _get(layer_params, "layers.0.hc_attn_base"),
        hc_mult=hc_mult,
        sinkhorn_iters=config.dsv4_hc_sinkhorn_iters,
        eps=config.dsv4_hc_eps,
        norm_eps=config.layernorm_epsilon,
    )
    x_attn = _rmsnorm(x_attn, _get(layer_params, "layers.0.input_layernorm.weight"), config.layernorm_epsilon)
    if trace is not None:
        trace[f"layer_{layer_idx}.input_layernorm"] = x_attn.detach()
    attn = layer_module.self_attention
    attn_config = copy.copy(config)
    attn_config.dsv4_compress_ratios = [int(getattr(attn, "compress_ratio", 0))]
    compressor_freqs = attn.compressor.freqs_cis.detach() if getattr(attn, "compress_ratio", 0) else None
    indexer_freqs = None
    indexer_compressor_freqs = None
    if getattr(attn, "compress_ratio", 0) == 4:
        indexer_freqs = attn.indexer.freqs_cis.detach()
        indexer_compressor_freqs = attn.indexer.compressor.freqs_cis.detach()
    attn_out = _attention_reference(
        x_attn,
        layer_params,
        attn.freqs_cis.detach(),
        compressor_freqs,
        indexer_freqs,
        indexer_compressor_freqs,
        config=attn_config,
    )
    if trace is not None:
        trace[f"layer_{layer_idx}.self_attention"] = attn_out.detach()
    hidden_states = _hc_post(attn_out, residual, attn_post, attn_comb)

    residual = hidden_states
    x_mlp, ffn_post, ffn_comb = _hc_pre(
        hidden_states,
        _get(layer_params, "layers.0.hc_ffn_fn"),
        _get(layer_params, "layers.0.hc_ffn_scale"),
        _get(layer_params, "layers.0.hc_ffn_base"),
        hc_mult=hc_mult,
        sinkhorn_iters=config.dsv4_hc_sinkhorn_iters,
        eps=config.dsv4_hc_eps,
        norm_eps=config.layernorm_epsilon,
    )
    x_mlp = _rmsnorm(x_mlp, _get(layer_params, "layers.0.pre_mlp_layernorm.weight"), config.layernorm_epsilon)
    if trace is not None:
        trace[f"layer_{layer_idx}.pre_mlp_layernorm"] = x_mlp.detach()
    mlp_out = _moe_ep_reference(
        x_mlp,
        layer_params,
        layer_buffers,
        config=config,
        input_ids=input_ids,
        layer_idx=layer_idx,
        trace=trace,
        replay_trace=replay_trace,
    )
    if trace is not None:
        trace[f"layer_{layer_idx}.mlp"] = mlp_out.detach()
    return _hc_post(mlp_out, residual, ffn_post, ffn_comb)


def _full_reference(
    module: torch.nn.Module,
    input_ids: torch.Tensor,
    trace: dict[str, torch.Tensor] | None = None,
    replay_trace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    params = _param_dict(module)
    buffers = _buffer_dict(module)
    model = _unwrap_model(module)
    config = model.config
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    hidden = F.embedding(input_ids, _get(params, "embedding.word_embeddings.weight")).transpose(0, 1).contiguous()
    hidden = hidden.unsqueeze(2).expand(-1, -1, config.dsv4_hc_mult, -1).contiguous()
    if trace is not None:
        trace["embedding"] = hidden.detach()
    for layer_idx, layer_module in enumerate(model.decoder.layers):
        hidden = _layer_reference(
            hidden,
            input_ids,
            _layer_view(params, layer_idx),
            _layer_view(buffers, layer_idx),
            layer_module,
            config=config,
            layer_idx=layer_idx,
            trace=trace,
            replay_trace=replay_trace,
        )
        if trace is not None:
            trace[f"layer_{layer_idx}"] = hidden.detach()
    hidden = _hc_head(
        hidden,
        _get(params, "decoder.hc_head_params.hc_head_fn"),
        _get(params, "decoder.hc_head_params.hc_head_scale"),
        _get(params, "decoder.hc_head_params.hc_head_base"),
        norm_eps=config.layernorm_epsilon,
        eps=config.dsv4_hc_eps,
    )
    hidden = _rmsnorm(hidden, _get(params, "decoder.final_layernorm.weight"), config.layernorm_epsilon)
    logits_sbv = F.linear(hidden, _get(params, "output_layer.weight")).float()
    return logits_sbv.transpose(0, 1).contiguous()


def _reference_saved_tensor_context(enabled: bool):
    if not enabled:
        return nullcontext()

    def pack(tensor: torch.Tensor):
        if not tensor.is_cuda:
            return ("cpu", None, tensor)
        return ("cuda", tensor.device, tensor.detach().cpu())

    def unpack(payload):
        tag, device, tensor = payload
        if tag == "cuda":
            return tensor.to(device=device, non_blocking=True)
        return tensor

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def _selected_delta_stats(model: list[torch.nn.Module], args: Any) -> dict[str, Any]:
    local_selected = 0
    local_selected_with_grad = 0
    local_nonfinite = 0
    local_grad_max_abs = 0.0
    local_grad_rel_max = 0.0
    local_state_max_abs = 0.0
    local_grad_norm_sq = 0.0
    local_param_norm_sq = 0.0
    local_grad_elems = 0
    worst_grad = {"name": "", "max_abs": 0.0, "relative_to_param_l2": 0.0}
    worst_state = {"name": "", "max_abs": 0.0}
    with torch.no_grad():
        for module in model:
            for name, param in module.named_parameters():
                if "experts.linear_fc" in name:
                    continue
                if not _is_selected_param(name, param, args.max_selected_numel):
                    continue
                local_selected += 1
                grad = _combined_grad(param)
                if grad is None:
                    continue
                local_selected_with_grad += 1
                finite = bool(torch.isfinite(grad).all().item())
                if not finite:
                    local_nonfinite += 1
                    continue
                grad_abs = float(grad.abs().max().item()) if grad.numel() else 0.0
                param_l2 = float(param.detach().float().square().sum().item())
                grad_l2 = float(grad.float().square().sum().item())
                rel = grad_l2 / max(param_l2, 1e-30)
                local_grad_norm_sq += grad_l2
                local_param_norm_sq += param_l2
                local_grad_elems += int(grad.numel())
                local_grad_max_abs = max(local_grad_max_abs, grad_abs)
                local_grad_rel_max = max(local_grad_rel_max, rel)
                if grad_abs > worst_grad["max_abs"]:
                    worst_grad = {"name": name, "max_abs": grad_abs, "relative_to_param_l2": rel}
                before = param.detach().clone()
                _manual_update_param(param, grad.to(param.device), args)
                after = param.detach().clone()
                param.copy_(before)
                state_abs = float((after.float() - before.float()).abs().max().item())
                local_state_max_abs = max(local_state_max_abs, state_abs)
                if state_abs > worst_state["max_abs"]:
                    worst_state = {"name": name, "max_abs": state_abs}
    global_grad_norm_sq = _all_reduce_float(local_grad_norm_sq, dist.ReduceOp.SUM)
    global_param_norm_sq = _all_reduce_float(local_param_norm_sq, dist.ReduceOp.SUM)
    return {
        "selected_tensors_local": local_selected,
        "selected_tensors_with_grad_local": local_selected_with_grad,
        "selected_tensors_global": _all_reduce_int(local_selected, dist.ReduceOp.SUM),
        "selected_tensors_with_grad_global": _all_reduce_int(local_selected_with_grad, dist.ReduceOp.SUM),
        "nonfinite_selected_grad_tensors_global": _all_reduce_int(local_nonfinite, dist.ReduceOp.SUM),
        "selected_grad_max_abs_global_max": _all_reduce_float(local_grad_max_abs, dist.ReduceOp.MAX),
        "selected_grad_relative_to_param_l2_global": global_grad_norm_sq / max(global_param_norm_sq, 1e-30),
        "selected_grad_relative_to_param_l2_global_max": _all_reduce_float(local_grad_rel_max, dist.ReduceOp.MAX),
        "selected_state_max_abs_global_max": _all_reduce_float(local_state_max_abs, dist.ReduceOp.MAX),
        "selected_grad_numel_global": _all_reduce_int(local_grad_elems, dist.ReduceOp.SUM),
        "worst_grad_local": worst_grad,
        "worst_state_local": worst_state,
        "gradient_method": "delta.backward_main_grad_buffers",
    }


def main() -> int:
    configure_logger()
    _init_distributed()
    args = _parse_args()
    try:
        init(args)
        os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = args.impl
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
        load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
        for module in model:
            module.train()
        _prepare_manual_grad_buffers(model, args)
        raw_model = _unwrap_model(model[0])
        model_config = raw_model.config

        parallel_state = create_megatron_parallel_state(model)
        ep_size = int(getattr(args, "expert_model_parallel_size", 1))
        if parallel_state.tp_size != 1 or parallel_state.cp_size != 1 or ep_size != dist.get_world_size():
            raise ValueError(
                "full reference expects TP=1, CP=1, and EP equal to world size; "
                f"got TP={parallel_state.tp_size} CP={parallel_state.cp_size} EP={ep_size} "
                f"world={dist.get_world_size()}"
            )

        device = torch.device("cuda", torch.cuda.current_device())
        rollout_data = _move_rollout_to_device(_load_rollout_data(args.rollout_data, args.max_samples), device)
        data_iterator = DataIterator(rollout_data, micro_batch_size=len(rollout_data["tokens"]))
        batch = get_batch(
            data_iterator,
            [
                "tokens",
                "multimodal_train_inputs",
                "packed_seq_params",
                "total_lengths",
                "response_lengths",
                "loss_masks",
                "log_probs",
                "ref_log_probs",
                "values",
                "advantages",
                "returns",
                "rollout_log_probs",
                "max_seq_lens",
            ],
            parallel_state,
            args.data_pad_size_multiplier,
            args.qkv_format,
            allgather_cp=args.allgather_cp,
        )
        if batch["multimodal_train_inputs"] is not None:
            raise ValueError("full external reference currently expects text-only SFT batches")

        forward_kwargs = {
            "input_ids": batch["tokens"],
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": get_packed_seq_params(batch, args),
            "loss_mask": batch["full_loss_masks"],
        }
        miles_traces = {}
        hooks = []
        if args.debug_layer_gaps or args.replay_miles_routing:
            miles_traces, hooks = _capture_decoder_layer_outputs(
                model[0],
                first_layer_submodules=args.debug_first_layer_submodules,
                all_layer_submodules=args.debug_all_layer_submodules,
                capture_router=args.replay_miles_routing or args.debug_all_layer_submodules or args.debug_first_layer_submodules,
            )
        try:
            miles_logits = model[0](**forward_kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        if not torch.isfinite(miles_logits).all():
            raise RuntimeError("Miles logits contain non-finite values")
        ref_traces = {} if args.debug_layer_gaps else None
        with _reference_saved_tensor_context(not args.no_offload_reference_saved_tensors):
            ref_logits = _full_reference(
                model[0],
                batch["tokens"],
                trace=ref_traces,
                replay_trace=miles_traces if args.replay_miles_routing else None,
            )
            if not torch.isfinite(ref_logits).all():
                raise RuntimeError("reference logits contain non-finite values")
            explicit_ref = _explicit_sft_reference(logits=ref_logits, batch=batch, args=args)

        miles_loss, normalizer, loss_log = loss_function(
            args,
            parallel_state,
            batch,
            num_microbatches=1,
            logits=miles_logits,
            apply_megatron_loss_scaling=False,
        )
        token_count_abs = abs(float(normalizer.detach().item()) - float(explicit_ref["token_count"].detach().item()))
        loss_abs = abs(float(miles_loss.detach().item()) - float(explicit_ref["loss_sum"].detach().item()))
        token_count = float(explicit_ref["token_count"].detach().item())
        loss_abs_per_token = loss_abs / max(token_count, 1.0)
        logit_gap = _tensor_gap(miles_logits, ref_logits)

        if args.skip_backward_check:
            grad_stats: dict[str, Any] = {"status": "SKIPPED_FORWARD_ONLY"}
        else:
            _zero_manual_grad_buffers(model)
            delta = miles_loss - explicit_ref["loss_sum"]
            delta.backward()
            grad_stats = _selected_delta_stats(model, args)

        local_failures = []
        if logit_gap["max_abs"] > args.max_logit_abs:
            local_failures.append("logit_max_abs")
        if logit_gap["mean_abs"] > args.max_logit_mean_abs:
            local_failures.append("logit_mean_abs")
        if logit_gap["p99_abs"] > args.max_logit_p99_abs:
            local_failures.append("logit_p99_abs")
        if logit_gap["relative_l2_gap"] > args.max_logit_rel_gap:
            local_failures.append("logit_relative_l2_gap")
        if loss_abs > args.max_loss_abs:
            local_failures.append("loss_abs")
        if args.max_loss_abs_per_token >= 0.0 and loss_abs_per_token > args.max_loss_abs_per_token:
            local_failures.append("loss_abs_per_token")
        if token_count_abs > args.max_token_count_abs:
            local_failures.append("token_count_abs")
        if getattr(args, "fp8", None):
            local_failures.append("fp8_math_reference_not_implemented")
        if not args.skip_backward_check:
            if grad_stats["selected_tensors_with_grad_local"] <= 0:
                local_failures.append("selected_grad_missing")
            if grad_stats["nonfinite_selected_grad_tensors_global"] > 0:
                local_failures.append("selected_grad_nonfinite")
            if grad_stats["selected_grad_max_abs_global_max"] > args.max_selected_grad_abs:
                local_failures.append("selected_grad_max_abs")
            if grad_stats["selected_grad_relative_to_param_l2_global"] > args.max_selected_grad_rel_gap:
                local_failures.append("selected_grad_relative_to_param_l2")
            if grad_stats["selected_state_max_abs_global_max"] > args.max_selected_state_abs:
                local_failures.append("selected_state_max_abs")

        gathered_failures: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_failures, local_failures)
        failures = sorted({item for row in gathered_failures for item in row})
        status = "PASS" if not failures else "FAIL"
        layer_gaps = None
        if args.debug_layer_gaps and ref_traces is not None:
            layer_gaps = {}
            for name in sorted(ref_traces):
                if name in miles_traces:
                    layer_gaps[name] = _global_tensor_gap(miles_traces[name], ref_traces[name])
        payload = {
            "date": "2026-05-31",
            "seed": SEED,
            "status": status,
            "scope": "DeepSeek-V4 loaded 4-layer mini checkpoint full-model external reference",
            "reference": (
                "explicit PyTorch embedding + 4 DeepSeek-V4 layers + dense attention reference + "
                "EP=8 hash-routed / score-routed MoE forward + output head + SFT loss"
            ),
            "world_size": dist.get_world_size(),
            "attention_impl": args.impl,
            "checkpoint_name": Path(str(args.load)).name if getattr(args, "load", None) else None,
            "rollout_data_name": args.rollout_data.name,
            "max_samples": args.max_samples,
            "model_config": {
                "num_layers": int(model_config.num_layers),
                "hidden_size": int(model_config.hidden_size),
                "num_attention_heads": int(model_config.num_attention_heads),
                "num_moe_experts": int(model_config.num_moe_experts),
                "dsv4_mode": bool(getattr(model_config, "dsv4_mode", False)),
                "expert_parallel_size": ep_size,
                "local_experts_per_rank": int(model_config.num_moe_experts // ep_size),
                "dsv4_compress_ratios": list(model_config.dsv4_compress_ratios),
                "dsv4_hc_mult": int(model_config.dsv4_hc_mult),
                "dsv4_hc_sinkhorn_iters": int(model_config.dsv4_hc_sinkhorn_iters),
                "dsv4_n_hash_layers": int(getattr(model_config, "dsv4_n_hash_layers", 0)),
                "vocab_size": _model_vocab_size(model[0], args),
            },
            "thresholds": {
                "max_logit_abs": args.max_logit_abs,
                "max_logit_mean_abs": args.max_logit_mean_abs,
                "max_logit_p99_abs": args.max_logit_p99_abs,
                "max_logit_rel_gap": args.max_logit_rel_gap,
                "max_loss_abs": args.max_loss_abs,
                "max_loss_abs_per_token": args.max_loss_abs_per_token,
                "max_token_count_abs": args.max_token_count_abs,
                "max_selected_grad_abs": args.max_selected_grad_abs,
                "max_selected_grad_rel_gap": args.max_selected_grad_rel_gap,
                "max_selected_state_abs": args.max_selected_state_abs,
            },
            "precision_mode": {
                "miles_fp8": str(getattr(args, "fp8", None)),
                "miles_fp8_recipe": str(getattr(args, "fp8_recipe", None)),
                "reference": "explicit BF16/FP32 PyTorch math; FP8 quantization is not simulated here",
            },
            "miles_loss": float(miles_loss.detach().item()),
            "reference_loss": float(explicit_ref["loss_sum"].detach().item()),
            "loss_abs_local": loss_abs,
            "loss_abs_global_max": _all_reduce_float(loss_abs, dist.ReduceOp.MAX),
            "loss_abs_per_token_local": loss_abs_per_token,
            "loss_abs_per_token_global_max": _all_reduce_float(loss_abs_per_token, dist.ReduceOp.MAX),
            "token_count_abs_local": token_count_abs,
            "token_count_abs_global_max": _all_reduce_float(token_count_abs, dist.ReduceOp.MAX),
            "miles_loss_log": _summarize_log(loss_log),
            "reference_formula": "full_model_reference_then_sum(-log_softmax(response_logits)[target_token] * loss_mask)",
            "routing_replay": {
                "mode": "miles_router_map_replay" if args.replay_miles_routing else "independent_reference_routing",
                "description": (
                    "When enabled, the reference fixes the discrete expert mask to the Miles router map "
                    "and recomputes routing probabilities from reference scores through that fixed mask."
                ),
            },
            "tolerance_profile": args.tolerance_profile,
            "logit_gap_local": logit_gap,
            "logit_gap_global": {
                "max_abs": _all_reduce_float(logit_gap["max_abs"], dist.ReduceOp.MAX),
                "mean_abs": _all_reduce_float(logit_gap["mean_abs"], dist.ReduceOp.MAX),
                "p99_abs": _all_reduce_float(logit_gap["p99_abs"], dist.ReduceOp.MAX),
                "relative_l2_gap": _all_reduce_float(logit_gap["relative_l2_gap"], dist.ReduceOp.MAX),
            },
            "selected_backward_update_delta": grad_stats,
            "manual_update": {"rule": args.manual_update_rule, "lr": args.manual_sgd_lr},
            "failures": failures,
            "layer_gap_global": layer_gaps,
            "backward_update_check": "SKIPPED_FORWARD_ONLY" if args.skip_backward_check else "RUN",
            "boundary": (
                "This is a full loaded-checkpoint external forward/loss reference and a selected "
                "non-expert backward/update delta check. It intentionally does not reclassify real "
                "dense/sparse/tilelang backend strict parity. Full local-expert EP all-to-all "
                "backward/update remains covered by the dedicated real EP=8 MoELayer external "
                "reference, because monolithic full-model replicated-weight gradients and "
                "local-expert all-source gradients have different distributed reduction semantics."
            ),
        }
        if dist.get_rank() == 0:
            print(json.dumps(payload, indent=2, sort_keys=True))
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        dist.barrier()
        return 0 if status == "PASS" else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")
    os.environ.setdefault("MILES_DSV4_CKPT_VERSION", "0415")
    os.environ.setdefault("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    raise SystemExit(main())

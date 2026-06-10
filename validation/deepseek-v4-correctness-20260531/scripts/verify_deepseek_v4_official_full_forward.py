#!/usr/bin/env python3
"""DeepSeek-V4 官方推理实现的全量前向一致性校验。

本校验器把同一份 Megatron 分布式 checkpoint 加载到官方
DeepSeek-V4 推理模型中，在同一条 rollout 样本上执行一次全序列
prefill 前向，并计算 response token 的对数概率。它可以把这些
对数概率与由 ``verify_deepseek_v4_mini_forward_parity.py``
生成的 Miles/Megatron 前向产物进行对比。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed.checkpoint as dist_cp
import torch.nn.functional as F
from typing_extensions import override


SEED = 20260531
APPLY_APE_HOTFIX = True


class UnpicklerWrapper(pickle.Unpickler):
    @override
    def find_class(self, mod_name: str, name: str):
        class DummyClass:
            def __init__(self, *args, **kwargs):
                pass

        if mod_name.startswith("megatron") or mod_name.startswith("glm"):
            return DummyClass
        return super().find_class(mod_name, name)


pickle.Unpickler = UnpicklerWrapper


class WrappedStorageReader(dist_cp.FileSystemReader):
    @override
    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as metadata_file:
            metadata = UnpicklerWrapper(metadata_file).load()
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = dist_cp.StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


class ModelOnlyLoadPlanner(dist_cp.default_planner.DefaultLoadPlanner):
    def __init__(self, key_filter=None):
        super().__init__()
        self.key_filter = key_filter

    @override
    def set_up_planner(
        self,
        state_dict: dist_cp.metadata.STATE_DICT_TYPE,
        metadata: dist_cp.metadata.Metadata | None = None,
        is_coordinator: bool = False,
    ) -> None:
        assert metadata is not None
        for key, value in metadata.state_dict_metadata.items():
            if "optimizer" in key or "_state" in key:
                continue
            if self.key_filter is not None and not self.key_filter(key):
                continue
            if isinstance(value, dist_cp.metadata.TensorStorageMetadata):
                value = torch.empty(value.size, dtype=value.properties.dtype, device="cpu")  # type: ignore[assignment]
            state_dict[key] = value
        super().set_up_planner(state_dict, metadata, is_coordinator)


def _load_official_model_module(inference_dir: Path):
    model_py = inference_dir / "model.py"
    kernel_py = inference_dir / "kernel.py"
    if not model_py.exists() or not kernel_py.exists():
        raise FileNotFoundError(f"{inference_dir} must contain model.py and kernel.py")

    sys.path.insert(0, str(inference_dir))
    spec = importlib.util.spec_from_file_location("deepseek_v4_official_full_model", model_py)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patch_head_to_return_all_logits(official_model: Any, *, logits_mode: str) -> None:
    def get_logits(self, x: torch.Tensor):
        if logits_mode == "fp32":
            return F.linear(x.float(), self.weight)
        if logits_mode == "bf16":
            return F.linear(x.bfloat16(), self.weight.bfloat16()).float()
        raise ValueError(f"unsupported head logits mode: {logits_mode}")

    official_model.ParallelHead.get_logits = get_logits


def _patch_expert_runtime(official_model: Any, *, expert_mode: str) -> None:
    if expert_mode == "official_fp32_activation":
        return
    if expert_mode != "megatron_bf16_activation":
        raise ValueError(f"unsupported expert runtime mode: {expert_mode}")

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights.to(x.dtype) * x
        return self.w2(x.to(dtype))

    official_model.Expert.forward = forward


def _patch_gate_runtime(official_model: Any, *, router_mode: str) -> None:
    if router_mode == "official_fp32_scores":
        return
    if router_mode != "megatron_bf16_scores":
        raise ValueError(f"unsupported router runtime mode: {router_mode}")

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor | None = None):
        logits = F.linear(x.bfloat16(), self.weight.bfloat16())
        if self.score_func == "softmax":
            scores = logits.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = logits.float().sigmoid().to(logits.dtype)
        else:
            scores = F.softplus(logits.float()).sqrt().to(logits.dtype)
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices

    official_model.Gate.forward = forward


def _load_rollout_data(path: Path, max_samples: int) -> dict[str, list[Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "rollout_data" in obj:
        rollout_data = obj["rollout_data"]
    elif isinstance(obj, dict) and "samples" in obj:
        samples = obj["samples"][:max_samples]
        rollout_data = {
            "tokens": [torch.as_tensor(sample["tokens"], dtype=torch.long) for sample in samples],
            "response_lengths": [int(sample["response_length"]) for sample in samples],
            "loss_masks": [torch.as_tensor(sample["loss_mask"], dtype=torch.int32) for sample in samples],
            "sample_indices": [int(sample.get("index", i)) for i, sample in enumerate(samples)],
            "total_lengths": [int(len(sample["tokens"])) for sample in samples],
        }
    else:
        raise ValueError(f"Unsupported rollout data format: {path}")

    required = ["tokens", "response_lengths", "loss_masks", "total_lengths"]
    for key in required:
        if key not in rollout_data:
            raise KeyError(f"rollout data missing {key}")
    out: dict[str, list[Any]] = {}
    n = min(max_samples, len(rollout_data["tokens"]))
    for key, value in rollout_data.items():
        if isinstance(value, list):
            out[key] = value[:n]
        else:
            out[key] = value
    out["tokens"] = [torch.as_tensor(x, dtype=torch.long) for x in out["tokens"]]
    out["loss_masks"] = [torch.as_tensor(x, dtype=torch.int32) for x in out["loss_masks"]]
    out["response_lengths"] = [int(x) for x in out["response_lengths"]]
    out["total_lengths"] = [int(x) for x in out["total_lengths"]]
    return out


def _apply_ape_hotfix_mirror(param: torch.Tensor) -> torch.Tensor:
    assert param.shape[0] == 4
    ape = torch.chunk(param, 2, dim=-1)
    return torch.cat([ape[0], ape[1]], dim=0).view(4, -1).contiguous()


def _maybe_convert_ape(param: torch.Tensor) -> torch.Tensor:
    if APPLY_APE_HOTFIX and param.shape[0] == 4:
        return _apply_ape_hotfix_mirror(param)
    return param


def _layer_match(name: str) -> tuple[str, str] | None:
    match = re.match(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        return None
    return match.group(1), match.group(2)


def _convert_layer_param(layer_idx: str, rest: str, param: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
    prefix = f"layers.{layer_idx}"

    direct_layer = {
        "hc_attn_fn": "hc_attn_fn",
        "hc_attn_base": "hc_attn_base",
        "hc_attn_scale": "hc_attn_scale",
        "hc_ffn_fn": "hc_ffn_fn",
        "hc_ffn_base": "hc_ffn_base",
        "hc_ffn_scale": "hc_ffn_scale",
        "input_layernorm.weight": "attn_norm.weight",
        "pre_mlp_layernorm.weight": "ffn_norm.weight",
    }
    if rest in direct_layer:
        yield f"{prefix}.{direct_layer[rest]}", param
        return

    attn_prefix = "self_attention."
    if rest.startswith(attn_prefix):
        attn_rest = rest.removeprefix(attn_prefix)
        direct_attn = {
            "attn_sink": "attn.attn_sink",
            "wq_a.weight": "attn.wq_a.weight",
            "q_norm.weight": "attn.q_norm.weight",
            "wq_b.weight": "attn.wq_b.weight",
            "wkv.weight": "attn.wkv.weight",
            "kv_norm.weight": "attn.kv_norm.weight",
            "wo_a.weight": "attn.wo_a.weight",
            "wo_b.weight": "attn.wo_b.weight",
            "compressor.wkv.weight": "attn.compressor.wkv.weight",
            "compressor.wgate.weight": "attn.compressor.wgate.weight",
            "compressor.norm.weight": "attn.compressor.norm.weight",
            "indexer.linear_wq_b.weight": "attn.indexer.wq_b.weight",
            "indexer.linear_weights_proj.weight": "attn.indexer.weights_proj.weight",
            "indexer.compressor.wkv.weight": "attn.indexer.compressor.wkv.weight",
            "indexer.compressor.wgate.weight": "attn.indexer.compressor.wgate.weight",
            "indexer.compressor.norm.weight": "attn.indexer.compressor.norm.weight",
        }
        if attn_rest == "compressor.ape":
            yield f"{prefix}.attn.compressor.ape", _maybe_convert_ape(param)
            return
        if attn_rest == "indexer.compressor.ape":
            yield f"{prefix}.attn.indexer.compressor.ape", _maybe_convert_ape(param)
            return
        if attn_rest in direct_attn:
            yield f"{prefix}.{direct_attn[attn_rest]}", param
            return

    if rest == "mlp.router.weight":
        yield f"{prefix}.ffn.gate.weight", param
        return
    if rest == "mlp.router.expert_bias":
        yield f"{prefix}.ffn.gate.bias", param
        return
    if rest == "mlp.router.tid2eid":
        yield f"{prefix}.ffn.gate.tid2eid", param
        return

    if rest == "mlp.experts.experts.linear_fc1.weight":
        assert param.ndim == 3
        for expert_id in range(param.shape[0]):
            gate_weight, up_weight = param[expert_id].chunk(2, dim=0)
            yield f"{prefix}.ffn.experts.{expert_id}.w1.weight", gate_weight
            yield f"{prefix}.ffn.experts.{expert_id}.w3.weight", up_weight
        return
    if rest == "mlp.experts.experts.linear_fc2.weight":
        assert param.ndim == 3
        for expert_id in range(param.shape[0]):
            yield f"{prefix}.ffn.experts.{expert_id}.w2.weight", param[expert_id]
        return
    if rest == "mlp.shared_experts.linear_fc1.weight":
        gate_weight, up_weight = param.chunk(2, dim=0)
        yield f"{prefix}.ffn.shared_experts.w1.weight", gate_weight
        yield f"{prefix}.ffn.shared_experts.w3.weight", up_weight
        return
    if rest == "mlp.shared_experts.linear_fc2.weight":
        yield f"{prefix}.ffn.shared_experts.w2.weight", param
        return

    raise ValueError(f"Unknown layer parameter: decoder.layers.{layer_idx}.{rest}")


def _convert_megatron_to_official_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    top_level = {
        "embedding.word_embeddings.weight": "embed.weight",
        "output_layer.weight": "head.weight",
        "decoder.final_layernorm.weight": "norm.weight",
        "decoder.hc_head_params.hc_head_fn": "hc_head_fn",
        "decoder.hc_head_params.hc_head_base": "hc_head_base",
        "decoder.hc_head_params.hc_head_scale": "hc_head_scale",
    }
    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        if name in top_level:
            converted[top_level[name]] = param
            continue
        layer = _layer_match(name)
        if layer is not None:
            for converted_name, converted_param in _convert_layer_param(layer[0], layer[1], param):
                converted[converted_name] = converted_param
            continue
        raise ValueError(f"Unknown parameter name: {name}")
    return converted


def _make_checkpoint_key_filter(num_layers: int):
    def keep(key: str) -> bool:
        if "optimizer" in key or "_state" in key:
            return False
        layer = _layer_match(key)
        if layer is None:
            return True
        return int(layer[0]) < num_layers

    return keep


def _load_megatron_state(checkpoint_dir: Path, num_layers: int) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}
    dist_cp.state_dict_loader._load_state_dict(
        state_dict,
        storage_reader=WrappedStorageReader(str(checkpoint_dir)),
        planner=ModelOnlyLoadPlanner(key_filter=_make_checkpoint_key_filter(num_layers)),
        no_dist=True,
    )
    return state_dict


def _build_model_args(official_model: Any, config: dict[str, Any], max_seq_len: int, max_batch_size: int):
    rope_scaling = config.get("rope_scaling") or {}
    return official_model.ModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dtype="bf16",
        scale_fmt=None,
        expert_dtype=None,
        scale_dtype="fp32",
        vocab_size=int(config["vocab_size"]),
        dim=int(config["hidden_size"]),
        moe_inter_dim=int(config["moe_intermediate_size"]),
        n_layers=int(config["num_hidden_layers"]),
        n_hash_layers=int(config.get("num_hash_layers", 0)),
        n_mtp_layers=0,
        n_heads=int(config["num_attention_heads"]),
        n_routed_experts=int(config["n_routed_experts"]),
        n_shared_experts=int(config.get("n_shared_experts", 1)),
        n_activated_experts=int(config["num_experts_per_tok"]),
        score_func=str(config.get("scoring_func", "sqrtsoftplus")),
        route_scale=float(config.get("routed_scaling_factor", 1.0)),
        swiglu_limit=float(config.get("swiglu_limit", 0.0)),
        q_lora_rank=int(config["q_lora_rank"]),
        head_dim=int(config["head_dim"]),
        rope_head_dim=int(config["qk_rope_head_dim"]),
        norm_eps=float(config["rms_norm_eps"]),
        o_groups=int(config["o_groups"]),
        o_lora_rank=int(config["o_lora_rank"]),
        window_size=int(config["sliding_window"]),
        compress_ratios=tuple(int(x) for x in config["compress_ratios"][: int(config["num_hidden_layers"])]),
        compress_rope_theta=float(config.get("compress_rope_theta", 40000.0)),
        original_seq_len=int(rope_scaling.get("original_max_position_embeddings", 0)),
        rope_theta=float(config.get("rope_theta", 10000.0)),
        rope_factor=float(rope_scaling.get("factor", 1.0)),
        beta_fast=int(rope_scaling.get("beta_fast", 32)),
        beta_slow=int(rope_scaling.get("beta_slow", 1)),
        index_n_heads=int(config["index_n_heads"]),
        index_head_dim=int(config["index_head_dim"]),
        index_topk=int(config["index_topk"]),
        hc_mult=int(config["hc_mult"]),
        hc_sinkhorn_iters=int(config["hc_sinkhorn_iters"]),
        hc_eps=float(config["hc_eps"]),
    )


def _tensor_sha256(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        data = tensor.detach().cpu().contiguous().float()
        h.update(str(tuple(data.shape)).encode("ascii"))
        h.update(data.numpy().tobytes())
    return h.hexdigest()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    x = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(torch.isfinite(x).all().item()),
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "std": float(x.std(unbiased=False).item()) if x.numel() else 0.0,
        "max_abs": float(x.abs().max().item()) if x.numel() else 0.0,
    }


def _summarize(log_probs: list[torch.Tensor]) -> dict[str, Any]:
    flat = torch.cat([x.detach().cpu().float().flatten() for x in log_probs])
    return {
        "num_samples": len(log_probs),
        "num_tokens": int(flat.numel()),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "sha256": _tensor_sha256(log_probs),
    }


def _named_tensors(value: Any, prefix: str = "out") -> list[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        return [(prefix, value)]
    out: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, (tuple, list)):
        for idx, item in enumerate(value):
            out.extend(_named_tensors(item, f"{prefix}{idx}"))
    elif isinstance(value, dict):
        for key, item in value.items():
            safe_key = str(key).replace(".", "_")
            out.extend(_named_tensors(item, f"{prefix}_{safe_key}"))
    return out


def _canonical_trace_tensor(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach().float().cpu()
    if out.ndim >= 2 and out.shape[0] == 1:
        dims = [1, 0, *range(2, out.ndim)]
        out = out.permute(*dims).contiguous()
    return out


def _install_trace_hooks(module: torch.nn.Module, traces: dict[str, torch.Tensor]) -> list[Any]:
    patterns = (
        re.compile(r"embed$"),
        re.compile(r"layers\.\d+$"),
        re.compile(r"layers\.\d+\.attn_norm$"),
        re.compile(r"layers\.\d+\.attn$"),
        re.compile(r"layers\.\d+\.attn\.wq_a$"),
        re.compile(r"layers\.\d+\.attn\.q_norm$"),
        re.compile(r"layers\.\d+\.attn\.wq_b$"),
        re.compile(r"layers\.\d+\.attn\.wkv$"),
        re.compile(r"layers\.\d+\.attn\.kv_norm$"),
        re.compile(r"layers\.\d+\.attn\.wo_b$"),
        re.compile(r"layers\.\d+\.ffn_norm$"),
        re.compile(r"layers\.\d+\.ffn$"),
        re.compile(r"layers\.\d+\.ffn\.gate$"),
        re.compile(r"layers\.\d+\.ffn\.shared_experts$"),
        re.compile(r"norm$"),
    )
    handles = []
    for name, submodule in module.named_modules():
        if not any(pattern.fullmatch(name) for pattern in patterns):
            continue

        def hook(_module, _inputs, output, *, trace_name=name):
            tensors = _named_tensors(output)
            if not tensors:
                return
            traces[trace_name] = _canonical_trace_tensor(tensors[0][1])
            if len(tensors) > 1:
                for suffix, tensor in tensors:
                    traces[f"{trace_name}.{suffix}"] = _canonical_trace_tensor(tensor)

        handles.append(submodule.register_forward_hook(hook))
    return handles


class _RestoreCallable:
    def __init__(self, restore):
        self._restore = restore

    def remove(self) -> None:
        self._restore()


def _install_attention_internal_trace_hooks(
    official_model: Any,
    module: torch.nn.Module,
    traces: dict[str, torch.Tensor],
) -> list[Any]:
    """在不修改官方模型代码的前提下追踪官方 attention 的内部 tensor。"""
    handles: list[Any] = []

    original_sparse_attn = official_model.sparse_attn
    call_idx = 0

    def sparse_attn_wrapper(q, kv, attn_sink, topk_idxs, softmax_scale):
        nonlocal call_idx
        prefix = f"layers.{call_idx}.attn"
        traces[f"{prefix}.q_after_rope"] = _canonical_trace_tensor(q)
        traces[f"{prefix}.kv_after_rope_qat"] = _canonical_trace_tensor(kv)
        traces[f"{prefix}.attn_sink"] = attn_sink.detach().float().cpu().contiguous()
        traces[f"{prefix}.topk_idxs"] = topk_idxs.detach().cpu().int().contiguous()
        out = original_sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
        traces[f"{prefix}.attention_core"] = _canonical_trace_tensor(out)
        call_idx += 1
        return out

    official_model.sparse_attn = sparse_attn_wrapper
    handles.append(_RestoreCallable(lambda: setattr(official_model, "sparse_attn", original_sparse_attn)))

    for name, submodule in module.named_modules():
        if not re.fullmatch(r"layers\.\d+\.attn\.wo_b", name):
            continue

        def pre_hook(_module, inputs, *, trace_name=name):
            if inputs:
                traces[f"{trace_name}.input"] = _canonical_trace_tensor(inputs[0])

        handles.append(submodule.register_forward_pre_hook(pre_hook))

    return handles


def _compute_response_log_probs(
    official: torch.nn.Module,
    rollout_data: dict[str, list[Any]],
    temperature: float,
    pad_to_seq_len: int | None,
) -> list[torch.Tensor]:
    log_probs: list[torch.Tensor] = []
    device = torch.device("cuda", torch.cuda.current_device())
    with torch.inference_mode():
        for tokens_cpu, total_length, response_length in zip(
            rollout_data["tokens"],
            rollout_data["total_lengths"],
            rollout_data["response_lengths"],
            strict=True,
        ):
            tokens_cpu = torch.as_tensor(tokens_cpu, dtype=torch.long)[:total_length]
            if pad_to_seq_len is not None and pad_to_seq_len > tokens_cpu.numel():
                tokens_cpu = F.pad(tokens_cpu, (0, pad_to_seq_len - tokens_cpu.numel()), value=0)
            tokens = tokens_cpu.to(device=device)
            logits = official(tokens.unsqueeze(0), 0).float().div(float(temperature))
            assert logits.shape[1] == tokens.numel(), f"{logits.shape=} {tokens.numel()=}"
            start = total_length - response_length
            response_logits = logits[0, start - 1 : total_length - 1]
            response_tokens = tokens[start:total_length]
            gathered = torch.log_softmax(response_logits, dim=-1).gather(-1, response_tokens.unsqueeze(-1))
            log_probs.append(gathered.squeeze(-1).detach().cpu().float())
    return log_probs


def _compare_log_probs(
    official_logs: list[torch.Tensor],
    miles_output: Path,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    miles_payload = torch.load(miles_output, map_location="cpu", weights_only=False)
    miles_logs = miles_payload["log_probs"]
    assert len(official_logs) == len(miles_logs)

    all_diffs: list[torch.Tensor] = []
    sample_summaries: list[dict[str, Any]] = []
    mismatch_count = 0
    total_count = 0
    rel_num = 0.0
    rel_den = 0.0
    for idx, (official, miles) in enumerate(zip(official_logs, miles_logs, strict=True)):
        official = official.float()
        miles = miles.float()
        assert official.shape == miles.shape, f"{official.shape=} {miles.shape=}"
        diff = (official - miles).abs().flatten()
        close = torch.isclose(official, miles, rtol=rtol, atol=atol)
        all_diffs.append(diff)
        mismatch_count += int((~close).sum().item())
        total_count += int(close.numel())
        rel_num += float((official * miles).sum().item())
        rel_den += float((official.square().sum() + miles.square().sum()).item())
        sample_summaries.append(
            {
                "sample": idx,
                "tokens": int(close.numel()),
                "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
                "mismatches": int((~close).sum().item()),
            }
        )

    diffs = torch.cat(all_diffs) if all_diffs else torch.empty(0)
    relative_l2_gap = 0.0 if rel_den == 0.0 else 1.0 - 2.0 * rel_num / rel_den
    return {
        "label": "official_inference_vs_miles",
        "miles_attention_impl": miles_payload.get("attention_impl"),
        "num_samples": len(official_logs),
        "num_tokens": int(total_count),
        "max_abs": float(diffs.max().item()) if diffs.numel() else 0.0,
        "mean_abs": float(diffs.mean().item()) if diffs.numel() else 0.0,
        "p50_abs": float(diffs.quantile(0.50).item()) if diffs.numel() else 0.0,
        "p95_abs": float(diffs.quantile(0.95).item()) if diffs.numel() else 0.0,
        "p99_abs": float(diffs.quantile(0.99).item()) if diffs.numel() else 0.0,
        "relative_l2_gap": float(relative_l2_gap),
        "mismatches": int(mismatch_count),
        "rtol": float(rtol),
        "atol": float(atol),
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "samples": sample_summaries,
    }


def _copy_safe_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copyfile(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-inference-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hf-config", type=Path, required=True)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--miles-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--save-logprobs", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--pad-input-to-max-seq-len", action="store_true")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--disable-ape-hotfix", action="store_true")
    parser.add_argument("--head-logits-mode", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--expert-runtime-mode",
        choices=["official_fp32_activation", "megatron_bf16_activation"],
        default="official_fp32_activation",
    )
    parser.add_argument(
        "--router-runtime-mode",
        choices=["official_fp32_scores", "megatron_bf16_scores"],
        default="official_fp32_scores",
    )
    args = parser.parse_args()

    global APPLY_APE_HOTFIX
    APPLY_APE_HOTFIX = not args.disable_ape_hotfix

    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_device(0)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    rollout_data = _load_rollout_data(args.rollout_data, args.max_samples)
    max_total_length = max(int(x) for x in rollout_data["total_lengths"])
    max_seq_len = int(args.max_seq_len or max_total_length)
    if max_seq_len < max_total_length:
        raise ValueError(f"--max-seq-len {max_seq_len} is smaller than rollout max length {max_total_length}")

    official_model = _load_official_model_module(args.official_inference_dir)
    _patch_head_to_return_all_logits(official_model, logits_mode=args.head_logits_mode)
    _patch_expert_runtime(official_model, expert_mode=args.expert_runtime_mode)
    _patch_gate_runtime(official_model, router_mode=args.router_runtime_mode)
    official_model.world_size = 1
    official_model.rank = 0
    official_model.default_dtype = torch.bfloat16
    official_model.scale_fmt = None
    official_model.scale_dtype = torch.float32

    config = json.loads(args.hf_config.read_text())
    model_args = _build_model_args(
        official_model,
        config,
        max_seq_len=max_seq_len,
        max_batch_size=args.max_samples,
    )

    print("loading Megatron distributed checkpoint", flush=True)
    megatron_state = _load_megatron_state(args.checkpoint_dir, num_layers=int(config["num_hidden_layers"]))
    print(f"loaded {len(megatron_state)} Megatron tensors", flush=True)
    official_state = _convert_megatron_to_official_state(megatron_state)
    print(f"converted to {len(official_state)} official tensors", flush=True)

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    official = official_model.Transformer(model_args).eval()
    missing, unexpected = official.load_state_dict(official_state, strict=True)
    assert not missing, missing
    assert not unexpected, unexpected
    del megatron_state
    del official_state
    torch.cuda.empty_cache()

    traces: dict[str, torch.Tensor] = {}
    handles = []
    if args.trace_output is not None:
        handles.extend(_install_trace_hooks(official, traces))
        handles.extend(_install_attention_internal_trace_hooks(official_model, official, traces))
    official_logs = _compute_response_log_probs(
        official,
        rollout_data,
        args.temperature,
        max_seq_len if args.pad_input_to_max_seq_len else None,
    )
    for handle in handles:
        handle.remove()
    payload: dict[str, Any] = {
        "seed": SEED,
        "reference": "DeepSeek-V4 official inference Transformer",
        "checkpoint": "4-layer DeepSeek-V4 mini checkpoint",
        "rollout": {
            "num_samples": len(official_logs),
            "num_tokens": int(sum(x.numel() for x in official_logs)),
            "max_total_length": int(max_total_length),
        },
        "runtime": {
            "CUDA_DEVICE_MAX_CONNECTIONS": os.getenv("CUDA_DEVICE_MAX_CONNECTIONS"),
            "MEGATRON_USE_KV_QAT": os.getenv("MEGATRON_USE_KV_QAT"),
            "PYTORCH_ALLOC_CONF": os.getenv("PYTORCH_ALLOC_CONF"),
        },
        "conversion": {
            "ape_hotfix_enabled": APPLY_APE_HOTFIX,
        },
        "runtime_variant": {
            "head_logits_mode": args.head_logits_mode,
            "expert_runtime_mode": args.expert_runtime_mode,
            "router_runtime_mode": args.router_runtime_mode,
        },
        "official_summary": _summarize(official_logs),
    }
    if args.trace_output is not None:
        payload["trace_summaries"] = {name: _tensor_summary(tensor) for name, tensor in sorted(traces.items())}
    if args.miles_output is not None:
        payload["comparison"] = _compare_log_probs(official_logs, args.miles_output, args.rtol, args.atol)

    if args.save_logprobs is not None:
        args.save_logprobs.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"log_probs": official_logs, **payload}, args.save_logprobs)

    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trace_tensors": traces, "log_probs": official_logs, **payload}, args.trace_output)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json_output}", flush=True)

    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

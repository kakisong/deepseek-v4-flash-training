#!/usr/bin/env python3
"""DeepSeek-V4 mini checkpoint 激活值 replay 探针。

本校验器先记录某一后端的逐 rank 激活值,然后在选定的模块边界处把这些
激活值 replay 到另一后端。它是面向 Miles/Megatron mini checkpoint 的
全模型漂移定位探针;不能替代外部参考实现的一致性测试。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_model

import miles_plugins.mbridge  # noqa: F401
from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model import forward_only
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state
from miles.backends.training_utils.data import DataIterator
from miles.backends.training_utils.loss import get_log_probs_and_entropy
from miles.utils.logging_utils import configure_logger


SEED = 20260531


def add_replay_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--rollout-data", type=Path)
    parser.add_argument("--parity-output", type=Path)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-a", type=Path)
    parser.add_argument("--compare-b", type=Path)
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument("--compare-label", type=str, default="")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--qkv-format", choices=["thd", "bshd"], default="thd")
    parser.add_argument("--data-pad-size-multiplier", type=int, default=128)
    parser.add_argument("--log-probs-chunk-size", type=int, default=-1)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--allgather-cp", action="store_true")
    parser.add_argument("--true-on-policy-mode", action="store_true")
    parser.add_argument("--use-rollout-entropy", action="store_true")
    parser.add_argument("--use-dynamic-batch-size", action="store_true")
    parser.add_argument("--routing-replay-stage", choices=["off", "record", "replay_forward"], default="off")
    parser.add_argument("--routing-replay-file", type=Path)
    parser.add_argument("--activation-replay-stage", choices=["off", "record", "replay_forward"], default="off")
    parser.add_argument("--activation-replay-file", type=Path)
    parser.add_argument("--activation-record-modules", nargs="*", default=None)
    parser.add_argument("--activation-replay-modules", nargs="*", default=None)
    return parser


def _default_activation_modules(args: Any) -> list[str]:
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    modules = [f"module.decoder.layers.{idx}" for idx in range(num_layers)]
    modules.append("module.decoder.final_layernorm")
    return modules


def _ranked_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def _init_distributed() -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )


def _parse_megatron_args():
    args = parse_args(extra_args_provider=add_replay_args)
    args = set_default_megatron_args(args)
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = True
    args.__dict__.setdefault("custom_megatron_before_log_prob_hook_path", None)
    args.save_interval = getattr(args, "save_interval", None) or 1
    args.micro_batch_size = getattr(args, "micro_batch_size", None) or args.max_samples
    args.global_batch_size = getattr(args, "global_batch_size", None) or args.micro_batch_size
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    validate_args(args)
    args.variable_seq_lengths = True
    return args


def _load_rollout_data(path: Path, max_samples: int) -> dict[str, list[Any]]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "rollout_data" in obj:
        rollout_data = obj["rollout_data"]
    elif isinstance(obj, dict) and "samples" in obj:
        samples = obj["samples"]
        rollout_data = {
            "tokens": [torch.as_tensor(sample.tokens, dtype=torch.long) for sample in samples],
            "response_lengths": [int(sample.response_length) for sample in samples],
            "loss_masks": [torch.as_tensor(sample.loss_mask, dtype=torch.int32) for sample in samples],
            "rewards": [float(getattr(sample, "reward", 0.0)) for sample in samples],
            "truncated": [int(getattr(sample, "truncated", 0)) for sample in samples],
            "sample_indices": [int(getattr(sample, "index", i)) for i, sample in enumerate(samples)],
            "total_lengths": [int(len(sample.tokens)) for sample in samples],
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
        out[key] = value[:n] if isinstance(value, list) else value
    return out


def _enable_routing_replay(stage: str):
    from miles.utils.replay_base import routing_replay_manager

    routing_replay_manager.enabled = stage != "off"
    routing_replay_manager.stage = "fallthrough" if stage == "off" else stage
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    return routing_replay_manager


def _load_routing_replay(manager, replay_file: Path, rank: int) -> None:
    payload = torch.load(_ranked_path(replay_file, rank), map_location="cpu", weights_only=False)
    saved = payload["top_indices_by_replay"]
    if len(saved) != len(manager.replays):
        raise ValueError(f"rank {rank}: replay count mismatch, saved={len(saved)}, current={len(manager.replays)}")
    for replay, tensors in zip(manager.replays, saved, strict=True):
        replay.top_indices_list = [tensor.cpu() for tensor in tensors]
        replay.forward_index = 0
        replay.backward_index = 0


def _save_routing_replay(manager, replay_file: Path, rank: int) -> dict[str, Any]:
    payload = {
        "rank": rank,
        "top_indices_by_replay": [
            [tensor.detach().cpu() for tensor in replay.top_indices_list] for replay in manager.replays
        ],
        "num_replays": len(manager.replays),
        "num_recorded_tensors": sum(len(replay.top_indices_list) for replay in manager.replays),
    }
    out = _ranked_path(replay_file, rank)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return {k: v for k, v in payload.items() if k != "top_indices_by_replay"}


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _replace_first_tensor(value: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(value):
        return replacement
    if isinstance(value, tuple):
        items = list(value)
        for idx, item in enumerate(items):
            replaced = _replace_first_tensor(item, replacement)
            if replaced is not item:
                items[idx] = replaced
                return tuple(items)
        return value
    if isinstance(value, list):
        items = list(value)
        for idx, item in enumerate(items):
            replaced = _replace_first_tensor(item, replacement)
            if replaced is not item:
                items[idx] = replaced
                return items
        return value
    if isinstance(value, dict):
        out = dict(value)
        for key, item in value.items():
            replaced = _replace_first_tensor(item, replacement)
            if replaced is not item:
                out[key] = replaced
                return out
        return value
    return value


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


def _install_activation_record_hooks(
    module: torch.nn.Module,
    module_names: set[str],
    activations: dict[str, torch.Tensor],
) -> list[Any]:
    handles = []
    for name, submodule in module.named_modules():
        if name not in module_names:
            continue

        def hook(_module, _inputs, output, *, trace_name=name):
            tensor = _first_tensor(output)
            if tensor is not None:
                activations[trace_name] = tensor.detach().cpu()

        handles.append(submodule.register_forward_hook(hook))
    return handles


def _install_activation_replay_hooks(
    module: torch.nn.Module,
    module_names: list[str],
    activations: dict[str, torch.Tensor],
    replay_stats: dict[str, Any],
) -> list[Any]:
    handles = []
    missing = [name for name in module_names if name not in activations]
    if missing:
        raise KeyError(f"activation replay file missing modules: {missing}")

    for name, submodule in module.named_modules():
        if name not in module_names:
            continue

        def hook(_module, _inputs, output, *, trace_name=name):
            current = _first_tensor(output)
            if current is None:
                raise TypeError(f"module {trace_name} produced no tensor output")
            saved = activations[trace_name].to(device=current.device, dtype=current.dtype)
            if tuple(saved.shape) != tuple(current.shape):
                raise ValueError(
                    f"module {trace_name} shape mismatch: saved={tuple(saved.shape)}, current={tuple(current.shape)}"
                )
            diff = (current.detach().float() - saved.detach().float()).abs()
            replay_stats[trace_name] = {
                "shape": list(current.shape),
                "dtype": str(current.dtype),
                "pre_replay_max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                "pre_replay_mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
                "pre_replay_nonzero_abs_count": int((diff != 0).sum().item()),
            }
            return _replace_first_tensor(output, saved)

        handles.append(submodule.register_forward_hook(hook))
    return handles


def _load_activation_replay(replay_file: Path, rank: int) -> dict[str, torch.Tensor]:
    payload = torch.load(_ranked_path(replay_file, rank), map_location="cpu", weights_only=False)
    return payload["activations"]


def _save_activation_replay(
    replay_file: Path,
    rank: int,
    modules: list[str],
    activations: dict[str, torch.Tensor],
) -> dict[str, Any]:
    payload = {
        "rank": rank,
        "modules": modules,
        "activations": {name: tensor.detach().cpu() for name, tensor in activations.items()},
        "activation_summaries": {
            name: _tensor_summary(tensor) for name, tensor in sorted(activations.items())
        },
    }
    out = _ranked_path(replay_file, rank)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return {
        "rank": rank,
        "modules": modules,
        "num_recorded_modules": len(activations),
        "activation_summaries": payload["activation_summaries"],
    }


def _tensor_sha256(tensors: list[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for tensor in tensors:
        data = tensor.detach().cpu().contiguous().float()
        h.update(str(tuple(data.shape)).encode("ascii"))
        h.update(data.numpy().tobytes())
    return h.hexdigest()


def _summarize_log_probs(log_probs: list[torch.Tensor]) -> dict[str, Any]:
    flat = torch.cat([x.detach().cpu().float().flatten() for x in log_probs])
    return {
        "num_samples": len(log_probs),
        "num_tokens": int(flat.numel()),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "min": float(flat.min().item()) if flat.numel() else 0.0,
        "max": float(flat.max().item()) if flat.numel() else 0.0,
        "sha256": _tensor_sha256(log_probs),
    }


def run_forward() -> int:
    configure_logger()
    _init_distributed()
    args = _parse_megatron_args()
    assert args.rollout_data is not None, "--rollout-data is required"
    assert args.parity_output is not None, "--parity-output is required"
    assert args.load is not None, "--load is required"
    if args.routing_replay_stage != "off":
        assert args.routing_replay_file is not None, "--routing-replay-file is required"
    if args.activation_replay_stage != "off":
        assert args.activation_replay_file is not None, "--activation-replay-file is required"

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    routing_replay_manager = _enable_routing_replay(args.routing_replay_stage)
    init(args)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    if args.routing_replay_stage == "replay_forward":
        _load_routing_replay(routing_replay_manager, args.routing_replay_file, dist.get_rank())
    for module in model:
        module.eval()

    record_modules = args.activation_record_modules or _default_activation_modules(args)
    replay_modules = args.activation_replay_modules or []
    activations: dict[str, torch.Tensor] = {}
    replay_stats: dict[str, Any] = {}
    handles = []
    if args.activation_replay_stage == "record":
        handles.extend(_install_activation_record_hooks(model[0], set(record_modules), activations))
    elif args.activation_replay_stage == "replay_forward":
        saved_activations = _load_activation_replay(args.activation_replay_file, dist.get_rank())
        handles.extend(_install_activation_replay_hooks(model[0], replay_modules, saved_activations, replay_stats))

    parallel_state = create_megatron_parallel_state(model)
    rollout_data = _load_rollout_data(args.rollout_data, args.max_samples)
    device = torch.device("cuda", torch.cuda.current_device())
    for key in ("tokens", "loss_masks"):
        rollout_data[key] = [tensor.to(device=device) for tensor in rollout_data[key]]
    data_iterator = [DataIterator(rollout_data, micro_batch_size=len(rollout_data["tokens"]))]
    with torch.no_grad():
        result = forward_only(
            get_log_probs_and_entropy,
            args,
            model,
            data_iterator,
            [1],
            parallel_state,
        )
    for handle in handles:
        handle.remove()

    routing_record_summary = None
    if args.routing_replay_stage == "record":
        routing_record_summary = _save_routing_replay(routing_replay_manager, args.routing_replay_file, dist.get_rank())
    activation_record_summary = None
    if args.activation_replay_stage == "record":
        activation_record_summary = _save_activation_replay(
            args.activation_replay_file,
            dist.get_rank(),
            record_modules,
            activations,
        )

    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0 and parallel_state.dp_cp_rank == 0:
        log_probs = [tensor.detach().cpu().float() for tensor in result["log_probs"]]
        payload = {
            "seed": SEED,
            "attention_impl": os.getenv("MEGATRON_SPARSE_ATTN_IMPL", "tilelang"),
            "runtime": {
                "deterministic_mode": bool(getattr(args, "deterministic_mode", False)),
                "NCCL_ALGO": os.getenv("NCCL_ALGO"),
                "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                "CUDA_DEVICE_MAX_CONNECTIONS": os.getenv("CUDA_DEVICE_MAX_CONNECTIONS"),
                "MEGATRON_USE_KV_QAT": os.getenv("MEGATRON_USE_KV_QAT"),
            },
            "rollout_data_name": args.rollout_data.name,
            "load_name": Path(args.load).name,
            "routing_replay": {
                "stage": args.routing_replay_stage,
                "file_name": args.routing_replay_file.name if args.routing_replay_file is not None else None,
                "num_replays": len(routing_replay_manager.replays),
                "num_recorded_tensors": sum(len(replay.top_indices_list) for replay in routing_replay_manager.replays),
                "record_summary": routing_record_summary,
            },
            "activation_replay": {
                "stage": args.activation_replay_stage,
                "file_name": args.activation_replay_file.name if args.activation_replay_file is not None else None,
                "record_modules": record_modules if args.activation_replay_stage == "record" else [],
                "replay_modules": replay_modules if args.activation_replay_stage == "replay_forward" else [],
                "record_summary": activation_record_summary,
                "replay_stats": replay_stats,
            },
            "summary": _summarize_log_probs(log_probs),
            "log_probs": log_probs,
        }
        args.parity_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.parity_output)
        print(json.dumps({k: v for k, v in payload.items() if k != "log_probs"}, indent=2, sort_keys=True))
        print(f"wrote {args.parity_output}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


def run_compare() -> int:
    parser = argparse.ArgumentParser()
    add_replay_args(parser)
    args, _ = parser.parse_known_args()
    assert args.compare_a is not None
    assert args.compare_b is not None
    a = torch.load(args.compare_a, map_location="cpu", weights_only=False)
    b = torch.load(args.compare_b, map_location="cpu", weights_only=False)
    a_logs = a["log_probs"]
    b_logs = b["log_probs"]
    assert len(a_logs) == len(b_logs)
    sample_summaries: list[dict[str, Any]] = []
    all_diffs: list[torch.Tensor] = []
    mismatch_count = 0
    total_count = 0
    rel_num = 0.0
    rel_den = 0.0
    for idx, (x, y) in enumerate(zip(a_logs, b_logs, strict=True)):
        x = x.float()
        y = y.float()
        diff = (x - y).abs().flatten()
        close = torch.isclose(x, y, rtol=args.rtol, atol=args.atol)
        mismatch_count += int((~close).sum().item())
        total_count += int(close.numel())
        all_diffs.append(diff)
        rel_num += float((x * y).sum().item())
        rel_den += float((x.square().sum() + y.square().sum()).item())
        sample_summaries.append(
            {
                "sample": idx,
                "tokens": int(x.numel()),
                "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
                "mismatches": int((~close).sum().item()),
            }
        )
    diffs = torch.cat(all_diffs) if all_diffs else torch.empty(0)
    rel = 0.0 if rel_den == 0.0 else 1.0 - 2.0 * rel_num / rel_den
    summary = {
        "label": args.compare_label,
        "a_impl": a.get("attention_impl"),
        "b_impl": b.get("attention_impl"),
        "a_routing_replay": a.get("routing_replay"),
        "b_routing_replay": b.get("routing_replay"),
        "a_activation_replay": a.get("activation_replay"),
        "b_activation_replay": b.get("activation_replay"),
        "num_samples": len(a_logs),
        "num_tokens": int(total_count),
        "max_abs": float(diffs.max().item()) if diffs.numel() else 0.0,
        "mean_abs": float(diffs.mean().item()) if diffs.numel() else 0.0,
        "p50_abs": float(diffs.quantile(0.50).item()) if diffs.numel() else 0.0,
        "p95_abs": float(diffs.quantile(0.95).item()) if diffs.numel() else 0.0,
        "p99_abs": float(diffs.quantile(0.99).item()) if diffs.numel() else 0.0,
        "relative_l2_gap": rel,
        "mismatches": mismatch_count,
        "rtol": args.rtol,
        "atol": args.atol,
        "status": "PASS" if mismatch_count == 0 else "FAIL",
        "samples": sample_summaries,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.compare_output is not None:
        args.compare_output.parent.mkdir(parents=True, exist_ok=True)
        args.compare_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if mismatch_count:
        raise AssertionError(f"{mismatch_count}/{total_count} log-probs exceeded rtol={args.rtol}, atol={args.atol}")
    return 0


def main() -> int:
    if "--compare" in os.sys.argv:
        return run_compare()
    return run_forward()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

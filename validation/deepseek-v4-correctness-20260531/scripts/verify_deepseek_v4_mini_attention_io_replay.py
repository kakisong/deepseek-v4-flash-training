#!/usr/bin/env python3
"""DeepSeek-V4 mini checkpoint 的 attention 输入/输出 replay 探针。

本校验器记录每一层 dense attention 的输入与输出，随后将 dense 输入
replay 到另一个 attention 后端中，并把计算得到的输出与 dense 结果比较。
这样可以把 attention 后端的数学运算与上游状态漂移、下游 logprob 放大
效应隔离开来。
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


def add_attention_io_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--parity-output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1)
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
    parser.add_argument("--attention-io-stage", choices=["record", "replay_inputs"], required=True)
    parser.add_argument("--attention-io-file", type=Path, required=True)
    parser.add_argument("--attention-modules", nargs="*", default=None)
    return parser


def _default_attention_modules(args: Any) -> list[str]:
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    return [f"module.decoder.layers.{idx}.self_attention" for idx in range(num_layers)]


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
    args = parse_args(extra_args_provider=add_attention_io_args)
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


def _replace_first_tensor(value: Any, replacement: torch.Tensor) -> tuple[Any, bool]:
    if torch.is_tensor(value):
        return replacement, True
    if isinstance(value, tuple):
        items = list(value)
        for idx, item in enumerate(items):
            replaced, did_replace = _replace_first_tensor(item, replacement)
            if did_replace:
                items[idx] = replaced
                return tuple(items), True
        return value, False
    if isinstance(value, list):
        items = list(value)
        for idx, item in enumerate(items):
            replaced, did_replace = _replace_first_tensor(item, replacement)
            if did_replace:
                items[idx] = replaced
                return items, True
        return value, False
    if isinstance(value, dict):
        out = dict(value)
        for key, item in value.items():
            replaced, did_replace = _replace_first_tensor(item, replacement)
            if did_replace:
                out[key] = replaced
                return out, True
        return value, False
    return value, False


def _compare_tensors(current: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    x = current.detach().float()
    y = reference.detach().to(device=current.device, dtype=current.dtype).float()
    diff = (x - y).abs()
    return {
        "shape": list(current.shape),
        "dtype": str(current.dtype),
        "reference_dtype": str(reference.dtype),
        "finite": bool(torch.isfinite(x).all().item() and torch.isfinite(y).all().item()),
        "nonzero_abs_count": int((diff != 0).sum().item()) if diff.numel() else 0,
        "exact_equal": bool(torch.equal(current.detach().cpu(), reference.detach().cpu().to(dtype=current.dtype))),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "p95_abs": float(diff.flatten().quantile(0.95).item()) if diff.numel() else 0.0,
        "p99_abs": float(diff.flatten().quantile(0.99).item()) if diff.numel() else 0.0,
    }


def _install_record_hooks(
    module: torch.nn.Module,
    module_names: set[str],
    records: dict[str, dict[str, torch.Tensor]],
) -> list[Any]:
    handles = []
    for name, submodule in module.named_modules():
        if name not in module_names:
            continue
        records.setdefault(name, {})

        def pre_hook(_module, inputs, kwargs, *, trace_name=name):
            tensor = _first_tensor((inputs, kwargs))
            if tensor is not None:
                records[trace_name]["input"] = tensor.detach().cpu()
            return None

        def post_hook(_module, _inputs, output, *, trace_name=name):
            tensor = _first_tensor(output)
            if tensor is not None:
                records[trace_name]["output"] = tensor.detach().cpu()

        handles.append(submodule.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(submodule.register_forward_hook(post_hook))
    return handles


def _install_replay_hooks(
    module: torch.nn.Module,
    module_names: list[str],
    records: dict[str, dict[str, torch.Tensor]],
    stats: dict[str, dict[str, Any]],
) -> list[Any]:
    handles = []
    missing = [name for name in module_names if name not in records or "input" not in records[name] or "output" not in records[name]]
    if missing:
        raise KeyError(f"attention IO replay file missing modules: {missing}")
    for name, submodule in module.named_modules():
        if name not in module_names:
            continue
        stats.setdefault(name, {})

        def pre_hook(_module, inputs, kwargs, *, trace_name=name):
            current = _first_tensor((inputs, kwargs))
            if current is None:
                raise TypeError(f"module {trace_name} produced no tensor input")
            saved = records[trace_name]["input"].to(device=current.device, dtype=current.dtype)
            stats[trace_name]["input_pre_replay"] = _compare_tensors(current, records[trace_name]["input"])
            replaced_inputs, did_replace = _replace_first_tensor(inputs, saved)
            if did_replace:
                return replaced_inputs, kwargs
            replaced_kwargs, did_replace = _replace_first_tensor(kwargs, saved)
            if not did_replace:
                raise TypeError(f"module {trace_name} input replacement failed")
            return inputs, replaced_kwargs

        def post_hook(_module, _inputs, output, *, trace_name=name):
            current = _first_tensor(output)
            if current is None:
                raise TypeError(f"module {trace_name} produced no tensor output")
            stats[trace_name]["output_from_replayed_input"] = _compare_tensors(
                current,
                records[trace_name]["output"],
            )

        handles.append(submodule.register_forward_pre_hook(pre_hook, with_kwargs=True))
        handles.append(submodule.register_forward_hook(post_hook))
    return handles


def _save_attention_io(path: Path, rank: int, modules: list[str], records: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    payload = {
        "rank": rank,
        "modules": modules,
        "records": records,
    }
    out = _ranked_path(path, rank)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    return {
        "rank": rank,
        "modules": modules,
        "num_recorded_modules": len(records),
        "recorded_keys": {name: sorted(value) for name, value in sorted(records.items())},
    }


def _load_attention_io(path: Path, rank: int) -> dict[str, dict[str, torch.Tensor]]:
    payload = torch.load(_ranked_path(path, rank), map_location="cpu", weights_only=False)
    return payload["records"]


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


def main() -> int:
    configure_logger()
    _init_distributed()
    args = _parse_megatron_args()
    assert args.load is not None, "--load is required"
    if args.routing_replay_stage != "off":
        assert args.routing_replay_file is not None, "--routing-replay-file is required"

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

    attention_modules = args.attention_modules or _default_attention_modules(args)
    records: dict[str, dict[str, torch.Tensor]] = {}
    replay_stats: dict[str, dict[str, Any]] = {}
    if args.attention_io_stage == "record":
        handles = _install_record_hooks(model[0], set(attention_modules), records)
    else:
        records = _load_attention_io(args.attention_io_file, dist.get_rank())
        handles = _install_replay_hooks(model[0], attention_modules, records, replay_stats)

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
    attention_record_summary = None
    if args.attention_io_stage == "record":
        attention_record_summary = _save_attention_io(
            args.attention_io_file,
            dist.get_rank(),
            attention_modules,
            records,
        )

    gathered_stats: list[Any] = [None for _ in range(dist.get_world_size())]
    local_stats = {
        "rank": dist.get_rank(),
        "attention_io_stage": args.attention_io_stage,
        "attention_record_summary": attention_record_summary,
        "attention_replay_stats": replay_stats,
    }
    dist.all_gather_object(gathered_stats, local_stats)

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
            "attention_io": {
                "stage": args.attention_io_stage,
                "file_name": args.attention_io_file.name,
                "modules": attention_modules,
                "per_rank": gathered_stats,
            },
            "summary": _summarize_log_probs(log_probs),
            "log_probs": log_probs,
        }
        args.parity_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.parity_output)
        printable = {k: v for k, v in payload.items() if k != "log_probs"}
        print(json.dumps(printable, indent=2, sort_keys=True))
        print(f"wrote {args.parity_output}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

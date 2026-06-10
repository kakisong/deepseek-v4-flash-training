#!/usr/bin/env python3
"""DeepSeek-V4 mini checkpoint 的逐层漂移探针。

先用 torchrun 运行以保存某一个 attention 后端的 trace tensor，然后对两个
trace 文件运行 ``--compare``。这是一个针对全模型漂移的诊断探针；
它并不是外部参考一致性测试。
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
TRACE_PATTERNS = (
    re.compile(r".*embedding\.word_embeddings$"),
    re.compile(r".*decoder\.layers\.\d+$"),
    re.compile(r".*decoder\.layers\.\d+\.input_layernorm$"),
    re.compile(r".*decoder\.layers\.\d+\.pre_mlp_layernorm$"),
    re.compile(r".*decoder\.layers\.\d+\.self_attention$"),
    re.compile(r".*decoder\.layers\.\d+\.self_attention\.wq_a$"),
    re.compile(r".*decoder\.layers\.\d+\.self_attention\.q_norm$"),
    re.compile(r".*decoder\.layers\.\d+\.self_attention\.wq_b$"),
    re.compile(r".*decoder\.layers\.\d+\.self_attention\.wkv$"),
    re.compile(r".*decoder\.layers\.\d+\.self_attention\.kv_norm$"),
    re.compile(r".*decoder\.layers\.\d+\.self_attention\.wo_b$"),
    re.compile(r".*decoder\.layers\.\d+\.mlp$"),
    re.compile(r".*decoder\.layers\.\d+\.mlp\.router$"),
    re.compile(r".*decoder\.layers\.\d+\.mlp\.experts$"),
    re.compile(r".*decoder\.layers\.\d+\.mlp\.shared_experts$"),
    re.compile(r".*decoder\.final_layernorm$"),
)


def add_trace_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--rollout-data", type=Path)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-a", type=Path)
    parser.add_argument("--compare-b", type=Path)
    parser.add_argument("--compare-output", type=Path)
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
    return parser


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
    args = parse_args(extra_args_provider=add_trace_args)
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
            "tokens": [torch.as_tensor(sample["tokens"], dtype=torch.long) for sample in samples],
            "response_lengths": [int(sample["response_length"]) for sample in samples],
            "loss_masks": [torch.as_tensor(sample["loss_mask"], dtype=torch.int32) for sample in samples],
            "rewards": [float(sample.get("reward", 0.0)) for sample in samples],
            "truncated": [int(sample.get("truncated", 0)) for sample in samples],
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
        out[key] = value[:n] if isinstance(value, list) else value
    return out


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


def _should_trace(name: str) -> bool:
    return any(pattern.fullmatch(name) for pattern in TRACE_PATTERNS)


def _ranked_replay_path(path: Path, rank: int) -> Path:
    return path.with_name(f"{path.stem}.rank{rank}{path.suffix}")


def _enable_routing_replay(stage: str):
    from miles.utils.replay_base import routing_replay_manager

    routing_replay_manager.enabled = stage != "off"
    routing_replay_manager.stage = "fallthrough" if stage == "off" else stage
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    return routing_replay_manager


def _load_routing_replay(manager, replay_file: Path, rank: int) -> None:
    payload = torch.load(_ranked_replay_path(replay_file, rank), map_location="cpu", weights_only=False)
    saved = payload["top_indices_by_replay"]
    if len(saved) != len(manager.replays):
        raise ValueError(f"rank {rank}: replay count mismatch, saved={len(saved)}, current={len(manager.replays)}")
    for replay, tensors in zip(manager.replays, saved, strict=True):
        replay.top_indices_list = [tensor.cpu() for tensor in tensors]
        replay.forward_index = 0
        replay.backward_index = 0


def _save_routing_replay(manager, replay_file: Path, rank: int) -> None:
    payload = {
        "rank": rank,
        "top_indices_by_replay": [
            [tensor.detach().cpu() for tensor in replay.top_indices_list] for replay in manager.replays
        ],
    }
    out = _ranked_replay_path(replay_file, rank)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)


def _install_hooks(module: torch.nn.Module, traces: dict[str, torch.Tensor]) -> list[Any]:
    handles = []
    for name, submodule in module.named_modules():
        if not _should_trace(name):
            continue

        def hook(_module, _inputs, output, *, trace_name=name):
            tensors = _named_tensors(output)
            if not tensors:
                return
            traces[trace_name] = tensors[0][1].detach().float().cpu()
            if len(tensors) > 1:
                for suffix, tensor in tensors:
                    traces[f"{trace_name}.{suffix}"] = tensor.detach().float().cpu()
            debug_tensors = getattr(_module, "_dsv4_debug_tensors", None)
            if debug_tensors:
                for suffix, tensor in debug_tensors.items():
                    traces[f"{trace_name}.{suffix}"] = tensor.detach().float().cpu()

        handles.append(submodule.register_forward_hook(hook))
    return handles


def _tensor_sha256(tensors: list[torch.Tensor]) -> str:
    import hashlib

    h = hashlib.sha256()
    for tensor in tensors:
        data = tensor.detach().cpu().contiguous().float()
        h.update(str(tuple(data.shape)).encode("ascii"))
        h.update(data.numpy().tobytes())
    return h.hexdigest()


def run_trace() -> int:
    configure_logger()
    _init_distributed()
    args = _parse_megatron_args()
    assert args.rollout_data is not None, "--rollout-data is required"
    assert args.trace_output is not None, "--trace-output is required"
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

    local_traces: dict[str, torch.Tensor] = {}
    handles = _install_hooks(model[0], local_traces)
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
    if args.routing_replay_stage == "record":
        _save_routing_replay(routing_replay_manager, args.routing_replay_file, dist.get_rank())

    if mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0 and parallel_state.dp_cp_rank == 0:
        log_probs = [tensor.detach().cpu().float() for tensor in result["log_probs"]]
        payload = {
            "seed": SEED,
            "attention_impl": os.getenv("MEGATRON_SPARSE_ATTN_IMPL", "tilelang"),
            "runtime": {
                "deterministic_mode": bool(getattr(args, "deterministic_mode", False)),
                "NCCL_ALGO": os.getenv("NCCL_ALGO"),
                "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
            },
            "rollout_data_name": args.rollout_data.name,
            "load_name": Path(args.load).name,
            "logprob_sha256": _tensor_sha256(log_probs),
            "routing_replay": {
                "stage": args.routing_replay_stage,
                "file_name": args.routing_replay_file.name if args.routing_replay_file is not None else None,
            },
            "trace_summaries": {name: _tensor_summary(tensor) for name, tensor in sorted(local_traces.items())},
            "trace_tensors": local_traces,
        }
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.trace_output)
        printable = {k: v for k, v in payload.items() if k != "trace_tensors"}
        print(json.dumps(printable, indent=2, sort_keys=True))
        print(f"wrote {args.trace_output}")

    dist.barrier()
    dist.destroy_process_group()
    return 0


def _rel_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = (xf.square().sum() + yf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (xf * yf).sum().item() / denom)


def run_compare() -> int:
    parser = argparse.ArgumentParser()
    add_trace_args(parser)
    args, _ = parser.parse_known_args()
    assert args.compare_a is not None
    assert args.compare_b is not None
    a = torch.load(args.compare_a, map_location="cpu", weights_only=False)
    b = torch.load(args.compare_b, map_location="cpu", weights_only=False)
    a_tensors = a["trace_tensors"]
    b_tensors = b["trace_tensors"]
    rows = []
    for name in sorted(set(a_tensors) & set(b_tensors)):
        x = a_tensors[name].float()
        y = b_tensors[name].float()
        if x.shape != y.shape:
            rows.append(
                {
                    "name": name,
                    "shape": None,
                    "shape_a": list(x.shape),
                    "shape_b": list(y.shape),
                    "shape_mismatch": True,
                    "numel_a": int(x.numel()),
                    "numel_b": int(y.numel()),
                    "numel": None,
                    "nonzero_abs_count": None,
                    "exact_equal": False,
                    "max_abs": None,
                    "mean_abs": None,
                    "relative_gap": None,
                }
            )
            continue
        diff = (x - y).abs()
        nonzero_count = int((diff != 0).sum().item())
        rows.append(
            {
                "name": name,
                "shape": list(x.shape),
                "shape_mismatch": False,
                "numel": int(diff.numel()),
                "nonzero_abs_count": nonzero_count,
                "exact_equal": nonzero_count == 0,
                "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
                "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
                "relative_gap": _rel_gap(x, y),
            }
        )
    comparable_rows = [row for row in rows if row["max_abs"] is not None]
    payload = {
        "a_impl": a.get("attention_impl"),
        "b_impl": b.get("attention_impl"),
        "a_routing_replay": a.get("routing_replay"),
        "b_routing_replay": b.get("routing_replay"),
        "num_common_traces": len(rows),
        "num_shape_mismatches": sum(1 for row in rows if row.get("shape_mismatch")),
        "max_abs_trace": max(comparable_rows, key=lambda item: item["max_abs"]) if comparable_rows else None,
        "max_relative_trace": max(comparable_rows, key=lambda item: item["relative_gap"]) if comparable_rows else None,
        "traces": rows,
    }
    if args.compare_output is not None:
        args.compare_output.parent.mkdir(parents=True, exist_ok=True)
        args.compare_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    if "--compare" in os.sys.argv:
        return run_compare()
    return run_trace()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

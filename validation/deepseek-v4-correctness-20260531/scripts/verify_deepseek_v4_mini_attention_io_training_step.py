#!/usr/bin/env python3
"""DeepSeek-V4 mini-checkpoint attention I/O training-step replay.

This verifier uses dense attention inputs recorded from the mini checkpoint,
then runs one local forward/backward/update step for each attention backend on
the same checkpoint attention weights and the same synthetic upstream gradient.
It proves the attention backend training surface on real checkpoint inputs,
separately from upstream state drift and downstream logprob amplification.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_model

import miles_plugins.mbridge  # noqa: F401
from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.logging_utils import configure_logger


SEED = 20260531


@dataclass
class LayerResult:
    loss: float
    output: torch.Tensor
    input_grad: torch.Tensor
    state_after_step: dict[str, torch.Tensor]
    num_params_with_grad: int
    params_without_grad: list[str]
    max_grad_abs: float


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--attention-io-file", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--attention-modules", nargs="*", default=None)
    parser.add_argument("--impls", nargs="+", default=["dense", "sparse", "tilelang"])
    parser.add_argument("--manual-sgd-lr", type=float, default=1e-7)
    parser.add_argument("--qkv-format", choices=["thd", "bshd"], default="thd")
    parser.add_argument("--data-pad-size-multiplier", type=int, default=128)
    parser.add_argument("--log-probs-chunk-size", type=int, default=-1)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--allgather-cp", action="store_true")
    parser.add_argument("--true-on-policy-mode", action="store_true")
    parser.add_argument("--use-rollout-entropy", action="store_true")
    parser.add_argument("--use-dynamic-batch-size", action="store_true")
    parser.add_argument("--max-output-abs", type=float, default=0.0625)
    parser.add_argument("--max-input-grad-abs", type=float, default=0.01)
    parser.add_argument("--max-state-abs", type=float, default=2e-5)
    parser.add_argument("--max-rel-gap", type=float, default=5e-4)
    return parser


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


def _parse_args():
    args = parse_args(extra_args_provider=add_args)
    args = set_default_megatron_args(args)
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = True
    args.__dict__.setdefault("custom_megatron_before_log_prob_hook_path", None)
    args.save_interval = getattr(args, "save_interval", None) or 1
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    validate_args(args)
    args.variable_seq_lengths = True
    return args


def _default_attention_modules(args: Any) -> list[str]:
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    return [f"module.decoder.layers.{idx}.self_attention" for idx in range(num_layers)]


def _load_attention_records(path: Path, rank: int) -> dict[str, dict[str, torch.Tensor]]:
    payload = torch.load(_ranked_path(path, rank), map_location="cpu", weights_only=False)
    return payload["records"]


def _make_upstream_grad(shape: torch.Size, *, layer_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(int(torch.tensor(shape).prod().item()), device=device, dtype=torch.float32).reshape(shape)
    values = torch.sin(values * 0.00017 + float(layer_idx + 1))
    return values.to(dtype=dtype)


def _tensor_rel_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = float((xf.square().sum() + yf.square().sum()).item())
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * float((xf * yf).sum().item()) / denom)


def _all_reduce_float(value: float, op: dist.ReduceOp) -> float:
    tensor = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _all_reduce_int(value: int, op: dist.ReduceOp) -> int:
    tensor = torch.tensor([value], device="cuda", dtype=torch.long)
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


def _zero_grad(module: torch.nn.Module) -> None:
    for param in module.parameters():
        param.grad = None
        main_grad = getattr(param, "main_grad", None)
        if main_grad is not None:
            main_grad.zero_()


def _prepare_manual_grad_buffers(model: list[torch.nn.Module], args: Any) -> None:
    main_grad_dtype = torch.float32 if getattr(args, "accumulate_allreduce_grads_in_fp32", False) else None
    for module in model:
        module.zero_grad(set_to_none=True)
        for param in module.parameters():
            if not param.requires_grad:
                continue
            dtype = main_grad_dtype or param.dtype
            param.main_grad = torch.zeros_like(param, dtype=dtype, memory_format=torch.preserve_format)


def _grad_buffer(param: torch.nn.Parameter) -> torch.Tensor | None:
    main_grad = getattr(param, "main_grad", None)
    if main_grad is not None:
        return main_grad
    return param.grad


def _run_impl(args: Any, impl: str, records: dict[str, dict[str, torch.Tensor]], module_names: list[str]) -> dict[str, LayerResult]:
    os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = impl
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    _prepare_manual_grad_buffers(model, args)
    for module in model:
        module.train()
    named_modules = dict(model[0].named_modules())

    results: dict[str, LayerResult] = {}
    for layer_idx, name in enumerate(module_names):
        if name not in named_modules:
            raise KeyError(f"module not found: {name}")
        if name not in records or "input" not in records[name]:
            raise KeyError(f"attention IO record missing input for {name}")
        module = named_modules[name]
        _zero_grad(module)
        x = records[name]["input"].to(device="cuda", dtype=torch.bfloat16).detach().clone().requires_grad_(True)
        output = module(x)
        if not torch.isfinite(output.detach().float()).all():
            raise RuntimeError(f"{impl}.{name}.output contains non-finite values")
        upstream = _make_upstream_grad(output.shape, layer_idx=layer_idx, device=output.device, dtype=output.dtype)
        loss = (output.float() * upstream.float()).mean()
        loss.backward()
        if x.grad is None:
            raise RuntimeError(f"{impl}.{name}.input_grad missing")
        if not torch.isfinite(x.grad.detach().float()).all():
            raise RuntimeError(f"{impl}.{name}.input_grad contains non-finite values")

        max_grad_abs = 0.0
        num_params_with_grad = 0
        params_without_grad: list[str] = []
        for param_name, param in module.named_parameters():
            grad = _grad_buffer(param)
            if grad is None:
                params_without_grad.append(param_name)
                continue
            grad = grad.detach()
            if not torch.isfinite(grad.float()).all():
                raise RuntimeError(f"{impl}.{name}.{param_name}.grad contains non-finite values")
            max_grad_abs = max(max_grad_abs, float(grad.float().abs().max().item()) if grad.numel() else 0.0)
            num_params_with_grad += 1
            param.data.add_(grad.to(dtype=param.dtype), alpha=-args.manual_sgd_lr)

        results[name] = LayerResult(
            loss=float(loss.detach().item()),
            output=output.detach().cpu().float(),
            input_grad=x.grad.detach().cpu().float(),
            state_after_step={
                param_name: tensor.detach().cpu().float()
                for param_name, tensor in module.state_dict().items()
                if torch.is_tensor(tensor) and tensor.numel() > 0
            },
            num_params_with_grad=num_params_with_grad,
            params_without_grad=sorted(params_without_grad),
            max_grad_abs=max_grad_abs,
        )

        del x, output, loss, upstream
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return results


def _compare_layer(left: LayerResult, right: LayerResult) -> dict[str, Any]:
    state_max_abs = 0.0
    state_max_rel = 0.0
    state_max_name = ""
    for name, left_tensor in left.state_after_step.items():
        right_tensor = right.state_after_step[name]
        max_abs = float((left_tensor - right_tensor).abs().max().item())
        rel = _tensor_rel_gap(left_tensor, right_tensor)
        if max_abs > state_max_abs:
            state_max_abs = max_abs
            state_max_name = name
        state_max_rel = max(state_max_rel, rel)
    return {
        "loss_abs": abs(left.loss - right.loss),
        "output_max_abs": float((left.output - right.output).abs().max().item()),
        "output_rel_gap": _tensor_rel_gap(left.output, right.output),
        "input_grad_max_abs": float((left.input_grad - right.input_grad).abs().max().item()),
        "input_grad_rel_gap": _tensor_rel_gap(left.input_grad, right.input_grad),
        "state_after_step_max_abs": state_max_abs,
        "state_after_step_max_abs_name": state_max_name,
        "state_after_step_max_rel_gap": state_max_rel,
    }


def _globalize_comparison(local: dict[str, Any]) -> dict[str, Any]:
    return {
        "loss_abs_global_max": _all_reduce_float(local["loss_abs"], dist.ReduceOp.MAX),
        "output_max_abs_global_max": _all_reduce_float(local["output_max_abs"], dist.ReduceOp.MAX),
        "output_rel_gap_global_max": _all_reduce_float(local["output_rel_gap"], dist.ReduceOp.MAX),
        "input_grad_max_abs_global_max": _all_reduce_float(local["input_grad_max_abs"], dist.ReduceOp.MAX),
        "input_grad_rel_gap_global_max": _all_reduce_float(local["input_grad_rel_gap"], dist.ReduceOp.MAX),
        "state_after_step_max_abs_global_max": _all_reduce_float(local["state_after_step_max_abs"], dist.ReduceOp.MAX),
        "state_after_step_max_rel_gap_global_max": _all_reduce_float(local["state_after_step_max_rel_gap"], dist.ReduceOp.MAX),
        "state_after_step_max_abs_name_rank_local": local["state_after_step_max_abs_name"],
    }


def main() -> int:
    configure_logger()
    _init_distributed()
    args = _parse_args()
    try:
        init(args)
        module_names = args.attention_modules or _default_attention_modules(args)
        records = _load_attention_records(args.attention_io_file, dist.get_rank())
        impl_results = {
            impl: _run_impl(args, impl, records, module_names)
            for impl in args.impls
        }

        local_layers: dict[str, Any] = {}
        failures: list[str] = []
        for name in module_names:
            impl_stats = {}
            for impl, result_by_layer in impl_results.items():
                layer_result = result_by_layer[name]
                impl_stats[impl] = {
                    "loss": layer_result.loss,
                    "num_params_with_grad": layer_result.num_params_with_grad,
                    "params_without_grad": layer_result.params_without_grad,
                    "max_grad_abs": layer_result.max_grad_abs,
                }
                if layer_result.num_params_with_grad <= 0:
                    failures.append(f"{name}.{impl}.no_parameter_gradients")
            comparisons = []
            for left, right in [("dense", "sparse"), ("dense", "tilelang"), ("sparse", "tilelang")]:
                if left in impl_results and right in impl_results:
                    local = _compare_layer(impl_results[left][name], impl_results[right][name])
                    globalized = _globalize_comparison(local)
                    globalized["label"] = f"{left}_vs_{right}"
                    comparisons.append(globalized)
                    checks = {
                        "output_max_abs": globalized["output_max_abs_global_max"] <= args.max_output_abs,
                        "output_rel_gap": globalized["output_rel_gap_global_max"] <= args.max_rel_gap,
                        "input_grad_max_abs": globalized["input_grad_max_abs_global_max"] <= args.max_input_grad_abs,
                        "input_grad_rel_gap": globalized["input_grad_rel_gap_global_max"] <= args.max_rel_gap,
                        "state_after_step_max_abs": globalized["state_after_step_max_abs_global_max"] <= args.max_state_abs,
                        "state_after_step_max_rel_gap": globalized["state_after_step_max_rel_gap_global_max"] <= args.max_rel_gap,
                    }
                    for check_name, passed in checks.items():
                        if not passed:
                            failures.append(f"{name}.{left}_vs_{right}.{check_name}")

            local_layers[name] = {
                "impls": impl_stats,
                "comparisons": comparisons,
            }

        gathered_layers: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            gathered_layers,
            {
                "rank": dist.get_rank(),
                "layers": local_layers,
                "failures": failures,
            },
        )
        all_failures = []
        for rank_payload in gathered_layers:
            for failure in rank_payload["failures"]:
                all_failures.append(f"rank{rank_payload['rank']}.{failure}")
        unique_failures = sorted(set(all_failures))

        payload = {
            "date": "2026-05-31",
            "seed": SEED,
            "scope": "4-layer DeepSeek-V4 mini checkpoint attention I/O local training-step replay",
            "status": "PASS" if not unique_failures else "FAIL",
            "checkpoint": Path(args.load).name if args.load else None,
            "attention_io_name": args.attention_io_file.name,
            "manual_sgd_lr": args.manual_sgd_lr,
            "thresholds": {
                "max_output_abs": args.max_output_abs,
                "max_input_grad_abs": args.max_input_grad_abs,
                "max_state_abs": args.max_state_abs,
                "max_rel_gap": args.max_rel_gap,
            },
            "impls": args.impls,
            "modules": module_names,
            "world_size": dist.get_world_size(),
            "per_rank": gathered_layers,
            "failures": unique_failures,
        }
        if dist.get_rank() == 0:
            args.train_output.parent.mkdir(parents=True, exist_ok=True)
            args.train_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2, sort_keys=True))
            print(f"wrote {args.train_output}")
        dist.barrier()
        return 0 if not unique_failures else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

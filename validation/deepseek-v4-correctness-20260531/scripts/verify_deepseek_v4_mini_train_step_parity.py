#!/usr/bin/env python3
"""DeepSeek-V4 迷你 checkpoint 的单步训练一致性校验。

本校验器加载同一份 Miles/Megatron 迷你 checkpoint 与同一份导出的
训练批次，然后对每个 attention 后端执行一次确定性的 SFT 步骤：

* ``MEGATRON_SPARSE_ATTN_IMPL=dense``
* ``MEGATRON_SPARSE_ATTN_IMPL=sparse``
* ``MEGATRON_SPARSE_ATTN_IMPL=tilelang``

对每个后端，它在已加载的模型上依次执行前向、SFT loss、反向、
梯度有限性检查，以及一次手动 SGD 更新。随后对比标量 loss、
选定的梯度以及选定的更新后参数 tensor。本检查是一个
端到端的 Miles/Megatron 后端一致性探针；它仍然不是与外部
HF/Transformers 参考实现的一致性测试。
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
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state, get_packed_seq_params
from miles.backends.training_utils.data import DataIterator, get_batch
from miles.backends.training_utils.loss import loss_function
from miles.utils.logging_utils import configure_logger


SEED = 20260531


SELECTED_PARAM_MARKERS = (
    "final_layernorm.weight",
    "self_attention.attn_sink",
    "self_attention.wq_a.weight",
    "self_attention.wkv.weight",
    "self_attention.compressor.wkv.weight",
    "self_attention.compressor.wgate.weight",
    "self_attention.kv_norm.weight",
    "self_attention.q_norm.weight",
    "hc_attn_scale",
    "hc_ffn_scale",
    "hc_head_params.hc_head_scale",
)


@dataclass
class ImplResult:
    impl: str
    loss: float
    log: dict[str, float]
    local_stats: dict[str, Any]
    global_stats: dict[str, Any]
    selected_grads: dict[str, torch.Tensor]
    selected_states: dict[str, torch.Tensor]
    routing_replay: dict[str, Any]
    routing_replay_snapshot: list[list[torch.Tensor]] | None
    attention_output_replay: dict[str, Any]
    attention_output_snapshot: dict[str, torch.Tensor] | None


def add_train_parity_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--train-parity-output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--impls", nargs="+", default=["dense", "sparse", "tilelang"])
    parser.add_argument("--manual-update-rule", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--manual-sgd-lr", type=float, default=1e-7)
    parser.add_argument("--manual-adamw-lr", type=float, default=None)
    parser.add_argument("--manual-adamw-beta1", type=float, default=0.9)
    parser.add_argument("--manual-adamw-beta2", type=float, default=0.98)
    parser.add_argument("--manual-adamw-eps", type=float, default=1e-8)
    parser.add_argument("--manual-adamw-weight-decay", type=float, default=0.1)
    parser.add_argument("--manual-update-selected-only", action="store_true")
    parser.add_argument("--max-selected-numel", type=int, default=5_000_000)
    parser.add_argument("--max-loss-abs", type=float, default=2e-2)
    parser.add_argument("--max-selected-grad-rel-gap", type=float, default=2e-3)
    parser.add_argument("--max-selected-state-abs", type=float, default=2e-5)
    parser.add_argument("--routing-replay-mode", choices=["off", "record_replay"], default="off")
    parser.add_argument("--routing-replay-reference-impl", type=str, default="dense")
    parser.add_argument("--attention-output-replay-mode", choices=["off", "record_replay"], default="off")
    parser.add_argument("--attention-output-replay-reference-impl", type=str, default="dense")
    parser.add_argument("--qkv-format", choices=["thd", "bshd"], default="thd")
    parser.add_argument("--data-pad-size-multiplier", type=int, default=128)
    parser.add_argument("--log-probs-chunk-size", type=int, default=-1)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--allgather-cp", action="store_true")
    parser.add_argument("--true-on-policy-mode", action="store_true")
    parser.add_argument("--use-rollout-entropy", action="store_true")
    parser.add_argument("--use-dynamic-batch-size", action="store_true")
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
    args = parse_args(extra_args_provider=add_train_parity_args)
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
        if isinstance(value, list):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _move_rollout_to_device(rollout_data: dict[str, list[Any]], device: torch.device) -> dict[str, list[Any]]:
    out = copy.copy(rollout_data)
    for key in ("tokens", "loss_masks", "log_probs", "ref_log_probs", "advantages", "returns", "rollout_log_probs"):
        if key in out and isinstance(out[key], list):
            out[key] = [
                value.to(device=device) if torch.is_tensor(value) else value
                for value in out[key]
            ]
    return out


def _tensor_rel_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = (xf.square().sum() + yf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (xf * yf).sum().item() / denom)


def _all_reduce_float(value: float, op: dist.ReduceOp) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _all_reduce_int(value: int, op: dist.ReduceOp) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device="cuda")
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


def _is_selected_param(name: str, param: torch.nn.Parameter, max_numel: int) -> bool:
    return param.numel() <= max_numel and any(marker in name for marker in SELECTED_PARAM_MARKERS)


def _summarize_log(loss_log: dict[str, list[str] | torch.Tensor]) -> dict[str, float]:
    keys = loss_log.get("keys", [])
    values = loss_log.get("values")
    if not isinstance(keys, list) or not torch.is_tensor(values):
        return {}
    vals = values.detach().float().cpu().tolist()
    # values[0] 是样本/token 计数；其余的值与 keys 一一对应。
    result = {"count": float(vals[0])} if vals else {}
    for key, value in zip(keys, vals[1:], strict=False):
        result[str(key)] = float(value)
    return result


def _grad_buffer(param: torch.nn.Parameter) -> torch.Tensor | None:
    main_grad = getattr(param, "main_grad", None)
    if main_grad is not None:
        return main_grad
    return param.grad


def _prepare_manual_grad_buffers(model: list[torch.nn.Module], args) -> None:
    main_grad_dtype = torch.float32 if args.accumulate_allreduce_grads_in_fp32 else None
    for module in model:
        module.zero_grad(set_to_none=True)
        for param in module.parameters():
            if not param.requires_grad:
                continue
            dtype = main_grad_dtype or param.dtype
            param.main_grad = torch.zeros_like(param, dtype=dtype, memory_format=torch.preserve_format)


def _zero_manual_grad_buffers(model: list[torch.nn.Module]) -> None:
    for module in model:
        module.zero_grad(set_to_none=True)
        for param in module.parameters():
            main_grad = getattr(param, "main_grad", None)
            if main_grad is not None:
                main_grad.zero_()


def _enable_routing_replay(stage: str):
    from miles.utils.replay_base import routing_replay_manager

    routing_replay_manager.enabled = stage != "off"
    routing_replay_manager.stage = "fallthrough" if stage == "off" else stage
    routing_replay_manager.replays = []
    routing_replay_manager.current = None
    return routing_replay_manager


def _snapshot_routing_replay(manager) -> list[list[torch.Tensor]]:
    return [[tensor.detach().cpu().clone() for tensor in replay.top_indices_list] for replay in manager.replays]


def _restore_routing_replay(manager, snapshot: list[list[torch.Tensor]]) -> None:
    if len(snapshot) != len(manager.replays):
        raise ValueError(f"routing replay count mismatch: saved={len(snapshot)}, current={len(manager.replays)}")
    for replay, tensors in zip(manager.replays, snapshot, strict=True):
        replay.top_indices_list = [tensor.detach().cpu().clone() for tensor in tensors]
        replay.forward_index = 0
        replay.backward_index = 0


def _routing_replay_summary(manager, stage: str, source_impl: str | None) -> dict[str, Any]:
    return {
        "stage": stage,
        "source_impl": source_impl,
        "num_replays": len(manager.replays),
        "num_recorded_tensors": sum(len(replay.top_indices_list) for replay in manager.replays),
    }


def _manual_update_param(param: torch.nn.Parameter, grad_buffer: torch.Tensor, args) -> None:
    if args.manual_update_rule == "sgd":
        param.add_(grad_buffer.to(param.dtype), alpha=-args.manual_sgd_lr)
        return

    if args.manual_update_rule != "adamw":
        raise ValueError(f"unsupported manual update rule: {args.manual_update_rule}")

    lr = args.manual_adamw_lr if args.manual_adamw_lr is not None else args.manual_sgd_lr
    grad = grad_buffer.detach().float()
    param_fp32 = param.detach().float()
    # 从零矩状态开始的第一步 AdamW。偏置校正会把 m_hat/v_hat 化简为
    # grad 和 grad**2，因此本单步一致性检查无需持久化的
    # 优化器状态。
    update = grad / (grad.abs() + args.manual_adamw_eps)
    updated = param_fp32.mul(1.0 - lr * args.manual_adamw_weight_decay).add(update, alpha=-lr)
    param.copy_(updated.to(dtype=param.dtype))


def _attention_module_names(model: list[torch.nn.Module]) -> list[str]:
    return sorted(
        name
        for name, _ in model[0].named_modules()
        if name.startswith("module.decoder.layers.") and name.endswith(".self_attention")
    )


def _install_attention_output_replay_hooks(
    model: list[torch.nn.Module],
    *,
    stage: str,
    snapshot: dict[str, torch.Tensor] | None,
) -> tuple[list[Any], dict[str, torch.Tensor], list[str]]:
    records: dict[str, torch.Tensor] = {}
    handles: list[Any] = []
    module_names = _attention_module_names(model)
    if stage == "off":
        return handles, records, module_names
    if stage == "replay_forward" and snapshot is None:
        raise ValueError("attention output replay snapshot is required for replay_forward")

    named_modules = dict(model[0].named_modules())
    for name in module_names:
        module = named_modules[name]

        def hook(_module, _inputs, output, *, module_name=name):
            if not torch.is_tensor(output):
                raise TypeError(f"{module_name} output is not a tensor: {type(output)!r}")
            if stage == "record":
                records[module_name] = output.detach().cpu().clone()
                return output

            assert snapshot is not None
            if module_name not in snapshot:
                raise KeyError(f"attention output replay missing {module_name}")
            reference = snapshot[module_name].to(device=output.device, dtype=output.dtype)
            if reference.shape != output.shape:
                raise ValueError(
                    f"attention output replay shape mismatch for {module_name}: "
                    f"reference={tuple(reference.shape)} actual={tuple(output.shape)}"
                )
            return reference + (output - output.detach())

        handles.append(module.register_forward_hook(hook))

    return handles, records, module_names


def _attention_output_replay_summary(
    *,
    stage: str,
    source_impl: str | None,
    module_names: list[str],
    records: dict[str, torch.Tensor],
    snapshot: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "source_impl": source_impl,
        "num_modules": len(module_names),
        "num_recorded_tensors": len(records) if stage == "record" else len(snapshot or {}),
        "straight_through": stage == "replay_forward",
    }


def _run_impl(
    impl: str,
    args,
    base_rollout_data: dict[str, list[Any]],
    *,
    routing_replay_stage: str = "off",
    routing_replay_snapshot: list[list[torch.Tensor]] | None = None,
    routing_replay_source_impl: str | None = None,
    attention_output_replay_stage: str = "off",
    attention_output_snapshot: dict[str, torch.Tensor] | None = None,
    attention_output_replay_source_impl: str | None = None,
) -> ImplResult:
    os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = impl
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    routing_replay_manager = _enable_routing_replay(routing_replay_stage)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    if routing_replay_stage == "replay_forward":
        assert routing_replay_snapshot is not None, "routing replay snapshot is required for replay_forward"
        _restore_routing_replay(routing_replay_manager, routing_replay_snapshot)
    for module in model:
        module.train()
    _prepare_manual_grad_buffers(model, args)
    attention_handles, attention_records, attention_module_names = _install_attention_output_replay_hooks(
        model,
        stage=attention_output_replay_stage,
        snapshot=attention_output_snapshot,
    )

    parallel_state = create_megatron_parallel_state(model)
    device = torch.device("cuda", torch.cuda.current_device())
    rollout_data = _move_rollout_to_device(base_rollout_data, device)
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

    forward_kwargs = {
        "input_ids": batch["tokens"],
        "position_ids": None,
        "attention_mask": None,
        "labels": None,
        "packed_seq_params": get_packed_seq_params(batch, args),
        "loss_mask": batch["full_loss_masks"],
    }
    if batch["multimodal_train_inputs"] is not None:
        forward_kwargs.update(batch["multimodal_train_inputs"])

    try:
        output_tensor = model[0](**forward_kwargs)
    finally:
        for handle in attention_handles:
            handle.remove()
    assert torch.isfinite(output_tensor).all(), f"{impl}: non-finite model output"
    loss, _, loss_log = loss_function(
        args,
        parallel_state,
        batch,
        num_microbatches=1,
        logits=output_tensor,
        apply_megatron_loss_scaling=False,
    )
    assert torch.isfinite(loss).all(), f"{impl}: non-finite loss"
    loss.backward()

    selected_grads: dict[str, torch.Tensor] = {}
    selected_states: dict[str, torch.Tensor] = {}
    local_num_params = 0
    local_num_params_with_grad = 0
    local_num_selected = 0
    local_nonfinite_grad_tensors = 0
    local_max_grad_abs = 0.0
    local_grad_norm_sq = 0.0
    local_num_grad_elems = 0

    with torch.no_grad():
        for module in model:
            for name, param in module.named_parameters():
                local_num_params += int(param.numel())
                selected = _is_selected_param(name, param, args.max_selected_numel)
                grad_buffer = _grad_buffer(param)
                if grad_buffer is not None:
                    grad = grad_buffer.detach()
                    local_num_params_with_grad += 1
                    if not torch.isfinite(grad).all():
                        local_nonfinite_grad_tensors += 1
                    grad_float = grad.float()
                    local_max_grad_abs = max(local_max_grad_abs, float(grad_float.abs().max().item()))
                    local_grad_norm_sq += float(grad_float.square().sum().item())
                    local_num_grad_elems += int(grad.numel())
                    if selected:
                        selected_grads[name] = grad_float.cpu()
                if grad_buffer is not None and (not args.manual_update_selected_only or selected):
                    _manual_update_param(param, grad_buffer, args)
                if selected:
                    selected_states[name] = param.detach().float().cpu()
                    local_num_selected += 1

    global_stats = {
        "num_parameters": _all_reduce_int(local_num_params, dist.ReduceOp.SUM),
        "num_params_with_grad": _all_reduce_int(local_num_params_with_grad, dist.ReduceOp.SUM),
        "num_grad_elements": _all_reduce_int(local_num_grad_elems, dist.ReduceOp.SUM),
        "num_selected_tensors": _all_reduce_int(local_num_selected, dist.ReduceOp.SUM),
        "nonfinite_grad_tensors": _all_reduce_int(local_nonfinite_grad_tensors, dist.ReduceOp.SUM),
        "max_grad_abs": _all_reduce_float(local_max_grad_abs, dist.ReduceOp.MAX),
        "grad_l2_norm": _all_reduce_float(local_grad_norm_sq, dist.ReduceOp.SUM) ** 0.5,
        "loss_min": _all_reduce_float(float(loss.detach().item()), dist.ReduceOp.MIN),
        "loss_max": _all_reduce_float(float(loss.detach().item()), dist.ReduceOp.MAX),
    }
    routing_snapshot = _snapshot_routing_replay(routing_replay_manager) if routing_replay_stage == "record" else None
    routing_replay = _routing_replay_summary(
        routing_replay_manager,
        routing_replay_stage,
        routing_replay_source_impl,
    )
    attention_output_replay = _attention_output_replay_summary(
        stage=attention_output_replay_stage,
        source_impl=attention_output_replay_source_impl,
        module_names=attention_module_names,
        records=attention_records,
        snapshot=attention_output_snapshot,
    )
    local_stats = {
        "rank": dist.get_rank(),
        "loss": float(loss.detach().item()),
        "num_parameters": local_num_params,
        "num_params_with_grad": local_num_params_with_grad,
        "num_grad_elements": local_num_grad_elems,
        "num_selected_tensors": local_num_selected,
        "nonfinite_grad_tensors": local_nonfinite_grad_tensors,
        "max_grad_abs": local_max_grad_abs,
        "grad_l2_norm": local_grad_norm_sq ** 0.5,
        "selected_tensor_names": sorted(selected_states),
    }

    _zero_manual_grad_buffers(model)
    del model, output_tensor, loss, batch, data_iterator
    torch.cuda.empty_cache()

    return ImplResult(
        impl=impl,
        loss=local_stats["loss"],
        log=_summarize_log(loss_log),
        local_stats=local_stats,
        global_stats=global_stats,
        selected_grads=selected_grads,
        selected_states=selected_states,
        routing_replay=routing_replay,
        routing_replay_snapshot=routing_snapshot,
        attention_output_replay=attention_output_replay,
        attention_output_snapshot=attention_records if attention_output_replay_stage == "record" else None,
    )


def _compare_selected(left: ImplResult, right: ImplResult) -> dict[str, Any]:
    common_grad_names = sorted(set(left.selected_grads) & set(right.selected_grads))
    common_state_names = sorted(set(left.selected_states) & set(right.selected_states))

    local_grad_max_abs = 0.0
    local_grad_max_rel = 0.0
    local_grad_max_name = ""
    for name in common_grad_names:
        x = left.selected_grads[name]
        y = right.selected_grads[name]
        max_abs = float((x - y).abs().max().item())
        rel = _tensor_rel_gap(x, y)
        if max_abs > local_grad_max_abs:
            local_grad_max_abs = max_abs
            local_grad_max_name = name
        local_grad_max_rel = max(local_grad_max_rel, rel)

    local_state_max_abs = 0.0
    local_state_max_rel = 0.0
    local_state_max_name = ""
    for name in common_state_names:
        x = left.selected_states[name]
        y = right.selected_states[name]
        max_abs = float((x - y).abs().max().item())
        rel = _tensor_rel_gap(x, y)
        if max_abs > local_state_max_abs:
            local_state_max_abs = max_abs
            local_state_max_name = name
        local_state_max_rel = max(local_state_max_rel, rel)

    gathered_names: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(
        gathered_names,
        {
            "rank": dist.get_rank(),
            "grad_max_abs_name": local_grad_max_name,
            "state_max_abs_name": local_state_max_name,
            "num_common_grad_tensors": len(common_grad_names),
            "num_common_state_tensors": len(common_state_names),
        },
    )

    return {
        "label": f"{left.impl}_vs_{right.impl}",
        "loss_abs_local": abs(left.loss - right.loss),
        "loss_abs_global_max": _all_reduce_float(abs(left.loss - right.loss), dist.ReduceOp.MAX),
        "selected_grad_max_abs": _all_reduce_float(local_grad_max_abs, dist.ReduceOp.MAX),
        "selected_grad_max_rel_gap": _all_reduce_float(local_grad_max_rel, dist.ReduceOp.MAX),
        "selected_state_max_abs": _all_reduce_float(local_state_max_abs, dist.ReduceOp.MAX),
        "selected_state_max_rel_gap": _all_reduce_float(local_state_max_rel, dist.ReduceOp.MAX),
        "num_common_grad_tensors_global": _all_reduce_int(len(common_grad_names), dist.ReduceOp.SUM),
        "num_common_state_tensors_global": _all_reduce_int(len(common_state_names), dist.ReduceOp.SUM),
        "rank_details": gathered_names if dist.get_rank() == 0 else [],
    }


def main() -> int:
    configure_logger()
    _init_distributed()
    args = _parse_megatron_args()
    try:
        init(args)
        base_rollout_data = _load_rollout_data(args.rollout_data, args.max_samples)
        impls = args.impls
        record_replay_references = []
        if args.routing_replay_mode == "record_replay":
            record_replay_references.append(args.routing_replay_reference_impl)
        if args.attention_output_replay_mode == "record_replay":
            record_replay_references.append(args.attention_output_replay_reference_impl)
        if len(set(record_replay_references)) > 1:
            raise ValueError(
                "routing replay and attention output replay must use the same reference impl "
                f"when both are enabled: {record_replay_references}"
            )
        if record_replay_references:
            reference_impl = record_replay_references[0]
            if reference_impl not in impls:
                raise ValueError(
                    f"record/replay reference impl {reference_impl} is not in --impls"
                )
            run_order = [reference_impl] + [impl for impl in impls if impl != reference_impl]
        else:
            run_order = impls

        results: dict[str, ImplResult] = {}
        routing_snapshot = None
        attention_output_snapshot = None
        for impl in run_order:
            if args.routing_replay_mode == "record_replay":
                if impl == args.routing_replay_reference_impl:
                    replay_stage = "record"
                    replay_source = None
                else:
                    replay_stage = "replay_forward"
                    replay_source = args.routing_replay_reference_impl
            else:
                replay_stage = "off"
                replay_source = None
            if args.attention_output_replay_mode == "record_replay":
                if impl == args.attention_output_replay_reference_impl:
                    attention_output_stage = "record"
                    attention_output_source = None
                else:
                    attention_output_stage = "replay_forward"
                    attention_output_source = args.attention_output_replay_reference_impl
            else:
                attention_output_stage = "off"
                attention_output_source = None
            result = _run_impl(
                impl,
                args,
                base_rollout_data,
                routing_replay_stage=replay_stage,
                routing_replay_snapshot=routing_snapshot,
                routing_replay_source_impl=replay_source,
                attention_output_replay_stage=attention_output_stage,
                attention_output_snapshot=attention_output_snapshot,
                attention_output_replay_source_impl=attention_output_source,
            )
            results[impl] = result
            if replay_stage == "record":
                routing_snapshot = result.routing_replay_snapshot
            if attention_output_stage == "record":
                attention_output_snapshot = result.attention_output_snapshot
        comparisons = []
        for i, left in enumerate(impls):
            for right in impls[i + 1 :]:
                comparisons.append(_compare_selected(results[left], results[right]))

        failures: list[str] = []
        for impl, result in results.items():
            if result.global_stats["nonfinite_grad_tensors"]:
                failures.append(f"{impl}.nonfinite_grad_tensors={result.global_stats['nonfinite_grad_tensors']}")
            if result.global_stats["num_params_with_grad"] <= 0:
                failures.append(f"{impl}.no_parameter_gradients")
        for item in comparisons:
            if item["loss_abs_global_max"] > args.max_loss_abs:
                failures.append(f"{item['label']}.loss_abs_global_max")
            if item["selected_grad_max_rel_gap"] > args.max_selected_grad_rel_gap:
                failures.append(f"{item['label']}.selected_grad_max_rel_gap")
            if item["selected_state_max_abs"] > args.max_selected_state_abs:
                failures.append(f"{item['label']}.selected_state_max_abs")

        gathered_local_stats: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            gathered_local_stats,
            {impl: result.local_stats for impl, result in results.items()},
        )

        payload = {
            "seed": SEED,
            "status": "PASS" if not failures else "FAIL",
            "scope": "4-layer DeepSeek-V4 mini checkpoint SFT one-step backend parity",
            "checkpoint": Path(args.load).name if args.load else None,
            "rollout_data_name": args.rollout_data.name,
            "runtime": {
                "deterministic_mode": bool(getattr(args, "deterministic_mode", False)),
                "NCCL_ALGO": os.getenv("NCCL_ALGO"),
                "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                "CUDA_DEVICE_MAX_CONNECTIONS": os.getenv("CUDA_DEVICE_MAX_CONNECTIONS"),
                "MEGATRON_USE_KV_QAT": os.getenv("MEGATRON_USE_KV_QAT"),
            },
            "routing_replay": {
                "mode": args.routing_replay_mode,
                "reference_impl": args.routing_replay_reference_impl
                if args.routing_replay_mode == "record_replay"
                else None,
            },
            "attention_output_replay": {
                "mode": args.attention_output_replay_mode,
                "reference_impl": args.attention_output_replay_reference_impl
                if args.attention_output_replay_mode == "record_replay"
                else None,
                "straight_through": args.attention_output_replay_mode == "record_replay",
            },
            "world_size": dist.get_world_size(),
            "max_samples": args.max_samples,
            "manual_sgd_lr": args.manual_sgd_lr,
            "manual_update": {
                "rule": args.manual_update_rule,
                "sgd_lr": args.manual_sgd_lr,
                "adamw_lr": args.manual_adamw_lr if args.manual_adamw_lr is not None else args.manual_sgd_lr,
                "adamw_beta1": args.manual_adamw_beta1,
                "adamw_beta2": args.manual_adamw_beta2,
                "adamw_eps": args.manual_adamw_eps,
                "adamw_weight_decay": args.manual_adamw_weight_decay,
                "adamw_step": 1 if args.manual_update_rule == "adamw" else None,
                "adamw_zero_moment_initial_state": args.manual_update_rule == "adamw",
                "selected_only": bool(args.manual_update_selected_only),
            },
            "thresholds": {
                "max_loss_abs": args.max_loss_abs,
                "max_selected_grad_rel_gap": args.max_selected_grad_rel_gap,
                "max_selected_state_abs": args.max_selected_state_abs,
            },
            "impls": {
                impl: {
                    "loss": result.loss,
                    "loss_log": result.log,
                    "global_stats": result.global_stats,
                    "routing_replay": result.routing_replay,
                    "attention_output_replay": result.attention_output_replay,
                }
                for impl, result in results.items()
            },
            "comparisons": comparisons,
            "local_stats_by_rank": gathered_local_stats if dist.get_rank() == 0 else [],
            "failures": failures,
        }
        if dist.get_rank() == 0:
            print(json.dumps(payload, indent=2, sort_keys=True))
            args.train_parity_output.parent.mkdir(parents=True, exist_ok=True)
            args.train_parity_output.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"wrote {args.train_parity_output}")
        dist.barrier()
        return 0 if not failures else 1
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")
    raise SystemExit(main())

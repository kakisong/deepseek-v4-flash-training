#!/usr/bin/env python3
"""DeepSeek-V4 mini checkpoint 的 SFT loss backward/update reference 检查。

本校验器加载 4 层 mini checkpoint 与固定的 rollout batch，运行一次真实的
Miles/Megatron forward，然后在同一份 logits 上计算两种 loss。一条路径使用
Miles 的 loss_function；另一条使用显式的 PyTorch SFT 目标函数：

    sum(-log_softmax(response_logits)[target_token] * loss_mask)

它直接比较标量 loss 与 token 数量，然后对 ``Miles loss - explicit loss``
做一次反向传播。由此得到的选定参数梯度差值和 SGD 状态差值必须为零或
极小。模型 forward 路径有意与 Miles/Megatron 共享；本脚本证明的是已加载
checkpoint 的 SFT 训练目标及其 backward/update 环节，而不是一个完全独立
的整体模型 forward 实现。
"""

from __future__ import annotations

import argparse
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

try:
    from verification.verify_deepseek_v4_mini_train_step_parity import (
        _grad_buffer,
        _is_selected_param,
        _manual_update_param,
        _prepare_manual_grad_buffers,
        _tensor_rel_gap,
        _zero_manual_grad_buffers,
    )
    from verification.verify_deepseek_v4_sft_loss_reference import (
        SEED,
        _all_reduce_float,
        _all_reduce_int,
        _explicit_sft_reference,
        _init_distributed,
        _load_rollout_data,
        _move_rollout_to_device,
        _summarize_log,
    )
except ModuleNotFoundError:
    from verify_deepseek_v4_mini_train_step_parity import (  # type: ignore
        _grad_buffer,
        _is_selected_param,
        _manual_update_param,
        _prepare_manual_grad_buffers,
        _tensor_rel_gap,
        _zero_manual_grad_buffers,
    )
    from verify_deepseek_v4_sft_loss_reference import (  # type: ignore
        SEED,
        _all_reduce_float,
        _all_reduce_int,
        _explicit_sft_reference,
        _init_distributed,
        _load_rollout_data,
        _move_rollout_to_device,
        _summarize_log,
    )


@dataclass
class LossPathResult:
    label: str
    loss: float
    token_count: float
    loss_log: dict[str, float]
    local_stats: dict[str, Any]
    global_stats: dict[str, Any]
    selected_grads: dict[str, torch.Tensor]
    selected_states: dict[str, torch.Tensor]
    per_sample: list[dict[str, Any]]


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--impl", choices=["dense", "sparse", "tilelang"], default="dense")
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
    parser.add_argument("--max-loss-abs", type=float, default=2e-3)
    parser.add_argument("--max-token-count-abs", type=float, default=0.0)
    parser.add_argument("--max-selected-grad-abs", type=float, default=1e-2)
    parser.add_argument("--max-selected-grad-rel-gap", type=float, default=1e-2)
    parser.add_argument("--max-selected-state-abs", type=float, default=2e-5)
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


def _run_loss_path(
    *,
    label: str,
    args: Any,
    base_rollout_data: dict[str, list[Any]],
    use_miles_loss_function: bool,
) -> LossPathResult:
    os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = args.impl
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    for module in model:
        module.train()
    _prepare_manual_grad_buffers(model, args)

    parallel_state = create_megatron_parallel_state(model)
    if parallel_state.tp_size != 1 or parallel_state.cp_size != 1:
        raise ValueError("SFT loss train reference expects TP=1 and CP=1")

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

    output_tensor = model[0](**forward_kwargs)
    if not torch.isfinite(output_tensor).all():
        raise RuntimeError(f"{label}: model output contains non-finite values")

    explicit = _explicit_sft_reference(logits=output_tensor, batch=batch, args=args)
    if use_miles_loss_function:
        loss, normalizer, loss_log = loss_function(
            args,
            parallel_state,
            batch,
            num_microbatches=1,
            logits=output_tensor,
            apply_megatron_loss_scaling=False,
        )
        log_summary = _summarize_log(loss_log)
        per_sample = []
    else:
        loss = explicit["loss_sum"]
        normalizer = explicit["token_count"]
        log_summary = {
            "count": float(normalizer.detach().item()),
            "loss": float(loss.detach().item()),
        }
        per_sample = explicit["per_sample"]

    if not torch.isfinite(loss).all():
        raise RuntimeError(f"{label}: loss contains non-finite values")
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
        "token_count_min": _all_reduce_float(float(normalizer.detach().item()), dist.ReduceOp.MIN),
        "token_count_max": _all_reduce_float(float(normalizer.detach().item()), dist.ReduceOp.MAX),
    }
    local_stats = {
        "rank": dist.get_rank(),
        "loss": float(loss.detach().item()),
        "token_count": float(normalizer.detach().item()),
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

    return LossPathResult(
        label=label,
        loss=local_stats["loss"],
        token_count=local_stats["token_count"],
        loss_log=log_summary,
        local_stats=local_stats,
        global_stats=global_stats,
        selected_grads=selected_grads,
        selected_states=selected_states,
        per_sample=per_sample,
    )


def _compare_paths(left: LossPathResult, right: LossPathResult) -> dict[str, Any]:
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
        "label": f"{left.label}_vs_{right.label}",
        "loss_abs_local": abs(left.loss - right.loss),
        "loss_abs_global_max": _all_reduce_float(abs(left.loss - right.loss), dist.ReduceOp.MAX),
        "token_count_abs_local": abs(left.token_count - right.token_count),
        "token_count_abs_global_max": _all_reduce_float(abs(left.token_count - right.token_count), dist.ReduceOp.MAX),
        "selected_grad_max_abs": _all_reduce_float(local_grad_max_abs, dist.ReduceOp.MAX),
        "selected_grad_max_rel_gap": _all_reduce_float(local_grad_max_rel, dist.ReduceOp.MAX),
        "selected_state_max_abs": _all_reduce_float(local_state_max_abs, dist.ReduceOp.MAX),
        "selected_state_max_rel_gap": _all_reduce_float(local_state_max_rel, dist.ReduceOp.MAX),
        "num_common_grad_tensors_global": _all_reduce_int(len(common_grad_names), dist.ReduceOp.SUM),
        "num_common_state_tensors_global": _all_reduce_int(len(common_state_names), dist.ReduceOp.SUM),
        "rank_details": gathered_names if dist.get_rank() == 0 else [],
    }


def _collect_current_path_result(
    *,
    label: str,
    model: list[torch.nn.Module],
    loss: torch.Tensor,
    token_count: torch.Tensor,
    loss_log: dict[str, float],
    per_sample: list[dict[str, Any]],
    args: Any,
) -> LossPathResult:
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
                if selected:
                    if grad_buffer is not None:
                        param_copy = torch.nn.Parameter(param.detach().clone())
                        _manual_update_param(param_copy, grad_buffer, args)
                        selected_states[name] = param_copy.detach().float().cpu()
                    else:
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
        "token_count_min": _all_reduce_float(float(token_count.detach().item()), dist.ReduceOp.MIN),
        "token_count_max": _all_reduce_float(float(token_count.detach().item()), dist.ReduceOp.MAX),
    }
    local_stats = {
        "rank": dist.get_rank(),
        "loss": float(loss.detach().item()),
        "token_count": float(token_count.detach().item()),
        "num_parameters": local_num_params,
        "num_params_with_grad": local_num_params_with_grad,
        "num_grad_elements": local_num_grad_elems,
        "num_selected_tensors": local_num_selected,
        "nonfinite_grad_tensors": local_nonfinite_grad_tensors,
        "max_grad_abs": local_max_grad_abs,
        "grad_l2_norm": local_grad_norm_sq ** 0.5,
        "selected_tensor_names": sorted(selected_states),
    }
    return LossPathResult(
        label=label,
        loss=local_stats["loss"],
        token_count=local_stats["token_count"],
        loss_log=loss_log,
        local_stats=local_stats,
        global_stats=global_stats,
        selected_grads=selected_grads,
        selected_states=selected_states,
        per_sample=per_sample,
    )


def _run_shared_forward_paths(args: Any, base_rollout_data: dict[str, list[Any]]) -> tuple[LossPathResult, LossPathResult]:
    os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = args.impl
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    for module in model:
        module.train()
    _prepare_manual_grad_buffers(model, args)

    parallel_state = create_megatron_parallel_state(model)
    if parallel_state.tp_size != 1 or parallel_state.cp_size != 1:
        raise ValueError("SFT loss train reference expects TP=1 and CP=1")

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

    output_tensor = model[0](**forward_kwargs)
    if not torch.isfinite(output_tensor).all():
        raise RuntimeError("shared forward output contains non-finite values")

    miles_loss, miles_normalizer, loss_log = loss_function(
        args,
        parallel_state,
        batch,
        num_microbatches=1,
        logits=output_tensor,
        apply_megatron_loss_scaling=False,
    )
    explicit = _explicit_sft_reference(logits=output_tensor, batch=batch, args=args)
    explicit_loss = explicit["loss_sum"]
    explicit_normalizer = explicit["token_count"]
    if not torch.isfinite(miles_loss).all() or not torch.isfinite(explicit_loss).all():
        raise RuntimeError("shared forward loss contains non-finite values")

    miles_loss.backward(retain_graph=True)
    miles_result = _collect_current_path_result(
        label="miles_loss_function",
        model=model,
        loss=miles_loss,
        token_count=miles_normalizer,
        loss_log=_summarize_log(loss_log),
        per_sample=[],
        args=args,
    )

    _zero_manual_grad_buffers(model)
    explicit_loss.backward()
    explicit_result = _collect_current_path_result(
        label="explicit_sft_formula",
        model=model,
        loss=explicit_loss,
        token_count=explicit_normalizer,
        loss_log={
            "count": float(explicit_normalizer.detach().item()),
            "loss": float(explicit_loss.detach().item()),
        },
        per_sample=explicit["per_sample"],
        args=args,
    )

    _zero_manual_grad_buffers(model)
    del model, output_tensor, miles_loss, explicit_loss, batch, data_iterator
    torch.cuda.empty_cache()
    return miles_result, explicit_result


def _loss_global_stats(loss: torch.Tensor, token_count: torch.Tensor) -> dict[str, float]:
    return {
        "loss_min": _all_reduce_float(float(loss.detach().item()), dist.ReduceOp.MIN),
        "loss_max": _all_reduce_float(float(loss.detach().item()), dist.ReduceOp.MAX),
        "token_count_min": _all_reduce_float(float(token_count.detach().item()), dist.ReduceOp.MIN),
        "token_count_max": _all_reduce_float(float(token_count.detach().item()), dist.ReduceOp.MAX),
    }


def _collect_delta_comparison(model: list[torch.nn.Module], args: Any) -> dict[str, Any]:
    if args.manual_update_rule != "sgd":
        raise ValueError("loss-delta update comparison currently supports --manual-update-rule sgd")

    local_grad_max_abs = 0.0
    local_grad_max_name = ""
    local_grad_norm_sq = 0.0
    local_param_norm_sq = 0.0
    local_state_max_abs = 0.0
    local_state_max_name = ""
    local_num_selected = 0
    local_num_selected_with_grad = 0
    local_nonfinite_grad_tensors = 0

    with torch.no_grad():
        for module in model:
            for name, param in module.named_parameters():
                if not _is_selected_param(name, param, args.max_selected_numel):
                    continue
                local_num_selected += 1
                local_param_norm_sq += float(param.detach().float().square().sum().item())
                grad_buffer = _grad_buffer(param)
                if grad_buffer is None:
                    continue
                local_num_selected_with_grad += 1
                grad = grad_buffer.detach().float()
                if not torch.isfinite(grad).all():
                    local_nonfinite_grad_tensors += 1
                grad_abs = float(grad.abs().max().item()) if grad.numel() else 0.0
                if grad_abs > local_grad_max_abs:
                    local_grad_max_abs = grad_abs
                    local_grad_max_name = name
                local_grad_norm_sq += float(grad.square().sum().item())

                state_delta = grad.to(dtype=param.dtype).mul(args.manual_sgd_lr).float()
                state_abs = float(state_delta.abs().max().item()) if state_delta.numel() else 0.0
                if state_abs > local_state_max_abs:
                    local_state_max_abs = state_abs
                    local_state_max_name = name

    grad_norm = _all_reduce_float(local_grad_norm_sq, dist.ReduceOp.SUM) ** 0.5
    param_norm = _all_reduce_float(local_param_norm_sq, dist.ReduceOp.SUM) ** 0.5
    rel_to_param = 0.0 if param_norm == 0.0 else grad_norm / param_norm

    gathered_names: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(
        gathered_names,
        {
            "rank": dist.get_rank(),
            "grad_delta_max_abs_name": local_grad_max_name,
            "state_delta_max_abs_name": local_state_max_name,
            "num_selected_tensors": local_num_selected,
            "num_selected_tensors_with_grad": local_num_selected_with_grad,
            "nonfinite_grad_tensors": local_nonfinite_grad_tensors,
        },
    )
    return {
        "label": "miles_loss_function_minus_explicit_sft_formula_delta",
        "selected_grad_max_abs": _all_reduce_float(local_grad_max_abs, dist.ReduceOp.MAX),
        "selected_grad_max_rel_gap": rel_to_param,
        "selected_grad_delta_l2_norm": grad_norm,
        "selected_grad_delta_relative_to_param_l2": rel_to_param,
        "selected_state_max_abs": _all_reduce_float(local_state_max_abs, dist.ReduceOp.MAX),
        "selected_state_max_rel_gap": 0.0,
        "num_common_grad_tensors_global": _all_reduce_int(local_num_selected_with_grad, dist.ReduceOp.SUM),
        "num_common_state_tensors_global": _all_reduce_int(local_num_selected, dist.ReduceOp.SUM),
        "nonfinite_grad_tensors_global": _all_reduce_int(local_nonfinite_grad_tensors, dist.ReduceOp.SUM),
        "rank_details": gathered_names if dist.get_rank() == 0 else [],
    }


def _run_loss_delta_reference(args: Any, base_rollout_data: dict[str, list[Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[Any]]:
    os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = args.impl
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
    for module in model:
        module.train()
    _prepare_manual_grad_buffers(model, args)

    parallel_state = create_megatron_parallel_state(model)
    if parallel_state.tp_size != 1 or parallel_state.cp_size != 1:
        raise ValueError("SFT loss train reference expects TP=1 and CP=1")

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

    output_tensor = model[0](**forward_kwargs)
    if not torch.isfinite(output_tensor).all():
        raise RuntimeError("shared forward output contains non-finite values")

    miles_loss, miles_normalizer, loss_log = loss_function(
        args,
        parallel_state,
        batch,
        num_microbatches=1,
        logits=output_tensor,
        apply_megatron_loss_scaling=False,
    )
    explicit = _explicit_sft_reference(logits=output_tensor, batch=batch, args=args)
    explicit_loss = explicit["loss_sum"]
    explicit_normalizer = explicit["token_count"]
    if not torch.isfinite(miles_loss).all() or not torch.isfinite(explicit_loss).all():
        raise RuntimeError("shared forward loss contains non-finite values")

    _zero_manual_grad_buffers(model)
    loss_delta = miles_loss - explicit_loss
    loss_delta.backward()
    comparison = _collect_delta_comparison(model, args)
    comparison.update(
        {
            "loss_abs_local": abs(float(loss_delta.detach().item())),
            "loss_abs_global_max": _all_reduce_float(abs(float(loss_delta.detach().item())), dist.ReduceOp.MAX),
            "token_count_abs_local": abs(float((miles_normalizer - explicit_normalizer).detach().item())),
            "token_count_abs_global_max": _all_reduce_float(
                abs(float((miles_normalizer - explicit_normalizer).detach().item())),
                dist.ReduceOp.MAX,
            ),
        }
    )

    miles_path = {
        "loss": float(miles_loss.detach().item()),
        "token_count": float(miles_normalizer.detach().item()),
        "loss_log": _summarize_log(loss_log),
        "global_stats": _loss_global_stats(miles_loss, miles_normalizer),
    }
    explicit_path = {
        "loss": float(explicit_loss.detach().item()),
        "token_count": float(explicit_normalizer.detach().item()),
        "loss_log": {
            "count": float(explicit_normalizer.detach().item()),
            "loss": float(explicit_loss.detach().item()),
        },
        "global_stats": _loss_global_stats(explicit_loss, explicit_normalizer),
        "per_sample": explicit["per_sample"],
    }

    gathered_local_stats: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(
        gathered_local_stats,
        {
            "rank": dist.get_rank(),
            "miles_loss": miles_path["loss"],
            "explicit_loss": explicit_path["loss"],
            "loss_delta": float(loss_delta.detach().item()),
            "token_count": miles_path["token_count"],
            "selected_grad_delta_max_abs": comparison["selected_grad_max_abs"],
        },
    )

    _zero_manual_grad_buffers(model)
    del model, output_tensor, miles_loss, explicit_loss, loss_delta, batch, data_iterator
    torch.cuda.empty_cache()
    return miles_path, explicit_path, comparison, gathered_local_stats if dist.get_rank() == 0 else []


def main() -> int:
    configure_logger()
    _init_distributed()
    args = _parse_args()
    try:
        init(args)
        base_rollout_data = _load_rollout_data(args.rollout_data, args.max_samples)
        miles_path, explicit_path, comparison, gathered_local_stats = _run_loss_delta_reference(args, base_rollout_data)

        failures: list[str] = []
        if comparison["nonfinite_grad_tensors_global"]:
            failures.append("comparison.nonfinite_grad_tensors")
        if comparison["loss_abs_global_max"] > args.max_loss_abs:
            failures.append("comparison.loss_abs_global_max")
        if comparison["token_count_abs_global_max"] > args.max_token_count_abs:
            failures.append("comparison.token_count_abs_global_max")
        if comparison["selected_grad_max_abs"] > args.max_selected_grad_abs:
            failures.append("comparison.selected_grad_max_abs")
        if comparison["selected_grad_max_rel_gap"] > args.max_selected_grad_rel_gap:
            failures.append("comparison.selected_grad_max_rel_gap")
        if comparison["selected_state_max_abs"] > args.max_selected_state_abs:
            failures.append("comparison.selected_state_max_abs")
        if comparison["num_common_grad_tensors_global"] <= 0:
            failures.append("comparison.no_common_grad_tensors")
        if comparison["num_common_state_tensors_global"] <= 0:
            failures.append("comparison.no_common_state_tensors")

        payload = {
            "date": "2026-05-31",
            "seed": SEED,
            "status": "PASS" if not failures else "FAIL",
            "scope": "4-layer DeepSeek-V4 mini checkpoint SFT loss backward/update explicit PyTorch reference",
            "boundary": (
                "Shares the Miles/Megatron model forward path; proves SFT objective/backward/update "
                "against an explicit PyTorch loss formula by backpropagating their loss delta, not a "
                "monolithic external model-forward reference."
            ),
            "checkpoint": Path(args.load).name if args.load else None,
            "rollout_data_name": args.rollout_data.name,
            "attention_impl": args.impl,
            "world_size": dist.get_world_size(),
            "max_samples": args.max_samples,
            "reference_formula": "sum(-log_softmax(response_logits)[target_token] * loss_mask)",
            "runtime": {
                "deterministic_mode": bool(getattr(args, "deterministic_mode", False)),
                "NCCL_ALGO": os.getenv("NCCL_ALGO"),
                "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                "CUDA_DEVICE_MAX_CONNECTIONS": os.getenv("CUDA_DEVICE_MAX_CONNECTIONS"),
                "MEGATRON_USE_KV_QAT": os.getenv("MEGATRON_USE_KV_QAT"),
            },
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
                "max_token_count_abs": args.max_token_count_abs,
                "max_selected_grad_abs": args.max_selected_grad_abs,
                "max_selected_grad_rel_gap": args.max_selected_grad_rel_gap,
                "max_selected_state_abs": args.max_selected_state_abs,
            },
            "paths": {
                "miles_loss_function": miles_path,
                "explicit_sft_formula": explicit_path,
            },
            "comparison": comparison,
            "local_stats_by_rank": gathered_local_stats,
            "failures": failures,
        }
        if dist.get_rank() == 0:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2, sort_keys=True))
            print(f"wrote {args.json_output}")
        dist.barrier()
        return 0 if not failures else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")
    raise SystemExit(main())

#!/usr/bin/env python3
"""DeepSeek-V4 mini checkpoint 的 SFT loss 显式参考检查。

本校验器加载 4 层 mini checkpoint 与固定的 rollout 批次，运行一次
Miles/Megatron 前向，并用显式的 PyTorch 公式重新计算 SFT 负对数似然：

    log_softmax(response_logits).gather(target_tokens) * loss_mask

该检查把 SFT loss 与模型前向隔离开来。它并不证明后端 logprob 的严格
一致性；它证明的是：训练步中使用的 SFT loss 在已加载的 mini checkpoint
logits 上与一个外部公式一致。
"""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List

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
from miles.backends.training_utils.loss import get_log_probs_and_entropy, loss_function
from miles.utils.logging_utils import configure_logger


SEED = 20260531


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--impl", choices=["dense", "sparse", "tilelang"], default="dense")
    parser.add_argument("--qkv-format", choices=["thd", "bshd"], default="thd")
    parser.add_argument("--data-pad-size-multiplier", type=int, default=128)
    parser.add_argument("--log-probs-chunk-size", type=int, default=-1)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--allgather-cp", action="store_true")
    parser.add_argument("--true-on-policy-mode", action="store_true")
    parser.add_argument("--use-rollout-entropy", action="store_true")
    parser.add_argument("--use-dynamic-batch-size", action="store_true")
    parser.add_argument("--max-loss-abs", type=float, default=2e-3)
    parser.add_argument("--max-logprob-abs", type=float, default=2e-3)
    parser.add_argument("--max-logprob-mean-abs", type=float, default=2e-5)
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
        device_id=torch.device("cuda:{}".format(local_rank)),
    )


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


def _load_rollout_data(path: Path, max_samples: int) -> Dict[str, List[Any]]:
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
        raise ValueError("Unsupported rollout data format: {}".format(path))

    for key in ("tokens", "response_lengths", "loss_masks", "total_lengths"):
        if key not in rollout_data:
            raise KeyError("rollout data missing {}".format(key))

    out = {}
    n = min(max_samples, len(rollout_data["tokens"]))
    for key, value in rollout_data.items():
        if isinstance(value, list):
            out[key] = value[:n]
        else:
            out[key] = value
    return out


def _move_rollout_to_device(rollout_data: Dict[str, List[Any]], device: torch.device) -> Dict[str, List[Any]]:
    out = copy.copy(rollout_data)
    for key in ("tokens", "loss_masks", "log_probs", "ref_log_probs", "advantages", "returns", "rollout_log_probs"):
        if key in out and isinstance(out[key], list):
            out[key] = [value.to(device=device) if torch.is_tensor(value) else value for value in out[key]]
    return out


def _summarize_log(loss_log: Dict[str, Any]) -> Dict[str, float]:
    keys = loss_log.get("keys", [])
    values = loss_log.get("values")
    if not isinstance(keys, list) or not torch.is_tensor(values):
        return {}
    vals = values.detach().float().cpu().tolist()
    result = {"count": float(vals[0])} if vals else {}
    for key, value in zip(keys, vals[1:]):
        result[str(key)] = float(value)
    return result


def _explicit_sft_reference(
    *,
    logits: torch.Tensor,
    batch: Dict[str, Any],
    args: Any,
) -> Dict[str, Any]:
    if args.qkv_format != "thd":
        raise ValueError("explicit SFT reference currently expects qkv_format=thd")
    if logits.shape[0] != 1:
        raise ValueError("expected logits shape [1, T, V], got {}".format(tuple(logits.shape)))

    logits_2d = logits.squeeze(0).float().div(float(args.rollout_temperature))
    offset = 0
    explicit_log_probs = []
    explicit_weighted_nll_sum = logits_2d.new_tensor(0.0)
    explicit_token_count = logits_2d.new_tensor(0.0)
    per_sample = []

    for idx, (tokens, total_length, response_length, loss_mask) in enumerate(
        zip(
            batch["unconcat_tokens"],
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
        )
    ):
        total_length = int(total_length)
        response_length = int(response_length)
        prompt_length = total_length - response_length
        start = offset + prompt_length
        end = offset + total_length
        response_logits = logits_2d[start - 1 : end - 1]
        targets = tokens[-response_length:].to(device=response_logits.device, dtype=torch.long)
        if response_logits.shape[0] != targets.shape[0]:
            raise ValueError(
                "sample {} response shape mismatch: logits={} targets={}".format(
                    idx,
                    tuple(response_logits.shape),
                    tuple(targets.shape),
                )
            )

        log_probs = torch.log_softmax(response_logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        mask = loss_mask.to(device=log_probs.device, dtype=log_probs.dtype)
        weighted_nll = -(log_probs * mask).sum()
        token_count = mask.sum()
        explicit_weighted_nll_sum = explicit_weighted_nll_sum + weighted_nll
        explicit_token_count = explicit_token_count + token_count
        explicit_log_probs.append(log_probs)
        per_sample.append(
            {
                "sample": idx,
                "total_length": total_length,
                "response_length": response_length,
                "loss_tokens": float(token_count.detach().item()),
                "explicit_loss_sum": float(weighted_nll.detach().item()),
            }
        )
        offset += total_length

    return {
        "loss_sum": explicit_weighted_nll_sum,
        "token_count": explicit_token_count,
        "log_probs": torch.cat(explicit_log_probs, dim=0),
        "per_sample": per_sample,
    }


def _tensor_gap(left: torch.Tensor, right: torch.Tensor) -> Dict[str, float]:
    left = left.detach().float().flatten()
    right = right.detach().float().flatten()
    diff = (left - right).abs()
    denom = float((left.square().sum() + right.square().sum()).item())
    rel = 0.0 if denom == 0.0 else float(1.0 - 2.0 * float((left * right).sum().item()) / denom)
    return {
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "p99_abs": float(diff.quantile(0.99).item()) if diff.numel() else 0.0,
        "relative_l2_gap": rel,
        "numel": int(diff.numel()),
    }


def _all_reduce_float(value: float, op: dist.ReduceOp) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _all_reduce_int(value: int, op: dist.ReduceOp) -> int:
    tensor = torch.tensor(value, dtype=torch.int64, device="cuda")
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


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

        parallel_state = create_megatron_parallel_state(model)
        if parallel_state.tp_size != 1 or parallel_state.cp_size != 1:
            raise ValueError("explicit SFT reference expects TP=1 and CP=1")

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

        logits = model[0](**forward_kwargs)
        if not torch.isfinite(logits).all():
            raise RuntimeError("model logits contain non-finite values")

        miles_loss, normalizer, loss_log = loss_function(
            args,
            parallel_state,
            batch,
            num_microbatches=1,
            logits=logits,
            apply_megatron_loss_scaling=False,
        )
        miles_log_probs = get_log_probs_and_entropy(
            logits,
            args=args,
            parallel_state=parallel_state,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=batch["total_lengths"],
            response_lengths=batch["response_lengths"],
            with_entropy=False,
            max_seq_lens=batch.get("max_seq_lens", None),
        )["log_probs"]
        miles_log_probs = torch.cat(miles_log_probs, dim=0)
        explicit = _explicit_sft_reference(logits=logits, batch=batch, args=args)

        logprob_gap = _tensor_gap(miles_log_probs, explicit["log_probs"])
        loss_abs = abs(float(miles_loss.detach().item()) - float(explicit["loss_sum"].detach().item()))
        token_count_abs = abs(float(normalizer.detach().item()) - float(explicit["token_count"].detach().item()))

        local_failures = []
        if loss_abs > args.max_loss_abs:
            local_failures.append("loss_abs")
        if logprob_gap["max_abs"] > args.max_logprob_abs:
            local_failures.append("logprob_max_abs")
        if logprob_gap["mean_abs"] > args.max_logprob_mean_abs:
            local_failures.append("logprob_mean_abs")
        if token_count_abs != 0.0:
            local_failures.append("token_count_abs")

        gathered: List[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            gathered,
            {
                "rank": dist.get_rank(),
                "miles_loss": float(miles_loss.detach().item()),
                "explicit_loss_sum": float(explicit["loss_sum"].detach().item()),
                "loss_abs": loss_abs,
                "miles_token_count": int(normalizer.detach().item()),
                "explicit_token_count": float(explicit["token_count"].detach().item()),
                "token_count_abs": token_count_abs,
                "logprob_gap": logprob_gap,
                "loss_log": _summarize_log(loss_log),
                "per_sample": explicit["per_sample"],
                "failures": local_failures,
            },
        )
        all_failures = []
        for row in gathered:
            for failure in row["failures"]:
                all_failures.append("rank{}.{}".format(row["rank"], failure))
        global_summary = {
            "loss_abs_global_max": _all_reduce_float(loss_abs, dist.ReduceOp.MAX),
            "logprob_max_abs_global_max": _all_reduce_float(logprob_gap["max_abs"], dist.ReduceOp.MAX),
            "logprob_mean_abs_global_max": _all_reduce_float(logprob_gap["mean_abs"], dist.ReduceOp.MAX),
            "token_count_abs_global_max": _all_reduce_float(token_count_abs, dist.ReduceOp.MAX),
            "num_logprob_tokens_global_sum": _all_reduce_int(logprob_gap["numel"], dist.ReduceOp.SUM),
        }

        payload = {
            "date": "2026-05-31",
            "seed": SEED,
            "status": "PASS" if not all_failures else "FAIL",
            "scope": "4-layer DeepSeek-V4 mini checkpoint SFT loss explicit PyTorch reference",
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
            "thresholds": {
                "max_loss_abs": args.max_loss_abs,
                "max_logprob_abs": args.max_logprob_abs,
                "max_logprob_mean_abs": args.max_logprob_mean_abs,
            },
            "global_summary": global_summary,
            "per_rank": gathered if dist.get_rank() == 0 else [],
            "failures": all_failures,
        }
        if dist.get_rank() == 0:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2, sort_keys=True))
            print("wrote {}".format(args.json_output))
        dist.barrier()
        return 0 if not all_failures else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")
    raise SystemExit(main())

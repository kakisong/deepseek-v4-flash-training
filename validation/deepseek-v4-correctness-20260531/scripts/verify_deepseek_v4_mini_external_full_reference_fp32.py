#!/usr/bin/env python3
"""DeepSeek-V4 已加载 mini checkpoint 的 FP32 严格外部全模型参考。

目的
----
编写本校验器是为了检验一个真正 FP32 的 Miles DeepSeek-V4 运行时能否
关闭 BF16 全外部参考刻意保留记录的两个尚未关闭的严格边界：

* ``strict_mini_backend_logprob_parity``（FAIL），以及
* ``external_reference_mini_checkpoint_one_step_train_parity``（FAIL_DIAGNOSTIC）。

在 BF16 下，这两个失败由*同一个*物理效应导致，而不是逻辑错误：

1. 不同的 attention 内核（dense / sparse / tilelang）以不同的浮点顺序
   累加相同的数学运算，产生约 1 ULP 的 BF16 输出漂移。
2. 该漂移在 MoE top-k 路由边界被放大：少数 token 距 top-k 平局足够近，
   导致离散的专家选择发生翻转，于是少量 per-token logprob 偏移达到
   ``O(0.1)``，而整体分布仍在 ``relative_l2 ~ 1.5e-5`` 水平上保持一致。
3. 将该残余前向漂移反向传播穿过完整的 4 层计算图时，会在小范数的
   normalization 参数（``q_norm.weight``）上爆炸，因为
   ``|grad_miles - grad_ref| / |param|`` 按构造就很大。

因此这里待检验的假设是精确且可证伪的：

    **如果残差纯粹是 BF16 舍入，那么以 FP32 运行*完全相同*的数学运算
    必然使严格差距坍缩：score 路由层停止翻转（独立路由下 router map
    完全一致），logit/loss 差距降至 FP32-epsilon 量级，并且 selected
    gradient 严格 delta 在不做任何路由 replay 的情况下回落到其自身的
    0.01 阈值之下。**

如果发生了这种坍缩，BF16 的 ``FAIL`` 就可以诚实地升级为
"FP32 严格 PASS；BF16 是有记录的精度边界"。如果在 FP32 下*没有*
坍缩，那正是这套脚手架要捕捉的真实 bug 信号。

2026-06-01 尝试的 8-rank 运行发现了一个更早的边界：当前的 Miles
DeepSeek-V4 生产运行时无法以真正的 FP32 实例化这条路径。
``DeepSeekV4Attention`` 在模型构造期间断言投影权重为 BF16，并且若干
DeepSeek-V4 内核/辅助路径假定输入为 BF16。参见
``artifacts/deepseek-v4-fp32-strict-closure-attempt-20260601.json``。

设计
----
本脚本逐字复用 ``verify_deepseek_v4_mini_external_full_reference``
中已验证的显式参考数学（embedding、四个 DeepSeek-V4 层、dense attention
参考、EP=8 hash 路由/score 路由 MoE、最终 norm、输出头、SFT loss，以及
selected 非专家 backward/update delta）。仅有的差异是：

* Miles/Megatron 模型与显式参考都被强制为 FP32
  （``--bf16``/``--fp16``/``--fp8`` 关闭，KV-QAT 量化默认关闭），从而使
  比较只隔离浮点精度本身，而不是 QAT 或 block-scaling 路径；
* 路由默认是**独立的**（不做 Miles router-map replay）：强论断是 FP32
  会消除分支翻转，因此 replay 必须是不必要的；
* 捕获每层的 router map，若任何 router map 在 FP32 下仍发生翻转，
  artifact 记为 FAIL（可通过 ``--allow-router-map-flip`` 配置）；
* 阈值默认采用 FP32 严格档位，并由 ``fp32_mode_not_realized`` 守卫在
  模型并未真正以 FP32 加载时拒绝输出 PASS。

完整的 local-expert EP all-to-all backward/update 仍由专用的真实 EP=8
MoELayer 参考覆盖，原因与 BF16 校验器中给出的分布式归约原因相同；
专家参数被排除在此处的 selected backward 检查之外。

状态：为可复现性而保留的诊断校验器。在当前 Miles DeepSeek-V4 运行时上，
若它真正强制 FP32，预期会在 checkpoint 加载之前就失败；该失败即为记录
的结果，而不是数值一致性结果。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_model

import miles_plugins.mbridge  # noqa: F401
from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.backends.megatron_utils.parallel import create_megatron_parallel_state, get_packed_seq_params
from miles.backends.training_utils.data import DataIterator, get_batch
from miles.utils.logging_utils import configure_logger

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from tools import verify_deepseek_v4_mini_external_full_reference as base
except (ImportError, ModuleNotFoundError):  # pragma: no cover - 集群上以裸路径方式执行
    import verify_deepseek_v4_mini_external_full_reference as base


# FP32 严格容差档位。这些阈值刻意收得很紧：重点正是要证明对*相同*数学
# 运算的 FP32 运行能使 BF16 差距坍缩。selected gradient 阈值刻意保持为
# BF16 校验器的取值（1e-2），这样"现在通过了"才是一个有意义的、
# 同口径可比的结果，而不是被挪动的门槛。
FP32_THRESHOLDS = {
    "max_logit_abs": 5e-2,
    "max_logit_mean_abs": 2e-3,
    "max_logit_p99_abs": 1e-2,
    "max_logit_rel_gap": 5e-6,
    "max_loss_abs": 1.0,            # sum-loss 守卫；下面的 per-token 才是起约束作用的门控
    "max_loss_abs_per_token": 1e-4,
    "max_token_count_abs": 0.0,
    "max_selected_grad_abs": 1e-2,
    "max_selected_grad_rel_gap": 1e-2,
    "max_selected_state_abs": 2e-5,
}


def _add_fp32_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    base.add_args(parser)
    parser.add_argument(
        "--allow-router-map-flip",
        action="store_true",
        help="Do not FAIL if a score-routed token still flips experts in FP32 "
        "(records the residual flip as a diagnostic instead of closing the gate).",
    )
    parser.add_argument(
        "--keep-kv-qat",
        action="store_true",
        help="Keep MEGATRON_USE_KV_QAT enabled. Off by default in FP32 mode so "
        "the comparison isolates float precision, not KV quantization.",
    )
    return parser


def _parse_args() -> argparse.Namespace:
    args = parse_args(extra_args_provider=_add_fp32_args)
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

    # --- 全程强制 FP32 -------------------------------------------------------
    args.bf16 = False
    args.fp16 = False
    args.fp8 = None
    args.fp8_recipe = None
    # 独立路由是这里的强论断：FP32 应当使 replay 变得不必要。
    args.replay_miles_routing = False
    # 捕获每层输出（含 router map），以便断言差距坍缩。
    args.debug_layer_gaps = True
    # 收紧的 FP32 严格阈值（本校验器的唯一权威来源）。
    for key, value in FP32_THRESHOLDS.items():
        setattr(args, key, value)
    args.tolerance_profile = "fp32_strict_external_reference"

    validate_args(args)
    args.variable_seq_lengths = True
    return args


def _router_map_gaps(layer_gaps: dict[str, Any] | None) -> dict[str, Any]:
    if not layer_gaps:
        return {"checked": 0, "max_router_map_abs": None, "flipped_layers": []}
    flipped = []
    worst = 0.0
    checked = 0
    for name, gap in layer_gaps.items():
        if not name.endswith(".router_map"):
            continue
        checked += 1
        value = gap.get("max_abs")
        if not isinstance(value, (int, float)):
            continue
        worst = max(worst, float(value))
        if float(value) > 0.0:
            flipped.append({"layer": name, "max_abs": float(value), "relative_l2_gap": gap.get("relative_l2_gap")})
    return {"checked": checked, "max_router_map_abs": worst, "flipped_layers": flipped}


def main() -> int:
    configure_logger()
    base._init_distributed()
    args = _parse_args()
    if not args.keep_kv_qat:
        os.environ["MEGATRON_USE_KV_QAT"] = "0"
    try:
        init(args)
        os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = args.impl
        torch.manual_seed(base.SEED)
        torch.cuda.manual_seed_all(base.SEED)

        model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
        base.load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)
        for module in model:
            module.train()
        base._prepare_manual_grad_buffers(model, args)
        raw_model = base._unwrap_model(model[0])
        model_config = raw_model.config

        # 硬性守卫：若 FP32 并未真正生效，则拒绝输出 PASS。
        param_dtypes = sorted({str(p.dtype) for p in model[0].parameters()})
        fp32_realized = param_dtypes == ["torch.float32"]

        parallel_state = create_megatron_parallel_state(model)
        ep_size = int(getattr(args, "expert_model_parallel_size", 1))
        if parallel_state.tp_size != 1 or parallel_state.cp_size != 1 or ep_size != dist.get_world_size():
            raise ValueError(
                "FP32 full reference expects TP=1, CP=1, and EP equal to world size; "
                f"got TP={parallel_state.tp_size} CP={parallel_state.cp_size} EP={ep_size} "
                f"world={dist.get_world_size()}"
            )

        device = torch.device("cuda", torch.cuda.current_device())
        rollout_data = base._move_rollout_to_device(
            base._load_rollout_data(args.rollout_data, args.max_samples), device
        )
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
            raise ValueError("FP32 full external reference currently expects text-only SFT batches")

        forward_kwargs = {
            "input_ids": batch["tokens"],
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": get_packed_seq_params(batch, args),
            "loss_mask": batch["full_loss_masks"],
        }
        # 独立路由 => 捕获 Miles 的 router map，与参考实现独立计算出的
        # 路由 map 进行比较。
        miles_traces, hooks = base._capture_decoder_layer_outputs(
            model[0],
            first_layer_submodules=False,
            all_layer_submodules=False,
            capture_router=True,
        )
        try:
            miles_logits = model[0](**forward_kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        if not torch.isfinite(miles_logits).all():
            raise RuntimeError("Miles logits contain non-finite values")

        ref_traces: dict[str, torch.Tensor] = {}
        with base._reference_saved_tensor_context(not args.no_offload_reference_saved_tensors):
            ref_logits = base._full_reference(
                model[0],
                batch["tokens"],
                trace=ref_traces,
                replay_trace=None,  # 独立路由
            )
            if not torch.isfinite(ref_logits).all():
                raise RuntimeError("reference logits contain non-finite values")
            explicit_ref = base._explicit_sft_reference(logits=ref_logits, batch=batch, args=args)

        miles_loss, normalizer, loss_log = base.loss_function(
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
        logit_gap = base._tensor_gap(miles_logits, ref_logits)

        if args.skip_backward_check:
            grad_stats: dict[str, Any] = {"status": "SKIPPED_FORWARD_ONLY"}
        else:
            base._zero_manual_grad_buffers(model)
            delta = miles_loss - explicit_ref["loss_sum"]
            delta.backward()
            grad_stats = base._selected_delta_stats(model, args)

        layer_gaps: dict[str, Any] = {}
        for name in sorted(ref_traces):
            if name in miles_traces:
                layer_gaps[name] = base._global_tensor_gap(miles_traces[name], ref_traces[name])
        router_summary = _router_map_gaps(layer_gaps)

        local_failures: list[str] = []
        if not fp32_realized:
            local_failures.append("fp32_mode_not_realized")
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
        if loss_abs_per_token > args.max_loss_abs_per_token:
            local_failures.append("loss_abs_per_token")
        if token_count_abs > args.max_token_count_abs:
            local_failures.append("token_count_abs")
        if getattr(args, "fp8", None):
            local_failures.append("fp8_set_in_fp32_mode")
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

        # 核心论断：FP32 消除了 score 路由的分支翻转，因此独立路由应当
        # 完全一致。残余翻转是致命的，除非显式允许（此时只作为诊断信息
        # 记录下来）。
        if router_summary["flipped_layers"] and not args.allow_router_map_flip:
            if "router_map_flip_in_fp32" not in failures:
                failures.append("router_map_flip_in_fp32")
                failures = sorted(failures)

        status = "PASS" if not failures else "FAIL"
        payload = {
            "date": "2026-05-31",
            "seed": base.SEED,
            "status": status,
            "scope": "DeepSeek-V4 loaded 4-layer mini checkpoint FP32 strict full-model external reference",
            "reference": (
                "explicit FP32 PyTorch embedding + 4 DeepSeek-V4 layers + dense attention reference + "
                "EP=8 hash-routed / score-routed MoE forward + output head + SFT loss"
            ),
            "hypothesis": (
                "If the BF16 strict logprob and selected-gradient failures are pure precision artifacts, "
                "the same math in FP32 collapses the gaps: independent routing stops flipping, logit/loss "
                "gaps fall to FP32-epsilon scale, and the selected-gradient strict delta passes its own "
                "0.01 threshold without routing replay."
            ),
            "world_size": dist.get_world_size(),
            "attention_impl": args.impl,
            "checkpoint_name": Path(str(args.load)).name if getattr(args, "load", None) else None,
            "rollout_data_name": args.rollout_data.name,
            "max_samples": args.max_samples,
            "fp32_mode_realized": fp32_realized,
            "param_dtypes": param_dtypes,
            "kv_qat_enabled": bool(args.keep_kv_qat),
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
                "vocab_size": base._model_vocab_size(model[0], args),
            },
            "thresholds": dict(FP32_THRESHOLDS),
            "precision_mode": {
                "miles_compute_dtype": "float32",
                "miles_fp8": str(getattr(args, "fp8", None)),
                "reference": "explicit FP32 PyTorch math; no BF16/FP8 quantization",
            },
            "routing_replay": {
                "mode": "independent_reference_routing",
                "description": "FP32 strict run uses independent routing; the strong claim is that FP32 "
                "removes the score-routed branch flips so Miles router-map replay is unnecessary.",
            },
            "tolerance_profile": args.tolerance_profile,
            "miles_loss": float(miles_loss.detach().item()),
            "reference_loss": float(explicit_ref["loss_sum"].detach().item()),
            "loss_abs_local": loss_abs,
            "loss_abs_global_max": base._all_reduce_float(loss_abs, dist.ReduceOp.MAX),
            "loss_abs_per_token_local": loss_abs_per_token,
            "loss_abs_per_token_global_max": base._all_reduce_float(loss_abs_per_token, dist.ReduceOp.MAX),
            "token_count_abs_local": token_count_abs,
            "token_count_abs_global_max": base._all_reduce_float(token_count_abs, dist.ReduceOp.MAX),
            "miles_loss_log": base._summarize_log(loss_log),
            "reference_formula": "full_model_reference_then_sum(-log_softmax(response_logits)[target_token] * loss_mask)",
            "logit_gap_local": logit_gap,
            "logit_gap_global": {
                "max_abs": base._all_reduce_float(logit_gap["max_abs"], dist.ReduceOp.MAX),
                "mean_abs": base._all_reduce_float(logit_gap["mean_abs"], dist.ReduceOp.MAX),
                "p99_abs": base._all_reduce_float(logit_gap["p99_abs"], dist.ReduceOp.MAX),
                "relative_l2_gap": base._all_reduce_float(logit_gap["relative_l2_gap"], dist.ReduceOp.MAX),
            },
            "router_map_collapse": router_summary,
            "layer_gap_global": layer_gaps,
            "selected_backward_update_delta": grad_stats,
            "manual_update": {"rule": args.manual_update_rule, "lr": args.manual_sgd_lr},
            "backward_update_check": "SKIPPED_FORWARD_ONLY" if args.skip_backward_check else "RUN",
            "failures": failures,
            "boundary": (
                "FP32 strict closure run for the loaded 4-layer mini checkpoint. A PASS here means the "
                "BF16 strict logprob parity and selected-gradient strict delta failures are pure precision "
                "artifacts: in FP32 the score-routed routing stops flipping and the gaps fall under the "
                "strict thresholds without any routing replay. A FAIL here is an honest signal that the "
                "residual is not purely precision and must be investigated, not a tolerance to be widened."
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
    os.environ.setdefault("MILES_DSV4_CKPT_VERSION", "0415")
    raise SystemExit(main())

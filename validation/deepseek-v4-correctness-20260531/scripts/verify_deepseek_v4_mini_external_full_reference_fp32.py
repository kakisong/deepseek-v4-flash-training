#!/usr/bin/env python3
"""DeepSeek-V4 loaded mini-checkpoint FP32 strict external full-model reference.

Purpose
-------
This verifier was written to test whether a true FP32 Miles DeepSeek-V4 runtime
could close the two remaining open strict boundaries that the BF16 full external
reference deliberately leaves recorded:

* ``strict_mini_backend_logprob_parity`` (FAIL) and
* ``external_reference_mini_checkpoint_one_step_train_parity`` (FAIL_DIAGNOSTIC).

In BF16 both failures are caused by the *same* physical effect, not by a logic
error:

1. Different attention kernels (dense / sparse / tilelang) accumulate the same
   math in different float orders, producing ~1 ULP BF16 output drift.
2. That drift is amplified at the MoE top-k routing boundary: a few tokens sit
   close enough to a top-k tie that the discrete expert choice flips, so a
   handful of per-token logprobs move by ``O(0.1)`` while the bulk distribution
   still agrees to ``relative_l2 ~ 1.5e-5``.
3. Back-propagating that residual forward drift through the full 4-layer graph
   blows up on small-norm normalization parameters (``q_norm.weight``) where
   ``|grad_miles - grad_ref| / |param|`` is large by construction.

The hypothesis under test here is therefore precise and falsifiable:

    **If the residual is purely BF16 rounding, then running the *identical*
    math in FP32 must make the strict gaps collapse: the score-routed layer
    stops flipping (router maps match exactly under independent routing), the
    logit/loss gaps fall to FP32-epsilon scale, and the selected-gradient
    strict delta drops back under its own 0.01 threshold without any routing
    replay.**

If that collapse happened, the BF16 ``FAIL`` could honestly upgrade to
"FP32 strict PASS; BF16 is a documented precision boundary". If it did *not*
collapse in FP32, that would be exactly the real-bug signal this scaffolding is
built to catch.

The attempted 8-rank run on 2026-06-01 found an earlier boundary: the current
Miles DeepSeek-V4 production runtime cannot instantiate this path in true FP32.
``DeepSeekV4Attention`` asserts BF16 projection weights during model
construction, and several DeepSeek-V4 kernel/helper paths assume BF16 input. See
``artifacts/deepseek-v4-fp32-strict-closure-attempt-20260601.json``.

Design
------
This script reuses the proven explicit reference math from
``verify_deepseek_v4_mini_external_full_reference`` verbatim (embedding, four
DeepSeek-V4 layers, dense attention reference, EP=8 hash-/score-routed MoE,
final norm, output head, SFT loss, and the selected non-expert backward/update
delta). The only differences are:

* the Miles/Megatron model and the explicit reference are both forced to FP32
  (``--bf16``/``--fp16``/``--fp8`` off, KV-QAT quantization off by default) so
  that the comparison isolates floating-point precision rather than the QAT or
  block-scaling paths;
* routing is **independent** by default (no Miles router-map replay): the strong
  claim is that FP32 removes the branch flips, so replay must be unnecessary;
* per-layer router maps are captured and the artifact FAILs if any router map
  still flips in FP32 (configurable via ``--allow-router-map-flip``);
* thresholds default to an FP32-strict profile, and a ``fp32_mode_not_realized``
  guard refuses to emit a PASS if the model did not actually load in FP32.

Full local-expert EP all-to-all backward/update remains covered by the dedicated
real EP=8 MoELayer reference, for the same distributed-reduction reason given in
the BF16 verifier; expert parameters are excluded from the selected backward
check here.

Status: diagnostic verifier kept for reproducibility. On the current Miles
DeepSeek-V4 runtime it is expected to fail before checkpoint load if it truly
forces FP32; that failure is the recorded result, not a numerical parity result.
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
except (ImportError, ModuleNotFoundError):  # pragma: no cover - bare-path execution on cluster
    import verify_deepseek_v4_mini_external_full_reference as base


# FP32-strict tolerance profile. These are intentionally tight: the whole point
# is that an FP32 run of the *same* math collapses the BF16 gaps. The selected
# gradient thresholds are deliberately left at the BF16 verifier's values
# (1e-2) so that "now it passes" is a meaningful, like-for-like result rather
# than a moved goalpost.
FP32_THRESHOLDS = {
    "max_logit_abs": 5e-2,
    "max_logit_mean_abs": 2e-3,
    "max_logit_p99_abs": 1e-2,
    "max_logit_rel_gap": 5e-6,
    "max_loss_abs": 1.0,            # sum-loss guard; per-token below is the binding gate
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

    # --- force FP32 everywhere ---------------------------------------------
    args.bf16 = False
    args.fp16 = False
    args.fp8 = None
    args.fp8_recipe = None
    # Independent routing is the strong claim: FP32 should make replay needless.
    args.replay_miles_routing = False
    # Capture per-layer outputs (incl. router maps) so we can assert collapse.
    args.debug_layer_gaps = True
    # Tight FP32-strict thresholds (single source of truth for this verifier).
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

        # Hard guard: refuse to emit a PASS if FP32 was not actually realized.
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
        # Independent routing => capture Miles router maps to compare against the
        # reference's independently computed routing maps.
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
                replay_trace=None,  # independent routing
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

        # The headline claim: FP32 removes the score-routed branch flips, so
        # independent routing matches exactly. A residual flip is fatal unless
        # explicitly allowed (then it is recorded as a diagnostic).
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

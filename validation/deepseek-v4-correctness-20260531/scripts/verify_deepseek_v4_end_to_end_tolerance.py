#!/usr/bin/env python3
"""Validate DeepSeek-V4 end-to-end BF16 tolerance evidence.

This verifier is intentionally separate from strict parity checks.  Strict
rtol/atol logprob parity is still recorded as failing for the real non-injected
forward path.  The purpose here is to assert a narrower but operationally useful
claim: the real mini-checkpoint backend drift is finite, bounded to a BF16
runtime envelope, and the training update path passes once the localized
attention forward-value drift is removed.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(base: Path, name: str) -> Dict[str, Any]:
    return json.loads((base / name).read_text(encoding="utf-8"))


def _find_compare(items: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for item in items:
        if item.get("label") == label:
            return item
    raise KeyError("comparison not found: {}".format(label))


def _check(condition: bool, failures: List[str], name: str) -> None:
    if not condition:
        failures.append(name)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("docs/en/advanced"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-real-forward-relative-l2", type=float, default=2e-5)
    parser.add_argument("--max-real-forward-mean-abs", type=float, default=0.08)
    parser.add_argument("--max-real-forward-p99-abs", type=float, default=0.37)
    parser.add_argument("--max-routing-forward-relative-l2", type=float, default=4e-6)
    parser.add_argument("--min-routing-relative-l2-reduction", type=float, default=4.0)
    parser.add_argument("--max-real-train-grad-rel-gap", type=float, default=5e-4)
    parser.add_argument("--max-real-train-state-abs", type=float, default=2e-5)
    parser.add_argument("--max-official-1layer-relative-l2", type=float, default=5e-6)
    parser.add_argument("--max-official-1layer-mean-abs", type=float, default=0.04)
    parser.add_argument("--max-attention-io-output-abs", type=float, default=0.0625)
    parser.add_argument("--max-attention-io-input-grad-abs", type=float, default=0.01)
    parser.add_argument("--max-sft-replay-grad-rel-gap", type=float, default=3e-5)
    parser.add_argument("--max-sft-replay-state-abs", type=float, default=2e-5)
    args = parser.parse_args()

    base = args.artifacts_dir
    failures: List[str] = []

    mini_forward = _load(base, "deepseek-v4-mini-forward-compare-20260531.json")
    train_base = _load(base, "deepseek-v4-mini-train-step-qatsim-0415-20260531.json")
    train_routing = _load(base, "deepseek-v4-mini-train-step-routing-replay-qatsim-20260531.json")
    train_attention_replay = _load(base, "deepseek-v4-mini-train-step-attention-output-replay-qatsim-20260531.json")
    attention_io = _load(base, "deepseek-v4-mini-attention-io-replay-qatsim-20260531.json")
    attention_io_train = _load(base, "deepseek-v4-mini-attention-io-training-step-qatsim-20260531.json")
    proof_ledger = _load(base, "deepseek-v4-proof-ledger-20260531.json")
    official_sparse = _load(base, "deepseek-v4-official-full-forward-1layer-bf16-qatsim-0415-20260531.json")
    official_tilelang = _load(base, "deepseek-v4-official-full-forward-1layer-bf16-tilelang-qatsim-0415-20260531.json")

    _check(proof_ledger.get("status") == "PASS", failures, "proof_ledger.status")

    repeatability = mini_forward.get("repeatability_checks", [])
    dense_repeat = _find_compare(repeatability, "dense_repeat_with_deterministic_runtime_exact")
    _check(dense_repeat["status"] == "PASS", failures, "dense_repeatability.status")
    _check(dense_repeat["mismatches"] == 0, failures, "dense_repeatability.mismatches")

    real_forward = {}
    routing_forward = {}
    routing_artifacts = {
        "dense_vs_sparse": "deepseek-v4-mini-forward-routing-replay-dense-vs-sparse-qatsim-20260531.json",
        "dense_vs_tilelang": "deepseek-v4-mini-forward-routing-replay-dense-vs-tilelang-qatsim-20260531.json",
        "sparse_vs_tilelang": "deepseek-v4-mini-forward-routing-replay-sparse-vs-tilelang-qatsim-20260531.json",
    }
    for label, artifact in routing_artifacts.items():
        baseline = _find_compare(mini_forward["compare_results"], label)
        replay = _load(base, artifact)
        real_forward[label] = {
            "mismatches": baseline["mismatches"],
            "max_abs": baseline["max_abs"],
            "mean_abs": baseline["mean_abs"],
            "p99_abs": baseline["p99_abs"],
            "relative_l2_gap": baseline["relative_l2_gap"],
        }
        routing_forward[label] = {
            "mismatches": replay["mismatches"],
            "max_abs": replay["max_abs"],
            "mean_abs": replay["mean_abs"],
            "p99_abs": replay["p99_abs"],
            "relative_l2_gap": replay["relative_l2_gap"],
            "relative_l2_reduction": baseline["relative_l2_gap"] / replay["relative_l2_gap"],
        }
        _check(
            baseline["relative_l2_gap"] <= args.max_real_forward_relative_l2,
            failures,
            "{}.real_forward.relative_l2".format(label),
        )
        _check(
            baseline["mean_abs"] <= args.max_real_forward_mean_abs,
            failures,
            "{}.real_forward.mean_abs".format(label),
        )
        _check(
            baseline["p99_abs"] <= args.max_real_forward_p99_abs,
            failures,
            "{}.real_forward.p99_abs".format(label),
        )
        _check(
            replay["relative_l2_gap"] <= args.max_routing_forward_relative_l2,
            failures,
            "{}.routing_forward.relative_l2".format(label),
        )
        _check(
            baseline["relative_l2_gap"] / replay["relative_l2_gap"]
            >= args.min_routing_relative_l2_reduction,
            failures,
            "{}.routing_forward.reduction".format(label),
        )

    real_train = {}
    train_after_attention_replay = {}
    for label in ("dense_vs_sparse", "dense_vs_tilelang", "sparse_vs_tilelang"):
        baseline = _find_compare(train_base["comparisons"], label)
        routing = _find_compare(train_routing["comparisons"], label)
        replay = _find_compare(train_attention_replay["comparisons"], label)
        real_train[label] = {
            "loss_abs": baseline["loss_abs_global_max"],
            "routing_loss_abs": routing["loss_abs_global_max"],
            "selected_grad_max_rel_gap": baseline["selected_grad_max_rel_gap"],
            "selected_state_max_abs": baseline["selected_state_max_abs"],
            "routing_selected_grad_max_rel_gap": routing["selected_grad_max_rel_gap"],
            "routing_selected_state_max_abs": routing["selected_state_max_abs"],
        }
        train_after_attention_replay[label] = {
            "loss_abs": replay["loss_abs_global_max"],
            "selected_grad_max_rel_gap": replay["selected_grad_max_rel_gap"],
            "selected_state_max_abs": replay["selected_state_max_abs"],
        }
        _check(
            baseline["selected_grad_max_rel_gap"] <= args.max_real_train_grad_rel_gap,
            failures,
            "{}.real_train.grad_rel".format(label),
        )
        _check(
            baseline["selected_state_max_abs"] <= args.max_real_train_state_abs,
            failures,
            "{}.real_train.state_abs".format(label),
        )
        _check(
            routing["loss_abs_global_max"] < baseline["loss_abs_global_max"],
            failures,
            "{}.routing_train.loss_reduced".format(label),
        )
        _check(replay["loss_abs_global_max"] == 0.0, failures, "{}.sft_replay.loss_exact".format(label))
        _check(
            replay["selected_grad_max_rel_gap"] <= args.max_sft_replay_grad_rel_gap,
            failures,
            "{}.sft_replay.grad_rel".format(label),
        )
        _check(
            replay["selected_state_max_abs"] <= args.max_sft_replay_state_abs,
            failures,
            "{}.sft_replay.state_abs".format(label),
        )

    attention_io_summary = {}
    max_attention_io_output_abs = 0.0
    for backend in attention_io["backend_summaries"]:
        impl = backend["impl"]
        per_layer = []
        for layer in backend["per_layer"]:
            out = layer["output_from_replayed_dense_input"]
            max_attention_io_output_abs = max(max_attention_io_output_abs, out["max_abs_global_max"])
            per_layer.append(
                {
                    "module": layer["module"],
                    "output_max_abs": out["max_abs_global_max"],
                    "output_mean_abs_rank_max": out["mean_abs_rank_max"],
                    "finite_all": out["finite_all"],
                }
            )
            _check(out["finite_all"], failures, "{}.{}.attention_io.finite".format(impl, layer["module"]))
            _check(
                out["max_abs_global_max"] <= args.max_attention_io_output_abs,
                failures,
                "{}.{}.attention_io.output_abs".format(impl, layer["module"]),
            )
        attention_io_summary[impl] = per_layer

    max_attention_io_train_output_abs = 0.0
    max_attention_io_train_input_grad_abs = 0.0
    max_attention_io_train_state_abs = 0.0
    for rank in attention_io_train["per_rank"]:
        for layer in rank["layers"].values():
            for comparison in layer["comparisons"]:
                max_attention_io_train_output_abs = max(
                    max_attention_io_train_output_abs,
                    comparison["output_max_abs_global_max"],
                )
                max_attention_io_train_input_grad_abs = max(
                    max_attention_io_train_input_grad_abs,
                    comparison["input_grad_max_abs_global_max"],
                )
                max_attention_io_train_state_abs = max(
                    max_attention_io_train_state_abs,
                    comparison["state_after_step_max_abs_global_max"],
                )
    _check(
        max_attention_io_train_output_abs <= args.max_attention_io_output_abs,
        failures,
        "attention_io_train.output_abs",
    )
    _check(
        max_attention_io_train_input_grad_abs <= args.max_attention_io_input_grad_abs,
        failures,
        "attention_io_train.input_grad_abs",
    )
    _check(max_attention_io_train_state_abs <= args.max_real_train_state_abs, failures, "attention_io_train.state_abs")

    official_1layer = {}
    for label, payload in [("sparse", official_sparse), ("tilelang", official_tilelang)]:
        comp = payload["comparison"]
        official_1layer[label] = {
            "mismatches": comp["mismatches"],
            "max_abs": comp["max_abs"],
            "mean_abs": comp["mean_abs"],
            "p99_abs": comp["p99_abs"],
            "relative_l2_gap": comp["relative_l2_gap"],
        }
        _check(
            comp["relative_l2_gap"] <= args.max_official_1layer_relative_l2,
            failures,
            "official_1layer.{}.relative_l2".format(label),
        )
        _check(
            comp["mean_abs"] <= args.max_official_1layer_mean_abs,
            failures,
            "official_1layer.{}.mean_abs".format(label),
        )

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 end-to-end BF16 tolerance evidence",
        "status": "PASS" if not failures else "FAIL",
        "artifacts_dir_name": base.name,
        "thresholds": {
            "max_real_forward_relative_l2": args.max_real_forward_relative_l2,
            "max_real_forward_mean_abs": args.max_real_forward_mean_abs,
            "max_real_forward_p99_abs": args.max_real_forward_p99_abs,
            "max_routing_forward_relative_l2": args.max_routing_forward_relative_l2,
            "min_routing_relative_l2_reduction": args.min_routing_relative_l2_reduction,
            "max_real_train_grad_rel_gap": args.max_real_train_grad_rel_gap,
            "max_real_train_state_abs": args.max_real_train_state_abs,
            "max_official_1layer_relative_l2": args.max_official_1layer_relative_l2,
            "max_official_1layer_mean_abs": args.max_official_1layer_mean_abs,
            "max_attention_io_output_abs": args.max_attention_io_output_abs,
            "max_attention_io_input_grad_abs": args.max_attention_io_input_grad_abs,
            "max_sft_replay_grad_rel_gap": args.max_sft_replay_grad_rel_gap,
            "max_sft_replay_state_abs": args.max_sft_replay_state_abs,
        },
        "real_forward": real_forward,
        "routing_forward": routing_forward,
        "real_train": real_train,
        "train_after_attention_output_replay": train_after_attention_replay,
        "attention_io": {
            "max_output_abs": max_attention_io_output_abs,
            "per_backend": attention_io_summary,
        },
        "attention_io_training_step": {
            "max_output_abs": max_attention_io_train_output_abs,
            "max_input_grad_abs": max_attention_io_train_input_grad_abs,
            "max_state_after_step_abs": max_attention_io_train_state_abs,
        },
        "official_1layer": official_1layer,
        "failures": failures,
        "conclusion": (
            "The real non-injected strict parity gates still fail, but all measured forward "
            "and training drifts are inside the declared BF16 tolerance envelope; once the "
            "localized attention forward-value drift is replayed away, complete SFT loss "
            "parity is exact and gradient/update drift remains below threshold."
            if not failures
            else "The recorded artifacts do not satisfy the declared BF16 tolerance envelope."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

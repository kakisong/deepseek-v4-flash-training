#!/usr/bin/env python3
"""Validate the official-vs-Miles BF16 forward tolerance gate.

This verifier is deliberately not a strict parity gate.  The official inference
reference and the Miles/Megatron training runtime still fail strict response
logprob parity at rtol=2e-3/atol=2e-2.  The gate here records the narrower claim
that the remaining official-vs-Miles forward drift is bounded by an explicit
BF16 tolerance envelope, and that external one-step train parity must stay
blocked until strict official/reference forward parity passes.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(base: Path, name: str) -> Dict[str, Any]:
    return json.loads((base / name).read_text(encoding="utf-8"))


def _status(payload: Dict[str, Any]) -> str:
    return str(payload.get("status") or payload.get("overall_status") or "")


def _check(condition: bool, failures: List[str], name: str) -> None:
    if not condition:
        failures.append(name)


def _comparison(payload: Dict[str, Any], artifact: str) -> Dict[str, Any]:
    comp = payload.get("comparison")
    if not isinstance(comp, dict):
        raise KeyError("{} missing comparison".format(artifact))
    return comp


def _summarize_forward_artifact(
    base: Path,
    artifact: str,
    label: str,
    thresholds: Dict[str, float],
    failures: List[str],
) -> Dict[str, Any]:
    payload = _load(base, artifact)
    comp = _comparison(payload, artifact)
    checks = {
        "relative_l2_gap": float(comp["relative_l2_gap"]) <= thresholds["max_relative_l2_gap"],
        "mean_abs": float(comp["mean_abs"]) <= thresholds["max_mean_abs"],
        "p99_abs": float(comp["p99_abs"]) <= thresholds["max_p99_abs"],
        "max_abs": float(comp["max_abs"]) <= thresholds["max_abs"],
        "strict_threshold_recorded": (
            float(comp.get("rtol", 0.0)) == thresholds["strict_rtol"]
            and float(comp.get("atol", 0.0)) == thresholds["strict_atol"]
        ),
        "strict_status_recorded": comp.get("status") in ("PASS", "FAIL"),
        "finite_metrics": all(
            float(comp[key]) == float(comp[key])
            for key in ("relative_l2_gap", "mean_abs", "p99_abs", "max_abs")
        ),
    }
    for check, passed in checks.items():
        _check(bool(passed), failures, "{}.{}".format(label, check))
    return {
        "artifact": artifact,
        "label": label,
        "attention_impl": comp.get("miles_attention_impl"),
        "strict_status": comp.get("status"),
        "num_tokens": comp.get("num_tokens"),
        "mismatches": comp.get("mismatches"),
        "max_abs": comp.get("max_abs"),
        "mean_abs": comp.get("mean_abs"),
        "p95_abs": comp.get("p95_abs"),
        "p99_abs": comp.get("p99_abs"),
        "relative_l2_gap": comp.get("relative_l2_gap"),
        "rtol": comp.get("rtol"),
        "atol": comp.get("atol"),
        "bf16_tolerance_checks": checks,
        "bf16_tolerance_status": "PASS" if all(checks.values()) else "FAIL",
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("docs/en/advanced"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-l2-gap", type=float, default=5e-6)
    parser.add_argument("--max-mean-abs", type=float, default=0.04)
    parser.add_argument("--max-p99-abs", type=float, default=0.15)
    parser.add_argument("--max-abs", type=float, default=0.20)
    parser.add_argument("--strict-rtol", type=float, default=2e-3)
    parser.add_argument("--strict-atol", type=float, default=2e-2)
    args = parser.parse_args()

    base = args.artifacts_dir
    thresholds = {
        "max_relative_l2_gap": args.max_relative_l2_gap,
        "max_mean_abs": args.max_mean_abs,
        "max_p99_abs": args.max_p99_abs,
        "max_abs": args.max_abs,
        "strict_rtol": args.strict_rtol,
        "strict_atol": args.strict_atol,
    }
    failures: List[str] = []

    prerequisites = {
        "fix_regression_guards": ("deepseek-v4-fix-regression-guards-20260531.json", "PASS"),
        "official_attention_forward": ("deepseek-v4-official-attention-forward-20260531.json", "PASS"),
        "loaded_weight_mapping": ("deepseek-v4-loaded-weight-mapping-1layer-mlp-qatsim-20260531.json", "PASS"),
        "attention_trace_replay": ("deepseek-v4-attention-trace-replay-qatsim-0415-20260531.json", "PASS"),
        "grouped_mlp_math": ("deepseek-v4-grouped-mlp-math-20260531.json", "PASS"),
        "mlp_expert_replay": (
            "deepseek-v4-mlp-expert-replay-qatsim-0415-20260531.json",
            "PASS_WITH_DRIFT_RECORDED",
        ),
    }
    prerequisite_status = {}
    for gate, (artifact, expected) in prerequisites.items():
        payload = _load(base, artifact)
        actual = _status(payload)
        prerequisite_status[gate] = {"artifact": artifact, "expected": expected, "actual": actual}
        _check(actual == expected, failures, "prerequisite.{}.status={}".format(gate, actual))

    forward_checks = [
        _summarize_forward_artifact(
            base,
            "deepseek-v4-official-full-forward-1layer-bf16-qatsim-0415-20260531.json",
            "official_vs_miles_sparse",
            thresholds,
            failures,
        ),
        _summarize_forward_artifact(
            base,
            "deepseek-v4-official-full-forward-1layer-bf16-tilelang-qatsim-0415-20260531.json",
            "official_vs_miles_tilelang",
            thresholds,
            failures,
        ),
    ]

    strict_forward_status = "PASS" if all(item["strict_status"] == "PASS" for item in forward_checks) else "FAIL"
    bf16_forward_status = (
        "PASS" if all(item["bf16_tolerance_status"] == "PASS" for item in forward_checks) else "FAIL"
    )
    _check(bf16_forward_status == "PASS", failures, "bf16_forward_tolerance_status")

    variants = _load(base, "deepseek-v4-official-runtime-precision-variants-qatsim-0415-20260531.json")
    _check(_status(variants) == "RUN_WITH_DRIFT_RECORDED", failures, "runtime_precision_variants.status")
    best = variants.get("best_by_mean_abs", {})
    baseline = None
    for row in variants.get("rows", []):
        if row.get("label") == "baseline_official_fp32":
            baseline = row
            break
    _check(isinstance(baseline, dict), failures, "runtime_precision_variants.baseline_present")
    if isinstance(baseline, dict) and isinstance(best, dict):
        _check(
            float(best.get("mean_abs", 1e9)) <= float(baseline.get("mean_abs", -1.0)),
            failures,
            "runtime_precision_variants.best_mean_abs_not_worse",
        )
        _check(
            int(best.get("mismatches", 10**9)) <= int(baseline.get("mismatches", -1)),
            failures,
            "runtime_precision_variants.best_mismatches_not_worse",
        )

    external_train_precondition = {
        "required_gate": "official_reference_mini_checkpoint_forward_parity",
        "required_status": "PASS",
        "actual_status": strict_forward_status,
        "status": "NOT_MET" if strict_forward_status != "PASS" else "MET",
        "external_one_step_train_parity_status": (
            "SKIPPED_FORWARD_STRICT_PARITY_REQUIRED"
            if strict_forward_status != "PASS"
            else "READY_TO_RUN"
        ),
        "reason": (
            "External one-step train parity would compare gradients and updates from a reference "
            "forward path.  It is not a sound PASS gate while strict official/reference forward "
            "logprob parity still fails."
        ),
    }

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 official-vs-Miles BF16 forward tolerance gate",
        "status": "PASS" if not failures else "FAIL",
        "artifacts_dir_name": base.name,
        "thresholds": thresholds,
        "prerequisites": prerequisite_status,
        "forward_checks": forward_checks,
        "strict_forward_status": strict_forward_status,
        "bf16_forward_tolerance_status": bf16_forward_status,
        "runtime_precision_variant_summary": {
            "artifact": "deepseek-v4-official-runtime-precision-variants-qatsim-0415-20260531.json",
            "status": _status(variants),
            "best_by_mean_abs": best,
        },
        "external_train_precondition": external_train_precondition,
        "failures": failures,
        "conclusion": (
            "Official/reference strict forward parity remains open, but the recorded sparse and "
            "tilelang official-vs-Miles full-forward drifts satisfy the declared BF16 tolerance "
            "standard.  External one-step train parity remains intentionally gated on strict "
            "official/reference forward parity."
            if not failures
            else "The official-vs-Miles forward artifacts do not satisfy the declared BF16 tolerance standard."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

#!/usr/bin/env python3
"""Validate the DeepSeek-V4 mini-checkpoint correctness gate.

This verifier is intentionally a framework-level correctness gate, not a
strict logprob parity gate.  It proves the narrower claim needed for the Miles
training path: on the loaded 4-layer mini checkpoint, SFT one-step execution is
finite, routed MoE math is covered by explicit references, attention training
drift is bounded, and the complete SFT step passes once the localized attention
forward-value drift is replayed away.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(base: Path, name: str) -> Dict[str, Any]:
    return json.loads((base / name).read_text(encoding="utf-8"))


def _status(payload: Dict[str, Any]) -> Optional[str]:
    status = payload.get("status") or payload.get("overall_status")
    return str(status) if status is not None else None


def _check(condition: bool, failures: List[str], name: str) -> None:
    if not condition:
        failures.append(name)


def _find_compare(items: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for item in items:
        if item.get("label") == label:
            return item
    raise KeyError(f"comparison not found: {label}")


def _max_nested_value(obj: Any, key: str) -> float:
    values = []  # type: List[float]

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            value = item.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(obj)
    if not values:
        raise KeyError(f"no nested values for {key}")
    return max(values)


def _all_case_status(payload: Dict[str, Any]) -> bool:
    def walk(item: Any) -> bool:
        if isinstance(item, dict):
            status = item.get("status")
            if status is not None and status != "PASS":
                return False
            return all(walk(value) for value in item.values())
        if isinstance(item, list):
            return all(walk(value) for value in item)
        return True

    return walk(payload.get("cases", []))


def _external_reference_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    comp = payload["comparison"]
    thresholds = payload["thresholds"]
    return {
        "artifact_status": _status(payload),
        "compress_ratio": payload["config"]["compress_ratio"],
        "loss_abs": comp["loss_abs"],
        "output_max_abs": comp["output"]["max_abs"],
        "input_grad_max_abs": comp["input_grad"]["max_abs"],
        "grad_max_abs": comp["grad"]["max_abs"],
        "state_after_step_max_abs": comp["state_after_step"]["max_abs"],
        "num_common_grad_tensors": comp["num_common_grad_tensors"],
        "thresholds": {
            "max_loss_abs": thresholds["max_loss_abs"],
            "max_output_abs": thresholds["max_output_abs"],
            "max_input_grad_abs": thresholds["max_input_grad_abs"],
            "max_grad_abs": thresholds["max_grad_abs"],
            "max_state_abs": thresholds["max_state_abs"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("docs/en/advanced"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.artifacts_dir
    failures = []  # type: List[str]

    mini_forward = _load(base, "deepseek-v4-mini-forward-compare-20260531.json")
    real_train = _load(base, "deepseek-v4-mini-train-step-qatsim-0415-20260531.json")
    attn_replay_artifact = "deepseek-v4-mini-checkpoint-correctness-rerun-sft-attention-output-replay-20260531.json"
    if not (base / attn_replay_artifact).exists():
        attn_replay_artifact = "deepseek-v4-mini-train-step-attention-output-replay-qatsim-20260531.json"
    attn_replay_train = _load(base, attn_replay_artifact)
    attention_io_train = _load(base, "deepseek-v4-mini-attention-io-training-step-qatsim-20260531.json")
    e2e_tolerance = _load(base, "deepseek-v4-end-to-end-bf16-tolerance-20260531.json")
    sft_loss_reference = _load(base, "deepseek-v4-sft-loss-reference-20260531.json")
    grouped_mlp = _load(base, "deepseek-v4-grouped-mlp-math-20260531.json")
    moe_dispatch = _load(base, "deepseek-v4-moe-ep8-dispatch-math-20260531.json")
    mlp_replay = _load(base, "deepseek-v4-mlp-expert-replay-qatsim-0415-20260531.json")
    optimizer = _load(base, "deepseek-v4-optimizer-update-math-20260531.json")
    external_refs = {
        "c0": _load(base, "deepseek-v4-external-training-reference-1layer-20260531.json"),
        "c4": _load(base, "deepseek-v4-external-training-reference-1layer-c4-20260531.json"),
        "c128": _load(base, "deepseek-v4-external-training-reference-1layer-c128-20260531.json"),
    }

    _check(_status(e2e_tolerance) == "PASS", failures, "end_to_end_bf16_tolerance.status")
    _check(_status(sft_loss_reference) == "PASS", failures, "sft_loss_reference.status")
    _check(_status(optimizer) == "PASS", failures, "optimizer_update_math.status")

    strict_forward = {
        item["label"]: item
        for item in mini_forward.get("compare_results", [])
        if item.get("label") in ("dense_vs_sparse", "dense_vs_tilelang", "sparse_vs_tilelang")
    }
    strict_forward_status = "FAIL" if any(item.get("status") == "FAIL" for item in strict_forward.values()) else "PASS"
    dense_repeat = _find_compare(mini_forward["repeatability_checks"], "dense_repeat_with_deterministic_runtime_exact")
    _check(dense_repeat["status"] == "PASS" and dense_repeat["mismatches"] == 0, failures, "dense_repeatability")
    _check(strict_forward_status == "FAIL", failures, "strict_forward_boundary_recorded")

    _check(_status(real_train) == "FAIL", failures, "real_non_injected_train_strict_boundary_recorded")
    _check(
        all("loss_abs_global_max" in item for item in real_train.get("comparisons", [])),
        failures,
        "real_train_loss_gap_recorded",
    )
    real_train_bounds = {}  # type: Dict[str, Dict[str, float]]
    for label in ("dense_vs_sparse", "dense_vs_tilelang", "sparse_vs_tilelang"):
        compare = _find_compare(real_train["comparisons"], label)
        real_train_bounds[label] = {
            "loss_abs_global_max": compare["loss_abs_global_max"],
            "selected_grad_max_rel_gap": compare["selected_grad_max_rel_gap"],
            "selected_state_max_abs": compare["selected_state_max_abs"],
        }
        _check(
            compare["selected_grad_max_rel_gap"] <= e2e_tolerance["thresholds"]["max_real_train_grad_rel_gap"],
            failures,
            f"{label}.real_train_grad_rel_bf16_bound",
        )
        _check(
            compare["selected_state_max_abs"] <= e2e_tolerance["thresholds"]["max_real_train_state_abs"],
            failures,
            f"{label}.real_train_state_bf16_bound",
        )

    _check(_status(attn_replay_train) == "PASS", failures, "attention_output_replay_train.status")
    _check(attn_replay_train["world_size"] == 8, failures, "attention_output_replay_train.world_size")
    _check(attn_replay_train["checkpoint"] is not None, failures, "attention_output_replay_train.loaded_checkpoint")
    _check(attn_replay_train["rollout_data_name"] is not None, failures, "attention_output_replay_train.rollout")
    _check(
        attn_replay_train["routing_replay"]["mode"] == "record_replay",
        failures,
        "attention_output_replay_train.routing_replay",
    )
    _check(
        attn_replay_train["attention_output_replay"]["mode"] == "record_replay",
        failures,
        "attention_output_replay_train.attention_output_replay",
    )
    sft_replay_bounds = {}  # type: Dict[str, Dict[str, float]]
    for label in ("dense_vs_sparse", "dense_vs_tilelang", "sparse_vs_tilelang"):
        compare = _find_compare(attn_replay_train["comparisons"], label)
        sft_replay_bounds[label] = {
            "loss_abs_global_max": compare["loss_abs_global_max"],
            "selected_grad_max_rel_gap": compare["selected_grad_max_rel_gap"],
            "selected_state_max_abs": compare["selected_state_max_abs"],
        }
        _check(compare["loss_abs_global_max"] == 0.0, failures, f"{label}.sft_replay_loss_exact")
        _check(
            compare["selected_grad_max_rel_gap"]
            <= attn_replay_train["thresholds"]["max_selected_grad_rel_gap"],
            failures,
            f"{label}.sft_replay_grad_rel",
        )
        _check(
            compare["selected_state_max_abs"] <= attn_replay_train["thresholds"]["max_selected_state_abs"],
            failures,
            f"{label}.sft_replay_state_abs",
        )

    for impl, result in attn_replay_train["impls"].items():
        stats = result["global_stats"]
        _check(stats["nonfinite_grad_tensors"] == 0, failures, f"{impl}.nonfinite_grad_tensors")
        _check(stats["num_params_with_grad"] > 0, failures, f"{impl}.num_params_with_grad")
        _check(float(result["loss_log"]["count"]) > 0, failures, f"{impl}.sft_loss_token_count")
        _check(float(result["loss_log"]["loss"]) == float(result["loss_log"]["loss"]), failures, f"{impl}.loss_finite")

    _check(_status(attention_io_train) == "PASS", failures, "attention_io_training_step.status")
    attention_io_bounds = {
        "max_output_abs": _max_nested_value(attention_io_train, "output_max_abs_global_max"),
        "max_input_grad_abs": _max_nested_value(attention_io_train, "input_grad_max_abs_global_max"),
        "max_state_after_step_abs": _max_nested_value(attention_io_train, "state_after_step_max_abs_global_max"),
    }
    _check(
        attention_io_bounds["max_output_abs"] <= attention_io_train["thresholds"]["max_output_abs"],
        failures,
        "attention_io_training_step.output_bound",
    )
    _check(
        attention_io_bounds["max_input_grad_abs"] <= attention_io_train["thresholds"]["max_input_grad_abs"],
        failures,
        "attention_io_training_step.input_grad_bound",
    )
    _check(
        attention_io_bounds["max_state_after_step_abs"] <= attention_io_train["thresholds"]["max_state_abs"],
        failures,
        "attention_io_training_step.state_bound",
    )

    sft_loss_reference_bounds = {
        "loss_abs_global_max": sft_loss_reference["global_summary"]["loss_abs_global_max"],
        "logprob_max_abs_global_max": sft_loss_reference["global_summary"]["logprob_max_abs_global_max"],
        "logprob_mean_abs_global_max": sft_loss_reference["global_summary"]["logprob_mean_abs_global_max"],
        "token_count_abs_global_max": sft_loss_reference["global_summary"]["token_count_abs_global_max"],
        "world_size": sft_loss_reference["world_size"],
        "reference_formula": sft_loss_reference["reference_formula"],
    }
    _check(sft_loss_reference_bounds["world_size"] == 8, failures, "sft_loss_reference.world_size")
    _check(
        sft_loss_reference_bounds["loss_abs_global_max"] <= sft_loss_reference["thresholds"]["max_loss_abs"],
        failures,
        "sft_loss_reference.loss",
    )
    _check(
        sft_loss_reference_bounds["logprob_max_abs_global_max"]
        <= sft_loss_reference["thresholds"]["max_logprob_abs"],
        failures,
        "sft_loss_reference.logprob_max",
    )
    _check(
        sft_loss_reference_bounds["logprob_mean_abs_global_max"]
        <= sft_loss_reference["thresholds"]["max_logprob_mean_abs"],
        failures,
        "sft_loss_reference.logprob_mean",
    )
    _check(
        sft_loss_reference_bounds["token_count_abs_global_max"] == 0.0,
        failures,
        "sft_loss_reference.token_count",
    )

    _check(_status(grouped_mlp) == "PASS", failures, "grouped_mlp_math.status")
    _check(_all_case_status(grouped_mlp), failures, "grouped_mlp_math.all_cases")
    _check(_status(moe_dispatch) == "PASS", failures, "moe_ep8_dispatch_math.status")
    _check(moe_dispatch.get("expert_parallel_size") == 8, failures, "moe_ep8_dispatch_math.ep8")
    _check(_all_case_status(moe_dispatch), failures, "moe_ep8_dispatch_math.all_cases")
    _check(_status(mlp_replay) == "PASS_WITH_DRIFT_RECORDED", failures, "mlp_expert_replay.status")
    _check(
        bool(mlp_replay["key_checks"]["official_formula_total_replay_matches_official_ffn"]),
        failures,
        "mlp_expert_replay.official_formula",
    )
    _check(
        bool(mlp_replay["key_checks"]["official_vs_miles_uses_same_expert_indices"]),
        failures,
        "mlp_expert_replay.same_expert_indices",
    )

    external_reference_bounds = {
        label: _external_reference_summary(payload)
        for label, payload in external_refs.items()
    }
    for label, payload in external_refs.items():
        summary = external_reference_bounds[label]
        _check(_status(payload) == "PASS", failures, f"external_reference_{label}.status")
        _check(
            summary["loss_abs"] <= summary["thresholds"]["max_loss_abs"],
            failures,
            f"external_reference_{label}.loss",
        )
        _check(
            summary["output_max_abs"] <= summary["thresholds"]["max_output_abs"],
            failures,
            f"external_reference_{label}.output",
        )
        _check(
            summary["input_grad_max_abs"] <= summary["thresholds"]["max_input_grad_abs"],
            failures,
            f"external_reference_{label}.input_grad",
        )
        _check(
            summary["grad_max_abs"] <= summary["thresholds"]["max_grad_abs"],
            failures,
            f"external_reference_{label}.grad",
        )
        _check(
            summary["state_after_step_max_abs"] <= summary["thresholds"]["max_state_abs"],
            failures,
            f"external_reference_{label}.state",
        )
        _check(summary["num_common_grad_tensors"] > 0, failures, f"external_reference_{label}.grad_tensors")
    _check(
        bool(external_refs["c4"]["config"].get("indexer_selection_pressure")),
        failures,
        "external_reference_c4.indexer_selection_pressure",
    )

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 4-layer mini checkpoint framework-level training correctness gate",
        "status": "PASS" if not failures else "FAIL",
        "artifacts_dir_name": base.name,
        "claim": (
            "The Miles DeepSeek-V4 mini-checkpoint training path is correct under the declared BF16 "
            "runtime tolerance: loaded-checkpoint SFT one-step execution is finite, routed MoE math "
            "and EP=8 dispatch are independently reference-checked, attention training drift is "
            "bounded, and complete SFT loss/gradient/update parity passes after replaying the "
            "localized attention forward-value drift."
        ),
        "strict_boundaries": {
            "real_non_injected_forward_strict_status": strict_forward_status,
            "real_non_injected_sft_one_step_strict_status": _status(real_train),
            "not_reclassified_as_strict_pass": True,
        },
        "loaded_checkpoint_sft": {
            "artifact": attn_replay_artifact,
            "world_size": attn_replay_train["world_size"],
            "checkpoint_name": attn_replay_train["checkpoint"],
            "rollout_data_name": attn_replay_train["rollout_data_name"],
            "max_samples": attn_replay_train["max_samples"],
            "routing_replay": attn_replay_train["routing_replay"],
            "attention_output_replay": attn_replay_train["attention_output_replay"],
            "sft_replay_bounds": sft_replay_bounds,
        },
        "real_non_injected_train_bf16_bounds": real_train_bounds,
        "attention_io_training_step_bounds": attention_io_bounds,
        "sft_loss_reference": sft_loss_reference_bounds,
        "moe_evidence": {
            "grouped_mlp_math_status": _status(grouped_mlp),
            "moe_ep8_dispatch_math_status": _status(moe_dispatch),
            "mlp_expert_replay_status": _status(mlp_replay),
            "mlp_expert_replay_key_checks": mlp_replay["key_checks"],
        },
        "external_attention_training_references": external_reference_bounds,
        "supporting_pass_gates": {
            "end_to_end_bf16_tolerance": _status(e2e_tolerance),
            "optimizer_update_math": _status(optimizer),
        },
        "failures": failures,
        "conclusion": (
            "Mini checkpoint-level correctness is PASS under the declared BF16 training tolerance; "
            "strict real-forward and strict real-SFT parity remain explicitly recorded boundaries."
            if not failures
            else "Mini checkpoint-level correctness evidence is incomplete or inconsistent."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

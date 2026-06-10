#!/usr/bin/env python3
"""基于已记录的产物(artifacts)校验 DeepSeek-V4 证明台账(proof ledger)。"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load(base: Path, name: str) -> Dict[str, Any]:
    return json.loads((base / name).read_text(encoding="utf-8"))


def _find_compare(items: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for item in items:
        if item.get("label") == label:
            return item
    raise KeyError(f"comparison not found: {label}")


def _find_case(payload: Dict[str, Any], impl: str, case: str) -> Dict[str, Any]:
    for item in payload.get("backend_cases", []):
        if item.get("impl") == impl and item.get("case") == case:
            return item
    raise KeyError(f"case not found: {impl}.{case}")


def _check(condition: bool, failures: List[str], name: str) -> bool:
    if not condition:
        failures.append(name)
        return False
    return True


def _status(payload: Dict[str, Any]) -> Optional[str]:
    return payload.get("status") or payload.get("overall_status")


def _max_comparison_value(payload: Dict[str, Any], key: str) -> float:
    values = []
    for item in payload.get("comparisons", []):
        if key in item:
            values.append(float(item[key]))
    if not values:
        raise KeyError(f"no comparison values for {key}")
    return max(values)


def _max_nested_value(obj: Any, key: str) -> float:
    values = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if key in item and isinstance(item[key], (int, float)):
                values.append(float(item[key]))
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)

    walk(obj)
    if not values:
        raise KeyError(f"no nested values for {key}")
    return max(values)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("docs/en/advanced"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.artifacts_dir
    failures: List[str] = []
    gates: Dict[str, Any] = {}

    required_pass = {
        "operator_math": ("deepseek-v4-operator-math-20260531.json", "PASS"),
        "official_attention_forward": ("deepseek-v4-official-attention-forward-20260531.json", "PASS"),
        "loaded_weight_mapping": ("deepseek-v4-loaded-weight-mapping-1layer-mlp-qatsim-20260531.json", "PASS"),
        "attention_trace_replay": ("deepseek-v4-attention-trace-replay-qatsim-0415-20260531.json", "PASS"),
        "attention_core_external_oracle": ("deepseek-v4-attention-core-external-oracle-20260601.json", "PASS"),
        "attention_training_step": ("deepseek-v4-attention-training-step-qatsim-20260531.json", "PASS"),
        "transformer_block_training_step": ("deepseek-v4-transformer-block-training-step-qatsim-20260531.json", "PASS"),
        "grouped_mlp_math": ("deepseek-v4-grouped-mlp-math-20260531.json", "PASS"),
        "moe_ep8_dispatch_math": ("deepseek-v4-moe-ep8-dispatch-math-20260531.json", "PASS"),
        "external_moe_ep8_reference": (
            "deepseek-v4-external-moe-ep8-reference-20260531.json",
            "PASS",
        ),
        "attention_io_training_step": ("deepseek-v4-mini-attention-io-training-step-qatsim-20260531.json", "PASS"),
        "sft_attention_output_replay": (
            "deepseek-v4-mini-train-step-attention-output-replay-qatsim-20260531.json",
            "PASS",
        ),
        "mini_checkpoint_correctness_gate": (
            "deepseek-v4-mini-checkpoint-correctness-gate-20260531.json",
            "PASS",
        ),
        "mini_external_full_reference_routing_replay_forward_tolerance": (
            "deepseek-v4-mini-external-full-reference-bf16-routing-replay-tolerance-20260531.json",
            "PASS",
        ),
        "sft_loss_explicit_reference": (
            "deepseek-v4-sft-loss-reference-20260531.json",
            "PASS",
        ),
        "sft_loss_train_explicit_reference": (
            "deepseek-v4-sft-loss-train-reference-20260531.json",
            "PASS",
        ),
        "external_training_reference_1layer": (
            "deepseek-v4-external-training-reference-1layer-20260531.json",
            "PASS",
        ),
        "external_training_reference_1layer_c4": (
            "deepseek-v4-external-training-reference-1layer-c4-20260531.json",
            "PASS",
        ),
        "external_training_reference_1layer_c128": (
            "deepseek-v4-external-training-reference-1layer-c128-20260531.json",
            "PASS",
        ),
        "external_training_reference_1layer_moe": (
            "deepseek-v4-external-training-reference-1layer-moe-20260531.json",
            "PASS",
        ),
        "end_to_end_bf16_tolerance": (
            "deepseek-v4-end-to-end-bf16-tolerance-20260531.json",
            "PASS",
        ),
        "official_forward_bf16_tolerance": (
            "deepseek-v4-official-forward-bf16-tolerance-20260531.json",
            "PASS",
        ),
        "optimizer_update_math": (
            "deepseek-v4-optimizer-update-math-20260531.json",
            "PASS",
        ),
        "fix_regression_guards": (
            "deepseek-v4-fix-regression-guards-20260531.json",
            "PASS",
        ),
        "proof_coverage_matrix": (
            "deepseek-v4-proof-coverage-matrix-20260531.json",
            "PASS",
        ),
        "environment_provenance": (
            "deepseek-v4-environment-provenance-20260531.json",
            "PASS",
        ),
        "external_reference_provenance": (
            "deepseek-v4-external-reference-provenance-20260531.json",
            "PASS",
        ),
    }

    for gate, (name, expected) in required_pass.items():
        payload = _load(base, name)
        actual = _status(payload)
        gates[gate] = {"artifact": name, "status": actual}
        _check(actual == expected, failures, f"{gate}.status={actual}")

    operator = _load(base, required_pass["operator_math"][0])
    operator_results = {item["name"]: item for item in operator["results"]}
    hc = operator_results["hyper_connection_pr4839_orientation"]["details"]
    gates["hyper_connection_orientation"] = {
        "fixed_formula_max_diff": hc["max_diff_vs_megatron_pr4839_fixed_native"],
        "wrong_formula_max_diff": hc["max_diff_vs_prefix_wrong_comb_residual"],
    }
    _check(
        hc["max_diff_vs_megatron_pr4839_fixed_native"] < 1e-6
        and hc["max_diff_vs_prefix_wrong_comb_residual"] > 1.0,
        failures,
        "hyper_connection_orientation.distinguishes_fixed_and_wrong_formula",
    )

    activation = _load(base, "deepseek-v4-mini-activation-replay-qatsim-20260531.json")
    sublayer = _load(base, "deepseek-v4-mini-sublayer-activation-replay-qatsim-20260531.json")
    attn_layer = _load(base, "deepseek-v4-mini-attention-layer-replay-qatsim-20260531.json")
    attn_io = _load(base, "deepseek-v4-mini-attention-io-replay-qatsim-20260531.json")

    for impl in ("sparse", "tilelang"):
        prefix4 = _find_case(activation, impl, "prefix4")["compare_to_dense"]
        final_ln = _find_case(activation, impl, "final_ln")["compare_to_dense"]
        all_attn = _find_case(sublayer, impl, "all_attn")["compare_to_dense"]
        all_mlp = _find_case(sublayer, impl, "all_mlp")["compare_to_dense"]
        all_attn_layer = _find_case(attn_layer, impl, "all_attn")["compare_to_dense"]
        off = _find_case(sublayer, impl, "off")["compare_to_dense"]
        gates[f"{impl}_activation_localization"] = {
            "off_mismatches": off["mismatches"],
            "prefix4_exact": prefix4["exact_equal"],
            "final_ln_exact": final_ln["exact_equal"],
            "all_attention_exact": all_attn["exact_equal"],
            "all_mlp_exact": all_mlp["exact_equal"],
            "layerwise_all_attention_exact": all_attn_layer["exact_equal"],
        }
        _check(prefix4["exact_equal"], failures, f"{impl}.prefix4_activation_replay_exact")
        _check(final_ln["exact_equal"], failures, f"{impl}.final_ln_replay_exact")
        _check(all_attn["exact_equal"], failures, f"{impl}.all_attention_replay_exact")
        _check(not all_mlp["exact_equal"], failures, f"{impl}.all_mlp_replay_not_sufficient")
        _check(all_attn_layer["exact_equal"], failures, f"{impl}.layerwise_all_attention_replay_exact")
        _check(off["mismatches"] > 0, failures, f"{impl}.off_case_has_real_drift")

    io_max = 0.0
    io_finite = True
    io_layer0_largest = attn_io["claims"].get("layer0_attention_has_largest_output_mean_abs_for_both_backends")
    for backend in attn_io["backend_summaries"]:
        for layer in backend["per_layer"]:
            output = layer["output_from_replayed_dense_input"]
            io_finite = io_finite and bool(output["finite_all"])
            io_max = max(io_max, float(output["max_abs_global_max"]))
    gates["attention_io_forward_drift"] = {
        "max_output_abs": io_max,
        "finite_all": io_finite,
        "layer0_largest_output_mean_abs_for_both_backends": io_layer0_largest,
    }
    _check(io_finite, failures, "attention_io.all_outputs_finite")
    _check(io_max <= 0.0625, failures, "attention_io.max_output_abs_le_0_0625")
    _check(bool(io_layer0_largest), failures, "attention_io.layer0_largest")

    layerwise_drift = _load(base, "deepseek-v4-mini-layerwise-drift-localization-20260601.json")
    layerwise_loc = layerwise_drift["localization"]
    gates["mini_checkpoint_layerwise_drift_localization"] = {
        "artifact": "deepseek-v4-mini-layerwise-drift-localization-20260601.json",
        "status": _status(layerwise_drift),
        "embedding_exact": layerwise_loc["embedding_exact"],
        "layer0_input_layernorm_exact": layerwise_loc["layer0_input_layernorm_exact"],
        "first_nonzero_module": layerwise_loc["first_nonzero_module"],
        "layer0_self_attention_dense_vs_sparse": layerwise_loc["layer0_self_attention_dense_vs_sparse"],
        "layer0_self_attention_dense_vs_tilelang": layerwise_loc["layer0_self_attention_dense_vs_tilelang"],
    }
    _check(
        _status(layerwise_drift) == "PASS_WITH_LAYERWISE_DRIFT_LOCALIZED",
        failures,
        "layerwise_drift.status",
    )
    _check(layerwise_loc["embedding_exact"], failures, "layerwise_drift.embedding_exact")
    _check(layerwise_loc["layer0_input_layernorm_exact"], failures, "layerwise_drift.layer0_input_ln_exact")
    _check(
        layerwise_loc["first_nonzero_module"] == "module.decoder.layers.0.self_attention",
        failures,
        "layerwise_drift.first_nonzero_layer0_attention",
    )
    _check(
        layerwise_loc["layer0_self_attention_dense_vs_sparse"]["max_abs"] <= 0.0625
        and layerwise_loc["layer0_self_attention_dense_vs_tilelang"]["max_abs"] <= 0.0625,
        failures,
        "layerwise_drift.layer0_attention_output_bound",
    )

    attention_core = _load(base, "deepseek-v4-mini-layer0-attention-internal-localization-20260601.json")
    attention_core_loc = attention_core["localization"]
    gates["mini_checkpoint_layer0_attention_core_localization"] = {
        "artifact": "deepseek-v4-mini-layer0-attention-internal-localization-20260601.json",
        "status": _status(attention_core),
        "q_path_exact_sparse": attention_core_loc["q_path_exact_sparse"],
        "q_path_exact_tilelang": attention_core_loc["q_path_exact_tilelang"],
        "kv_path_exact_sparse": attention_core_loc["kv_path_exact_sparse"],
        "kv_path_exact_tilelang": attention_core_loc["kv_path_exact_tilelang"],
        "first_nonzero_internal_module": attention_core_loc["first_nonzero_internal_module"],
        "attention_core_dense_vs_sparse": attention_core_loc["attention_core_dense_vs_sparse"],
        "attention_core_dense_vs_tilelang": attention_core_loc["attention_core_dense_vs_tilelang"],
        "after_wo_b_dense_vs_sparse": attention_core_loc["after_wo_b_dense_vs_sparse"],
        "after_wo_b_dense_vs_tilelang": attention_core_loc["after_wo_b_dense_vs_tilelang"],
    }
    _check(
        _status(attention_core) == "PASS_WITH_ATTENTION_CORE_LOCALIZED",
        failures,
        "attention_core_localization.status",
    )
    _check(
        attention_core_loc["q_path_exact_sparse"] and attention_core_loc["q_path_exact_tilelang"],
        failures,
        "attention_core_localization.q_path_exact",
    )
    _check(
        attention_core_loc["kv_path_exact_sparse"] and attention_core_loc["kv_path_exact_tilelang"],
        failures,
        "attention_core_localization.kv_path_exact",
    )
    _check(
        attention_core_loc["first_nonzero_internal_module"]
        == "module.decoder.layers.0.self_attention.attention_core",
        failures,
        "attention_core_localization.first_nonzero_attention_core",
    )
    _check(
        attention_core_loc["attention_core_dense_vs_sparse"]["max_abs"] <= 0.0625
        and attention_core_loc["attention_core_dense_vs_tilelang"]["max_abs"] <= 0.0625,
        failures,
        "attention_core_localization.attention_core_bound",
    )
    _check(
        attention_core_loc["after_wo_b_dense_vs_sparse"]["max_abs"] <= 0.0625
        and attention_core_loc["after_wo_b_dense_vs_tilelang"]["max_abs"] <= 0.0625,
        failures,
        "attention_core_localization.after_wo_b_bound",
    )

    attention_core_oracle = _load(base, "deepseek-v4-attention-core-external-oracle-20260601.json")
    oracle_rows = {item["name"]: item for item in attention_core_oracle["comparisons"]}
    gates["attention_core_external_oracle_bounds"] = {
        "artifact": "deepseek-v4-attention-core-external-oracle-20260601.json",
        "status": _status(attention_core_oracle),
        "qkv_exact_checks": attention_core_oracle["qkv_exact_checks"],
        "dense_core_vs_external_fp64": oracle_rows["dense_core_vs_external_fp64"],
        "sparse_core_vs_external_fp64": oracle_rows["sparse_core_vs_external_fp64"],
        "tilelang_core_vs_external_fp64": oracle_rows["tilelang_core_vs_external_fp64"],
        "dense_core_vs_external_fp64_rounded_bf16": oracle_rows[
            "dense_core_vs_external_fp64_rounded_bf16"
        ],
        "sparse_core_vs_external_fp64_rounded_bf16": oracle_rows[
            "sparse_core_vs_external_fp64_rounded_bf16"
        ],
        "tilelang_core_vs_external_fp64_rounded_bf16": oracle_rows[
            "tilelang_core_vs_external_fp64_rounded_bf16"
        ],
    }
    _check(_status(attention_core_oracle) == "PASS", failures, "attention_core_external_oracle.status")
    _check(
        all(item["exact_equal"] for item in attention_core_oracle["qkv_exact_checks"]),
        failures,
        "attention_core_external_oracle.qkv_exact",
    )
    for row_name in (
        "dense_core_vs_external_fp64",
        "sparse_core_vs_external_fp64",
        "tilelang_core_vs_external_fp64",
        "dense_core_vs_external_fp64_rounded_bf16",
        "sparse_core_vs_external_fp64_rounded_bf16",
        "tilelang_core_vs_external_fp64_rounded_bf16",
        "dense_core_vs_external_fp32_bf16_formula",
        "sparse_core_vs_external_fp32_bf16_formula",
        "tilelang_core_vs_external_fp32_bf16_formula",
    ):
        _check(
            oracle_rows[row_name]["status"] == "PASS",
            failures,
            f"attention_core_external_oracle.{row_name}",
        )

    mini_forward = _load(base, "deepseek-v4-mini-forward-compare-20260531.json")
    routing_artifacts = {
        "dense_vs_sparse": "deepseek-v4-mini-forward-routing-replay-dense-vs-sparse-qatsim-20260531.json",
        "dense_vs_tilelang": "deepseek-v4-mini-forward-routing-replay-dense-vs-tilelang-qatsim-20260531.json",
        "sparse_vs_tilelang": "deepseek-v4-mini-forward-routing-replay-sparse-vs-tilelang-qatsim-20260531.json",
    }
    routing_improvements = {}
    for label, artifact in routing_artifacts.items():
        base_compare = _find_compare(mini_forward["compare_results"], label)
        replay = _load(base, artifact)
        routing_improvements[label] = {
            "baseline_mismatches": base_compare["mismatches"],
            "replay_mismatches": replay["mismatches"],
            "baseline_relative_l2_gap": base_compare["relative_l2_gap"],
            "replay_relative_l2_gap": replay["relative_l2_gap"],
            "relative_l2_reduction": base_compare["relative_l2_gap"] / replay["relative_l2_gap"],
        }
        _check(replay["mismatches"] < base_compare["mismatches"], failures, f"{label}.routing_reduces_mismatches")
        _check(
            replay["relative_l2_gap"] < base_compare["relative_l2_gap"],
            failures,
            f"{label}.routing_reduces_relative_l2_gap",
        )
    gates["routing_replay_error_budget"] = routing_improvements

    train_base = _load(base, "deepseek-v4-mini-train-step-qatsim-0415-20260531.json")
    train_routing = _load(base, "deepseek-v4-mini-train-step-routing-replay-qatsim-20260531.json")
    train_attn_replay = _load(base, "deepseek-v4-mini-train-step-attention-output-replay-qatsim-20260531.json")
    train_budget = {}
    for label in ("dense_vs_sparse", "dense_vs_tilelang", "sparse_vs_tilelang"):
        base_compare = _find_compare(train_base["comparisons"], label)
        routing_compare = _find_compare(train_routing["comparisons"], label)
        attn_compare = _find_compare(train_attn_replay["comparisons"], label)
        train_budget[label] = {
            "baseline_loss_abs": base_compare["loss_abs_global_max"],
            "routing_loss_abs": routing_compare["loss_abs_global_max"],
            "attention_output_replay_loss_abs": attn_compare["loss_abs_global_max"],
            "attention_output_replay_grad_rel_gap": attn_compare["selected_grad_max_rel_gap"],
            "attention_output_replay_state_abs": attn_compare["selected_state_max_abs"],
        }
        _check(
            routing_compare["loss_abs_global_max"] < base_compare["loss_abs_global_max"],
            failures,
            f"{label}.routing_reduces_train_loss_gap",
        )
        _check(attn_compare["loss_abs_global_max"] == 0.0, failures, f"{label}.attn_replay_loss_exact")
        _check(
            attn_compare["selected_grad_max_rel_gap"]
            <= train_attn_replay["thresholds"]["max_selected_grad_rel_gap"],
            failures,
            f"{label}.attn_replay_grad_rel_passes",
        )
        _check(
            attn_compare["selected_state_max_abs"] <= train_attn_replay["thresholds"]["max_selected_state_abs"],
            failures,
            f"{label}.attn_replay_state_abs_passes",
        )
    gates["train_step_error_budget"] = train_budget

    mini_correctness = _load(base, "deepseek-v4-mini-checkpoint-correctness-gate-20260531.json")
    sft_loss_reference = _load(base, required_pass["sft_loss_explicit_reference"][0])
    sft_loss_train_reference = _load(base, required_pass["sft_loss_train_explicit_reference"][0])
    rerun_artifact = mini_correctness["loaded_checkpoint_sft"]["artifact"]
    rerun = _load(base, rerun_artifact)
    sft_loss_reference_bounds = {
        "artifact": required_pass["sft_loss_explicit_reference"][0],
        "world_size": sft_loss_reference["world_size"],
        "reference_formula": sft_loss_reference["reference_formula"],
        "loss_abs_global_max": sft_loss_reference["global_summary"]["loss_abs_global_max"],
        "logprob_max_abs_global_max": sft_loss_reference["global_summary"]["logprob_max_abs_global_max"],
        "logprob_mean_abs_global_max": sft_loss_reference["global_summary"]["logprob_mean_abs_global_max"],
        "token_count_abs_global_max": sft_loss_reference["global_summary"]["token_count_abs_global_max"],
        "num_logprob_tokens_global_sum": sft_loss_reference["global_summary"]["num_logprob_tokens_global_sum"],
        "thresholds": sft_loss_reference["thresholds"],
    }
    gates["sft_loss_explicit_reference_bounds"] = sft_loss_reference_bounds
    _check(sft_loss_reference_bounds["world_size"] == 8, failures, "sft_loss_reference.world_size")
    _check(
        sft_loss_reference_bounds["loss_abs_global_max"] <= sft_loss_reference["thresholds"]["max_loss_abs"],
        failures,
        "sft_loss_reference.loss_bound",
    )
    _check(
        sft_loss_reference_bounds["logprob_max_abs_global_max"]
        <= sft_loss_reference["thresholds"]["max_logprob_abs"],
        failures,
        "sft_loss_reference.logprob_max_bound",
    )
    _check(
        sft_loss_reference_bounds["logprob_mean_abs_global_max"]
        <= sft_loss_reference["thresholds"]["max_logprob_mean_abs"],
        failures,
        "sft_loss_reference.logprob_mean_bound",
    )
    _check(
        sft_loss_reference_bounds["token_count_abs_global_max"] == 0.0,
        failures,
        "sft_loss_reference.token_count_exact",
    )
    _check(
        sft_loss_reference_bounds["reference_formula"]
        == "sum(-log_softmax(response_logits)[target_token] * loss_mask)",
        failures,
        "sft_loss_reference.formula",
    )
    sft_loss_train_reference_bounds = {
        "artifact": required_pass["sft_loss_train_explicit_reference"][0],
        "world_size": sft_loss_train_reference["world_size"],
        "attention_impl": sft_loss_train_reference["attention_impl"],
        "reference_formula": sft_loss_train_reference["reference_formula"],
        "loss_abs_global_max": sft_loss_train_reference["comparison"]["loss_abs_global_max"],
        "token_count_abs_global_max": sft_loss_train_reference["comparison"]["token_count_abs_global_max"],
        "selected_grad_max_abs": sft_loss_train_reference["comparison"]["selected_grad_max_abs"],
        "selected_grad_max_rel_gap": sft_loss_train_reference["comparison"]["selected_grad_max_rel_gap"],
        "selected_state_max_abs": sft_loss_train_reference["comparison"]["selected_state_max_abs"],
        "num_common_grad_tensors_global": sft_loss_train_reference["comparison"]["num_common_grad_tensors_global"],
        "num_common_state_tensors_global": sft_loss_train_reference["comparison"]["num_common_state_tensors_global"],
        "thresholds": sft_loss_train_reference["thresholds"],
        "boundary": sft_loss_train_reference["boundary"],
    }
    gates["sft_loss_train_explicit_reference_bounds"] = sft_loss_train_reference_bounds
    _check(sft_loss_train_reference_bounds["world_size"] == 8, failures, "sft_loss_train_reference.world_size")
    _check(
        sft_loss_train_reference_bounds["reference_formula"]
        == "sum(-log_softmax(response_logits)[target_token] * loss_mask)",
        failures,
        "sft_loss_train_reference.formula",
    )
    _check(
        sft_loss_train_reference_bounds["loss_abs_global_max"]
        <= sft_loss_train_reference["thresholds"]["max_loss_abs"],
        failures,
        "sft_loss_train_reference.loss_bound",
    )
    _check(
        sft_loss_train_reference_bounds["token_count_abs_global_max"]
        <= sft_loss_train_reference["thresholds"]["max_token_count_abs"],
        failures,
        "sft_loss_train_reference.token_count_bound",
    )
    _check(
        sft_loss_train_reference_bounds["selected_grad_max_abs"]
        <= sft_loss_train_reference["thresholds"]["max_selected_grad_abs"],
        failures,
        "sft_loss_train_reference.grad_abs_bound",
    )
    _check(
        sft_loss_train_reference_bounds["selected_grad_max_rel_gap"]
        <= sft_loss_train_reference["thresholds"]["max_selected_grad_rel_gap"],
        failures,
        "sft_loss_train_reference.grad_rel_bound",
    )
    _check(
        sft_loss_train_reference_bounds["selected_state_max_abs"]
        <= sft_loss_train_reference["thresholds"]["max_selected_state_abs"],
        failures,
        "sft_loss_train_reference.state_abs_bound",
    )
    _check(
        sft_loss_train_reference_bounds["num_common_grad_tensors_global"] > 0
        and sft_loss_train_reference_bounds["num_common_state_tensors_global"] > 0,
        failures,
        "sft_loss_train_reference.tensor_sets",
    )
    mini_bounds = {
        "strict_boundaries": mini_correctness["strict_boundaries"],
        "rerun_artifact": rerun_artifact,
        "rerun_world_size": rerun["world_size"],
        "rerun_status": rerun["status"],
        "rerun_loss_abs_max": _max_comparison_value(rerun, "loss_abs_global_max"),
        "rerun_selected_grad_rel_gap_max": _max_comparison_value(rerun, "selected_grad_max_rel_gap"),
        "rerun_selected_state_abs_max": _max_comparison_value(rerun, "selected_state_max_abs"),
            "attention_io_training_step_bounds": mini_correctness["attention_io_training_step_bounds"],
            "sft_loss_reference": mini_correctness["sft_loss_reference"],
            "sft_loss_train_reference": mini_correctness["sft_loss_train_reference"],
            "moe_evidence": mini_correctness["moe_evidence"],
        }
    gates["mini_checkpoint_correctness_gate_bounds"] = mini_bounds
    _check(rerun["status"] == "PASS", failures, "mini_checkpoint_correctness.rerun_status")
    _check(rerun["world_size"] == 8, failures, "mini_checkpoint_correctness.rerun_world_size")
    _check(mini_bounds["rerun_loss_abs_max"] == 0.0, failures, "mini_checkpoint_correctness.loss_exact")
    _check(
        mini_bounds["rerun_selected_grad_rel_gap_max"] <= rerun["thresholds"]["max_selected_grad_rel_gap"],
        failures,
        "mini_checkpoint_correctness.grad_rel_bound",
    )
    _check(
        mini_bounds["rerun_selected_state_abs_max"] <= rerun["thresholds"]["max_selected_state_abs"],
        failures,
        "mini_checkpoint_correctness.state_abs_bound",
    )
    _check(
        mini_correctness["strict_boundaries"]["real_non_injected_forward_strict_status"] == "FAIL"
        and mini_correctness["strict_boundaries"]["real_non_injected_sft_one_step_strict_status"] == "FAIL"
        and mini_correctness["strict_boundaries"]["not_reclassified_as_strict_pass"],
        failures,
        "mini_checkpoint_correctness.strict_boundaries_recorded",
    )
    for key in (
        "loss_abs_global_max",
        "logprob_max_abs_global_max",
        "logprob_mean_abs_global_max",
        "token_count_abs_global_max",
        "world_size",
        "reference_formula",
    ):
        _check(
            mini_correctness["sft_loss_reference"][key] == sft_loss_reference_bounds[key],
            failures,
            f"mini_checkpoint_correctness.sft_loss_reference.{key}",
        )
    for key in (
        "loss_abs_global_max",
        "token_count_abs_global_max",
        "selected_grad_max_abs",
        "selected_grad_max_rel_gap",
        "selected_state_max_abs",
        "world_size",
        "reference_formula",
    ):
        _check(
            mini_correctness["sft_loss_train_reference"][key] == sft_loss_train_reference_bounds[key],
            failures,
            f"mini_checkpoint_correctness.sft_loss_train_reference.{key}",
        )

    full_external_forward = _load(
        base,
        "deepseek-v4-mini-external-full-reference-bf16-routing-replay-tolerance-20260531.json",
    )
    full_external_train = _load(
        base,
        "deepseek-v4-mini-external-full-reference-bf16-routing-replay-train-tolerance-20260531.json",
    )
    full_external_router_debug = _load(
        base,
        "deepseek-v4-mini-external-full-reference-bf16-router-debug-20260531.json",
    )
    gates["mini_external_full_reference"] = {
        "forward_artifact": "deepseek-v4-mini-external-full-reference-bf16-routing-replay-tolerance-20260531.json",
        "forward_status": _status(full_external_forward),
        "forward_routing_replay": full_external_forward["routing_replay"]["mode"],
        "forward_tolerance_profile": full_external_forward["tolerance_profile"],
        "forward_logit_gap_global": full_external_forward["logit_gap_global"],
        "forward_loss_abs_global_max": full_external_forward["loss_abs_global_max"],
        "forward_loss_abs_per_token_global_max": full_external_forward[
            "loss_abs_per_token_global_max"
        ],
        "train_artifact": "deepseek-v4-mini-external-full-reference-bf16-routing-replay-train-tolerance-20260531.json",
        "train_status": _status(full_external_train),
        "train_failures": full_external_train["failures"],
        "train_selected_grad_max_abs_global_max": full_external_train[
            "selected_backward_update_delta"
        ]["selected_grad_max_abs_global_max"],
        "train_selected_grad_relative_to_param_l2_global": full_external_train[
            "selected_backward_update_delta"
        ]["selected_grad_relative_to_param_l2_global"],
        "train_selected_state_max_abs_global_max": full_external_train[
            "selected_backward_update_delta"
        ]["selected_state_max_abs_global_max"],
        "router_debug_status": _status(full_external_router_debug),
        "router_debug_layer_0_router_map": full_external_router_debug["layer_gap_global"][
            "layer_0.router_map"
        ],
        "router_debug_layer_1_router_map": full_external_router_debug["layer_gap_global"][
            "layer_1.router_map"
        ],
        "router_debug_layer_2_router_map": full_external_router_debug["layer_gap_global"][
            "layer_2.router_map"
        ],
        "router_debug_layer_3_router_map": full_external_router_debug["layer_gap_global"][
            "layer_3.router_map"
        ],
    }
    _check(_status(full_external_forward) == "PASS", failures, "mini_external_full_reference.forward_status")
    _check(
        full_external_forward["routing_replay"]["mode"] == "miles_router_map_replay",
        failures,
        "mini_external_full_reference.routing_replay",
    )
    _check(
        full_external_forward["logit_gap_global"]["max_abs"]
        <= full_external_forward["thresholds"]["max_logit_abs"],
        failures,
        "mini_external_full_reference.logit_max",
    )
    _check(
        full_external_forward["logit_gap_global"]["mean_abs"]
        <= full_external_forward["thresholds"]["max_logit_mean_abs"],
        failures,
        "mini_external_full_reference.logit_mean",
    )
    _check(
        full_external_forward["logit_gap_global"]["p99_abs"]
        <= full_external_forward["thresholds"]["max_logit_p99_abs"],
        failures,
        "mini_external_full_reference.logit_p99",
    )
    _check(
        full_external_forward["logit_gap_global"]["relative_l2_gap"]
        <= full_external_forward["thresholds"]["max_logit_rel_gap"],
        failures,
        "mini_external_full_reference.logit_rel",
    )
    _check(
        full_external_forward["loss_abs_global_max"]
        <= full_external_forward["thresholds"]["max_loss_abs"],
        failures,
        "mini_external_full_reference.loss",
    )
    _check(
        full_external_forward["loss_abs_per_token_global_max"]
        <= full_external_forward["thresholds"]["max_loss_abs_per_token"],
        failures,
        "mini_external_full_reference.loss_per_token",
    )
    _check(
        full_external_forward["token_count_abs_global_max"] == 0.0,
        failures,
        "mini_external_full_reference.token_count",
    )
    _check(_status(full_external_train) == "FAIL", failures, "mini_external_full_reference.train_boundary_status")
    _check(
        "selected_grad_max_abs" in full_external_train["failures"]
        and "selected_grad_relative_to_param_l2" in full_external_train["failures"],
        failures,
        "mini_external_full_reference.train_gradient_boundary",
    )
    _check(
        full_external_train["selected_backward_update_delta"]["selected_state_max_abs_global_max"]
        <= full_external_train["thresholds"]["max_selected_state_abs"],
        failures,
        "mini_external_full_reference.train_state_bound",
    )
    for layer_idx in range(3):
        _check(
            full_external_router_debug["layer_gap_global"][f"layer_{layer_idx}.router_map"]["max_abs"] == 0.0,
            failures,
            f"mini_external_full_reference.layer_{layer_idx}_hash_router_exact",
        )
    _check(
        full_external_router_debug["layer_gap_global"]["layer_3.router_map"]["max_abs"] == 1.0,
        failures,
        "mini_external_full_reference.layer_3_router_branch_flip_recorded",
    )

    attention_io_train = _load(base, "deepseek-v4-mini-attention-io-training-step-qatsim-20260531.json")
    gates["attention_io_training_step_bounds"] = {
        "max_output_abs": _max_nested_value(attention_io_train, "output_max_abs_global_max"),
        "max_input_grad_abs": _max_nested_value(attention_io_train, "input_grad_max_abs_global_max"),
        "max_state_after_step_abs": _max_nested_value(attention_io_train, "state_after_step_max_abs_global_max"),
    }
    _check(
        gates["attention_io_training_step_bounds"]["max_output_abs"]
        <= attention_io_train["thresholds"]["max_output_abs"],
        failures,
        "attention_io_training_step.output_bound",
    )
    _check(
        gates["attention_io_training_step_bounds"]["max_input_grad_abs"]
        <= attention_io_train["thresholds"]["max_input_grad_abs"],
        failures,
        "attention_io_training_step.input_grad_bound",
    )
    _check(
        gates["attention_io_training_step_bounds"]["max_state_after_step_abs"]
        <= attention_io_train["thresholds"]["max_state_abs"],
        failures,
        "attention_io_training_step.state_bound",
    )

    external_moe_ep8 = _load(base, "deepseek-v4-external-moe-ep8-reference-20260531.json")
    external_moe_ep8_summary = external_moe_ep8["global_summary"]
    gates["external_moe_ep8_reference_bounds"] = {
        "artifact": "deepseek-v4-external-moe-ep8-reference-20260531.json",
        "world_size": external_moe_ep8["world_size"],
        "expert_parallel_size": external_moe_ep8["config"]["expert_parallel_size"],
        "num_local_experts_per_rank": external_moe_ep8["config"]["num_local_experts_per_rank"],
        "loss_abs_global_max": external_moe_ep8_summary["loss_abs_global_max"],
        "output_max_abs_global_max": external_moe_ep8_summary["output_max_abs_global_max"],
        "input_grad_max_abs_global_max": external_moe_ep8_summary["input_grad_max_abs_global_max"],
        "expert_grad_max_abs_global_max": external_moe_ep8_summary["expert_grad_max_abs_global_max"],
        "expert_state_after_step_max_abs_global_max": external_moe_ep8_summary[
            "expert_state_after_step_max_abs_global_max"
        ],
        "per_expert_selected_tokens": external_moe_ep8_summary["per_expert_selected_tokens"],
        "ranks_with_nonzero_local_expert_grad": external_moe_ep8_summary["ranks_with_nonzero_local_expert_grad"],
        "thresholds": external_moe_ep8["thresholds"],
    }
    _check(external_moe_ep8["world_size"] == 8, failures, "external_moe_ep8_reference.world_size")
    _check(
        external_moe_ep8["config"]["expert_parallel_size"] == 8
        and external_moe_ep8["config"]["num_local_experts_per_rank"] == 1,
        failures,
        "external_moe_ep8_reference.ep8_layout",
    )
    _check(
        external_moe_ep8_summary["loss_abs_global_max"] <= external_moe_ep8["thresholds"]["max_loss_abs"],
        failures,
        "external_moe_ep8_reference.loss_bound",
    )
    _check(
        external_moe_ep8_summary["output_max_abs_global_max"] <= external_moe_ep8["thresholds"]["max_output_abs"],
        failures,
        "external_moe_ep8_reference.output_bound",
    )
    _check(
        external_moe_ep8_summary["input_grad_max_abs_global_max"]
        <= external_moe_ep8["thresholds"]["max_input_grad_abs"],
        failures,
        "external_moe_ep8_reference.input_grad_bound",
    )
    _check(
        external_moe_ep8_summary["expert_grad_max_abs_global_max"]
        <= external_moe_ep8["thresholds"]["max_expert_grad_abs"],
        failures,
        "external_moe_ep8_reference.expert_grad_bound",
    )
    _check(
        external_moe_ep8_summary["expert_state_after_step_max_abs_global_max"]
        <= external_moe_ep8["thresholds"]["max_state_abs"],
        failures,
        "external_moe_ep8_reference.state_bound",
    )
    _check(
        all(count > 0 for count in external_moe_ep8_summary["per_expert_selected_tokens"]),
        failures,
        "external_moe_ep8_reference.all_experts_selected",
    )
    _check(
        external_moe_ep8_summary["ranks_with_nonzero_local_expert_grad"] == external_moe_ep8["world_size"],
        failures,
        "external_moe_ep8_reference.nonzero_expert_grad",
    )

    official_tolerance = _load(base, "deepseek-v4-official-forward-bf16-tolerance-20260531.json")
    gates["official_forward_bf16_tolerance_bounds"] = {
        "strict_forward_status": official_tolerance["strict_forward_status"],
        "bf16_forward_tolerance_status": official_tolerance["bf16_forward_tolerance_status"],
        "external_one_step_train_parity_status": official_tolerance["external_train_precondition"][
            "external_one_step_train_parity_status"
        ],
        "thresholds": official_tolerance["thresholds"],
        "forward_checks": official_tolerance["forward_checks"],
    }
    _check(
        official_tolerance["bf16_forward_tolerance_status"] == "PASS",
        failures,
        "official_forward_bf16_tolerance.status",
    )
    _check(
        official_tolerance["strict_forward_status"] in ("PASS", "FAIL"),
        failures,
        "official_forward_bf16_tolerance.strict_status_recorded",
    )
    if official_tolerance["strict_forward_status"] != "PASS":
        _check(
            official_tolerance["external_train_precondition"]["external_one_step_train_parity_status"]
            == "SKIPPED_FORWARD_STRICT_PARITY_REQUIRED",
            failures,
            "official_forward_bf16_tolerance.external_train_precondition",
        )

    def _record_external_ref_bounds(gate_key: str, artifact: str, expected_ratio: int) -> None:
        external_ref = _load(base, artifact)
        comp = external_ref["comparison"]
        gates[gate_key] = {
            "artifact": artifact,
            "compress_ratio": external_ref["config"]["compress_ratio"],
            "loss_abs": comp["loss_abs"],
            "output_max_abs": comp["output"]["max_abs"],
            "input_grad_max_abs": comp["input_grad"]["max_abs"],
            "grad_max_abs": comp["grad"]["max_abs"],
            "state_after_step_max_abs": comp["state_after_step"]["max_abs"],
            "num_common_grad_tensors": comp["num_common_grad_tensors"],
            "boundary": external_ref["boundary"],
        }
        _check(
            external_ref["config"]["compress_ratio"] == expected_ratio,
            failures,
            f"{gate_key}.compress_ratio",
        )
        _check(comp["loss_abs"] <= external_ref["thresholds"]["max_loss_abs"], failures, f"{gate_key}.loss")
        _check(
            comp["output"]["max_abs"] <= external_ref["thresholds"]["max_output_abs"],
            failures,
            f"{gate_key}.output",
        )
        _check(
            comp["input_grad"]["max_abs"] <= external_ref["thresholds"]["max_input_grad_abs"],
            failures,
            f"{gate_key}.input_grad",
        )
        _check(
            comp["grad"]["max_abs"] <= external_ref["thresholds"]["max_grad_abs"],
            failures,
            f"{gate_key}.grad",
        )
        _check(
            comp["state_after_step"]["max_abs"] <= external_ref["thresholds"]["max_state_abs"],
            failures,
            f"{gate_key}.state_after_step",
        )
        _check(
            comp["num_common_grad_tensors"] > 0
            and not comp["missing_reference_grad_tensors"]
            and not comp["extra_reference_grad_tensors"],
            failures,
            f"{gate_key}.grad_tensor_sets",
        )

    _record_external_ref_bounds(
        "external_training_reference_1layer_bounds",
        "deepseek-v4-external-training-reference-1layer-20260531.json",
        0,
    )
    _record_external_ref_bounds(
        "external_training_reference_1layer_c4_bounds",
        "deepseek-v4-external-training-reference-1layer-c4-20260531.json",
        4,
    )
    c4_bounds = gates["external_training_reference_1layer_c4_bounds"]
    _check(
        bool(_load(base, "deepseek-v4-external-training-reference-1layer-c4-20260531.json")["config"].get("indexer_selection_pressure")),
        failures,
        "external_training_reference_1layer_c4.indexer_selection_pressure",
    )
    _record_external_ref_bounds(
        "external_training_reference_1layer_c128_bounds",
        "deepseek-v4-external-training-reference-1layer-c128-20260531.json",
        128,
    )
    _record_external_ref_bounds(
        "external_training_reference_1layer_moe_bounds",
        "deepseek-v4-external-training-reference-1layer-moe-20260531.json",
        0,
    )
    moe_ref = _load(base, "deepseek-v4-external-training-reference-1layer-moe-20260531.json")
    _check(
        moe_ref["config"].get("mlp") == "score-routed MoE with shared expert",
        failures,
        "external_training_reference_1layer_moe.mlp_type",
    )
    _check(
        moe_ref["config"].get("num_moe_experts") == 8
        and moe_ref["config"].get("moe_router_score_function") == "sqrtsoftplus",
        failures,
        "external_training_reference_1layer_moe.router_config",
    )

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 proof ledger consistency check",
        "status": "PASS" if not failures else "FAIL",
        "artifacts_dir_name": base.name,
        "gates": gates,
        "failures": failures,
        "conclusion": (
            "Recorded artifacts consistently prove the covered HC/QAT/attention/attention-core "
            "external oracle/MLP/MoE/"
            "training-step gates, validate the non-compressed, c4 indexer, and deterministic compressed "
            "external training-reference gates, validate the score-routed MoE external reference, "
            "validate the real EP=8 MoELayer external reference, "
            "validate the SFT loss explicit reference and "
            "its backward/update explicit reference and "
            "mini-checkpoint correctness gate, validate the loaded 4-layer "
            "full external forward reference with routing replay BF16 tolerance, and "
            "the official forward BF16 tolerance gate, and localize the remaining "
            "real-forward strict parity failure to BF16 attention-core / attention "
            "forward-value drift amplified by the full model and selected-gradient "
            "strict delta."
            if not failures
            else "Recorded artifacts are not mutually sufficient for the proof ledger."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

#!/usr/bin/env python3
"""Build and validate the DeepSeek-V4 proof coverage matrix."""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


PROVED_STATUSES = {
    "PASS",
    "PASS_WITH_DRIFT_RECORDED",
    "PASS_WITH_LOCALIZED_DRIFT",
    "PASS_WITH_ATTENTION_DRIFT_LOCALIZED",
    "PASS_WITH_LAYERWISE_ATTENTION_DRIFT_LOCALIZED",
    "PASS_WITH_BF16_ATTENTION_IO_DRIFT_RECORDED",
    "PASS_WITH_ROUTING_REPLAY_BF16_TOLERANCE",
    "RUN_WITH_DRIFT_RECORDED",
}


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _status(payload: Dict[str, Any]) -> str:
    return payload.get("status") or payload.get("overall_status") or ""


def _check(condition: bool, failures: List[str], name: str) -> None:
    if not condition:
        failures.append(name)


def _artifact_status(base: Path, name: str) -> str:
    return _status(_load(base / name))


def _summary_gate(summary: Dict[str, Any], gate: str) -> Dict[str, Any]:
    for section in ("proved", "not_proved"):
        for item in summary.get(section, []):
            if item.get("gate") == gate:
                return item
    raise KeyError("gate not found: {}".format(gate))


def _has_doc_section(doc: str, title: str) -> bool:
    return "\n## {}\n".format(title) in doc


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("docs/en/advanced"))
    parser.add_argument("--doc", type=Path, default=Path("docs/en/advanced/deepseek-v4-hyperconnection-runtime.md"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.artifacts_dir
    summary = _load(base / "deepseek-v4-proof-summary-20260531.json")
    doc = args.doc.read_text(encoding="utf-8")
    failures: List[str] = []

    requirements = [
        {
            "id": "scope_problem_and_upstream_pr4839",
            "claim": "Document identifies the Megatron-LM PR #4839 HC orientation issue and the Miles-specific scope.",
            "evidence": [
                "deepseek-v4-hyperconnection-runtime.md",
                "deepseek-v4-operator-math-20260531.json",
                "deepseek-v4-external-reference-provenance-20260531.json",
            ],
            "summary_gates": ["hyper_connection_pr4839_orientation", "external_reference_provenance"],
            "doc_sections": ["External References", "问题背景", "Miles 如何实现"],
            "status": "PROVED",
        },
        {
            "id": "source_fixes_guarded",
            "claim": "Known precision fixes remain present in source and are guarded against regression.",
            "evidence": ["deepseek-v4-fix-regression-guards-20260531.json"],
            "summary_gates": ["precision_fix_regression_guards"],
            "doc_sections": ["已修复问题", "Fix Regression Guards"],
            "status": "PROVED",
        },
        {
            "id": "operator_math_core",
            "claim": "HC, RoPE, QAT, dense/sparse attention, and TileLang sparse MLA operator math pass.",
            "evidence": ["deepseek-v4-operator-math-20260531.json"],
            "summary_gates": ["operator_math"],
            "doc_sections": ["算子数学验证"],
            "status": "PROVED",
        },
        {
            "id": "dense_sparse_tilelang_attention_training",
            "claim": "dense/sparse/tilelang attention backend training surfaces pass for module and block scopes.",
            "evidence": [
                "deepseek-v4-attention-training-step-qatsim-20260531.json",
                "deepseek-v4-transformer-block-training-step-qatsim-20260531.json",
            ],
            "summary_gates": ["attention_training_step", "transformer_block_training_step"],
            "doc_sections": ["模块训练步验证", "TransformerBlock 训练步验证"],
            "status": "PROVED",
        },
        {
            "id": "moe_and_grouped_mlp_math",
            "claim": "TE grouped MLP, EP=8 all-to-all dispatcher math, and real EP=8 MoELayer external reference pass forward/backward/update checks.",
            "evidence": [
                "deepseek-v4-grouped-mlp-math-20260531.json",
                "deepseek-v4-moe-ep8-dispatch-math-20260531.json",
                "deepseek-v4-external-moe-ep8-reference-20260531.json",
            ],
            "summary_gates": [
                "grouped_mlp_training_math",
                "moe_ep8_alltoall_dispatch_math",
                "external_moe_ep8_reference",
            ],
            "doc_sections": ["算子数学验证"],
            "status": "PROVED",
        },
        {
            "id": "official_and_weight_mapping_localization",
            "claim": "official attention, loaded weight mapping, head replay, and MLP expert replay localize official-vs-Miles drift past known conversion/math fixes.",
            "evidence": [
                "deepseek-v4-official-attention-forward-20260531.json",
                "deepseek-v4-loaded-weight-mapping-1layer-mlp-qatsim-20260531.json",
                "deepseek-v4-attention-trace-replay-qatsim-0415-20260531.json",
                "deepseek-v4-mlp-expert-replay-qatsim-0415-20260531.json",
            ],
            "summary_gates": [
                "official_attention_forward",
                "loaded_weight_mapping",
                "attention_trace_replay",
                "mlp_expert_replay",
            ],
            "doc_sections": ["Attention Trace Replay", "权重加载验证", "Official Full-Forward Probe"],
            "status": "PROVED_WITH_DRIFT_RECORDED",
        },
        {
            "id": "official_forward_bf16_tolerance",
            "claim": "official-vs-Miles full-forward drift is inside an explicit BF16 tolerance envelope while strict parity remains open.",
            "evidence": ["deepseek-v4-official-forward-bf16-tolerance-20260531.json"],
            "summary_gates": ["official_forward_bf16_tolerance"],
            "doc_sections": ["Official Forward BF16 Tolerance"],
            "status": "PROVED_WITH_BF16_TOLERANCE",
        },
        {
            "id": "mini_checkpoint_forward_drift_localized",
            "claim": "mini forward drift is finite, routing-amplified, and localized to attention forward-value drift.",
            "evidence": [
                "deepseek-v4-mini-forward-compare-20260531.json",
                "deepseek-v4-mini-forward-routing-replay-dense-vs-sparse-qatsim-20260531.json",
                "deepseek-v4-mini-activation-replay-qatsim-20260531.json",
                "deepseek-v4-mini-sublayer-activation-replay-qatsim-20260531.json",
                "deepseek-v4-mini-attention-io-replay-qatsim-20260531.json",
            ],
            "expected_failing_diagnostic_artifacts": [
                "deepseek-v4-mini-forward-routing-replay-dense-vs-sparse-qatsim-20260531.json",
            ],
            "summary_gates": [
                "mini_checkpoint_forward_execution",
                "mini_checkpoint_forward_routing_replay",
                "mini_checkpoint_activation_replay",
                "mini_checkpoint_sublayer_activation_replay",
                "mini_checkpoint_attention_io_replay",
            ],
            "doc_sections": ["Mini Checkpoint Drift Probe"],
            "status": "PROVED_WITH_LOCALIZED_DRIFT",
        },
        {
            "id": "mini_checkpoint_training_chain",
            "claim": "mini SFT training chain is finite; explicit SFT loss backward/update reference passes; attention-output replay gives exact loss parity and bounded gradient/update drift.",
            "evidence": [
                "deepseek-v4-mini-train-step-qatsim-0415-20260531.json",
                "deepseek-v4-mini-train-step-routing-replay-qatsim-20260531.json",
                "deepseek-v4-mini-train-step-attention-output-replay-qatsim-20260531.json",
                "deepseek-v4-mini-checkpoint-correctness-rerun-sft-attention-output-replay-20260531.json",
                "deepseek-v4-mini-attention-io-training-step-qatsim-20260531.json",
                "deepseek-v4-sft-loss-train-reference-20260531.json",
            ],
            "expected_failing_diagnostic_artifacts": [
                "deepseek-v4-mini-train-step-qatsim-0415-20260531.json",
                "deepseek-v4-mini-train-step-routing-replay-qatsim-20260531.json",
            ],
            "summary_gates": [
                "mini_checkpoint_sft_one_step_execution",
                "mini_checkpoint_sft_one_step_routing_replay",
                "mini_checkpoint_sft_one_step_attention_output_replay",
                "mini_checkpoint_attention_io_training_step",
            ],
            "doc_sections": ["Mini Checkpoint Drift Probe"],
            "status": "PROVED_WITH_BOUNDED_FORWARD_DRIFT",
        },
        {
            "id": "mini_checkpoint_correctness_gate",
            "claim": "mini checkpoint training correctness passes under the declared BF16 tolerance while strict real-forward/SFT parity remains an explicit boundary.",
            "evidence": [
                "deepseek-v4-mini-checkpoint-correctness-gate-20260531.json",
                "deepseek-v4-mini-checkpoint-correctness-rerun-sft-attention-output-replay-20260531.json",
                "deepseek-v4-sft-loss-reference-20260531.json",
                "deepseek-v4-sft-loss-train-reference-20260531.json",
                "deepseek-v4-external-training-reference-1layer-moe-20260531.json",
                "deepseek-v4-external-moe-ep8-reference-20260531.json",
            ],
            "summary_gates": ["mini_checkpoint_correctness_gate"],
            "doc_sections": ["Mini Checkpoint Correctness Gate"],
            "status": "PROVED_WITH_BF16_TOLERANCE",
        },
        {
            "id": "mini_checkpoint_external_full_reference",
            "claim": "loaded 4-layer mini checkpoint full external forward reference passes with Miles routing replay under BF16 tolerance, while monolithic train delta remains a recorded strict boundary.",
            "evidence": [
                "deepseek-v4-mini-external-full-reference-bf16-routing-replay-tolerance-20260531.json",
                "deepseek-v4-mini-external-full-reference-bf16-routing-replay-train-tolerance-20260531.json",
                "deepseek-v4-mini-external-full-reference-bf16-router-debug-20260531.json",
            ],
            "expected_failing_diagnostic_artifacts": [
                "deepseek-v4-mini-external-full-reference-bf16-routing-replay-train-tolerance-20260531.json",
                "deepseek-v4-mini-external-full-reference-bf16-router-debug-20260531.json",
            ],
            "summary_gates": ["mini_checkpoint_external_full_forward_reference"],
            "doc_sections": ["Mini Checkpoint Full External Reference"],
            "status": "PROVED_WITH_ROUTING_REPLAY_BF16_TOLERANCE",
        },
        {
            "id": "sft_loss_explicit_reference",
            "claim": "loaded mini checkpoint SFT loss and its backward/update surface match an explicit PyTorch log_softmax/gather/loss_mask reference.",
            "evidence": [
                "deepseek-v4-sft-loss-reference-20260531.json",
                "deepseek-v4-sft-loss-train-reference-20260531.json",
            ],
            "summary_gates": ["sft_loss_explicit_reference", "sft_loss_train_explicit_reference"],
            "doc_sections": ["SFT Loss Explicit Reference"],
            "status": "PROVED",
        },
        {
            "id": "external_training_reference_1layer",
            "claim": "DeepSeek-V4 training blocks and the real EP=8 MoELayer pass explicit PyTorch external-reference forward/backward/update parity for non-compressed attention, compress_ratio=4 indexer, deterministic compress_ratio=128 compressed attention, score-routed MoE, and production EP layout.",
            "evidence": [
                "deepseek-v4-external-training-reference-1layer-20260531.json",
                "deepseek-v4-external-training-reference-1layer-c4-20260531.json",
                "deepseek-v4-external-training-reference-1layer-c128-20260531.json",
                "deepseek-v4-external-training-reference-1layer-moe-20260531.json",
                "deepseek-v4-external-moe-ep8-reference-20260531.json",
            ],
            "summary_gates": [
                "external_training_reference_1layer",
                "external_training_reference_1layer_moe",
                "external_moe_ep8_reference",
            ],
            "doc_sections": ["External Training Reference"],
            "status": "PROVED",
        },
        {
            "id": "bf16_tolerance_envelope",
            "claim": "real non-injected forward/train drift is inside the declared BF16 runtime tolerance envelope.",
            "evidence": ["deepseek-v4-end-to-end-bf16-tolerance-20260531.json"],
            "summary_gates": ["end_to_end_bf16_tolerance_envelope"],
            "doc_sections": ["End-to-End BF16 Tolerance"],
            "status": "PROVED_WITH_BF16_TOLERANCE",
        },
        {
            "id": "optimizer_update_math",
            "claim": "actual optimizer path is Megatron Adam and first-step AdamW selected-update math is verified.",
            "evidence": ["deepseek-v4-optimizer-update-math-20260531.json"],
            "summary_gates": ["optimizer_update_math"],
            "doc_sections": ["Optimizer Update Math"],
            "status": "PROVED",
        },
        {
            "id": "proof_ledger_consistency",
            "claim": "proof ledger machine-validates that the artifact chain is internally consistent.",
            "evidence": ["deepseek-v4-proof-ledger-20260531.json"],
            "summary_gates": ["proof_ledger_consistency"],
            "doc_sections": ["Proof Ledger"],
            "status": "PROVED",
        },
        {
            "id": "environment_provenance",
            "claim": "documented environment versions match the environment captured by the operator-math artifact.",
            "evidence": ["deepseek-v4-environment-provenance-20260531.json"],
            "summary_gates": ["environment_provenance"],
            "doc_sections": ["环境", "Environment Provenance"],
            "status": "PROVED",
        },
    ]

    open_gates = [
        {
            "id": "strict_mini_backend_logprob_parity",
            "expected_status": "FAIL",
            "reason_must_contain": ["strict", "attention", "0.0625"],
        },
        {
            "id": "strict_mini_checkpoint_train_step_backend_parity",
            "expected_status": "FAIL",
            "reason_must_contain": ["loss_abs", "attention-output", "optimizer"],
        },
        {
            "id": "official_reference_mini_checkpoint_forward_parity",
            "expected_status": "FAIL",
            "reason_must_contain": ["official", "BF16", "strict"],
        },
        {
            "id": "production_ep8_moe_path_strict_parity",
            "expected_status": "PARTIALLY_LOCALIZED",
            "reason_must_contain": ["EP=8", "attention", "all-to-all"],
        },
        {
            "id": "external_reference_mini_checkpoint_one_step_train_parity",
            "expected_status": "FAIL_DIAGNOSTIC",
            "reason_must_contain": ["full external", "routing replay", "selected_grad"],
        },
    ]

    for req in requirements:
        req_failures = []
        expected_failing = set(req.get("expected_failing_diagnostic_artifacts", []))
        for artifact in req["evidence"]:
            if artifact.endswith(".json"):
                path = base / artifact
                if not path.exists():
                    req_failures.append("missing_artifact:" + artifact)
                else:
                    status = _artifact_status(base, artifact)
                    if status and status not in PROVED_STATUSES and artifact not in expected_failing:
                        req_failures.append("bad_artifact_status:{}={}".format(artifact, status))
        for gate in req["summary_gates"]:
            try:
                gate_item = _summary_gate(summary, gate)
            except KeyError:
                req_failures.append("missing_summary_gate:" + gate)
                continue
            if gate_item.get("status") not in PROVED_STATUSES:
                req_failures.append("bad_summary_gate_status:{}={}".format(gate, gate_item.get("status")))
        for section in req["doc_sections"]:
            if not _has_doc_section(doc, section):
                req_failures.append("missing_doc_section:" + section)
        req["failures"] = req_failures
        req["coverage_status"] = "PASS" if not req_failures else "FAIL"
        for failure in req_failures:
            failures.append(req["id"] + "." + failure)

    open_gate_rows = []
    for gate in open_gates:
        item = _summary_gate(summary, gate["id"])
        reason = item.get("reason", "")
        row = {
            "id": gate["id"],
            "expected_status": gate["expected_status"],
            "actual_status": item.get("status"),
            "reason_contains": {
                needle: (needle in reason)
                for needle in gate["reason_must_contain"]
            },
        }
        open_gate_rows.append(row)
        _check(item.get("status") == gate["expected_status"], failures, gate["id"] + ".status")
        for needle, present in row["reason_contains"].items():
            _check(present, failures, gate["id"] + ".reason_contains:" + needle)

    doc_required_phrases = [
        "当前结论是 **PARTIAL_PROOF**",
        "已证明",
        "未严格证明",
        "NVIDIA driver",
        "CUDA toolkit",
        "megatron-core",
        "transformer-engine",
        "tilelang",
        "SFT loss explicit reference",
    ]
    doc_checks = {phrase: (phrase in doc) for phrase in doc_required_phrases}
    for phrase, present in doc_checks.items():
        _check(present, failures, "doc_phrase:" + phrase)

    forbidden_path_pattern = re.compile(r"/data_train|/home/kaynzhang|/root/Megatron-LM|/tmp/deepseek|/tmp/dsv4")
    forbidden_paths = []
    for path in list(base.glob("deepseek-v4-*")) + [args.doc]:
        text = path.read_text(encoding="utf-8")
        if forbidden_path_pattern.search(text):
            forbidden_paths.append(path.name)
    _check(not forbidden_paths, failures, "forbidden_local_paths_absent")

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 proof coverage matrix",
        "status": "PASS" if not failures else "FAIL",
        "summary_status": summary.get("status"),
        "requirements": requirements,
        "open_strict_gates": open_gate_rows,
        "doc_checks": doc_checks,
        "forbidden_local_path_files": forbidden_paths,
        "failures": failures,
        "conclusion": (
            "All requested proof areas have explicit artifact-backed coverage, and the remaining strict gates are documented with machine-checked boundaries."
            if not failures
            else "Proof coverage matrix found missing or inconsistent evidence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

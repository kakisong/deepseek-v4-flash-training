#!/usr/bin/env python3
"""Static regression guards for DeepSeek-V4 precision fixes.

The numerical verifiers prove behavior; this script guards the source shapes of
the fixes that made those verifiers pass.  It intentionally uses text checks so
it can run with the local Python 3.6 interpreter even though some source files
use newer type annotation syntax.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _check(condition: bool, failures: List[str], name: str) -> bool:
    if not condition:
        failures.append(name)
        return False
    return True


def _contains(text: str, pattern: str) -> bool:
    return pattern in text


def _compact_contains(text: str, pattern: str) -> bool:
    return _compact(pattern) in _compact(text)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("docs/en/advanced"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root
    failures: List[str] = []
    checks: Dict[str, Any] = {}

    deepseek_v4 = _read(root / "miles_plugins/models/deepseek_v4/deepseek_v4.py")
    act_quant = _read(root / "miles_plugins/models/deepseek_v4/ops/kernel/act_quant.py")
    qat = _read(root / "miles_plugins/models/deepseek_v4/ops/qat.py")
    compressor = _read(root / "miles_plugins/models/deepseek_v4/ops/compressor.py")

    checks["attention_trace_gating"] = {
        "has_trace_env_guard": _contains(deepseek_v4, 'MILES_DSV4_TRACE_INTERNALS", "0") != "1"'),
        "has_detach_float_cpu_capture": _contains(deepseek_v4, "tensor.detach().float().cpu()"),
    }
    for key, passed in checks["attention_trace_gating"].items():
        _check(passed, failures, "attention_trace_gating." + key)

    old_kv_qat_pattern = "kv_vanilla = fp8_simulate_qat(kv_vanilla, 64)"
    checks["attention_kv_qat_non_rope_only"] = {
        "uses_torch_cat": _contains(deepseek_v4, "kv_vanilla = torch.cat("),
        "qats_non_rope_slice": _compact_contains(
            deepseek_v4,
            "fp8_simulate_qat(kv_vanilla[..., :-rd].contiguous(), 64)",
        ),
        "preserves_rope_slice": _compact_contains(deepseek_v4, "kv_vanilla[..., -rd:]"),
        "old_full_kv_qat_pattern_absent": old_kv_qat_pattern not in deepseek_v4,
    }
    for key, passed in checks["attention_kv_qat_non_rope_only"].items():
        _check(passed, failures, "attention_kv_qat_non_rope_only." + key)

    checks["act_quant_official_compatible"] = {
        "has_fe8m0_dtype": _contains(act_quant, 'FE8M0 = "float8_e8m0fnu"'),
        "kernel_supports_inplace": _contains(act_quant, "inplace=False"),
        "api_supports_scale_dtype": _contains(act_quant, "scale_dtype: torch.dtype = torch.float32"),
        "api_supports_inplace": _contains(act_quant, "inplace: bool = False"),
        "uses_requested_scale_dtype": _contains(act_quant, "dtype=scale_dtype"),
        "casts_scale_to_kernel_dtype": _contains(act_quant, "T.Cast(scale_dtype, s_local[i])"),
        "inplace_returns_input_dtype": _contains(act_quant, "y = torch.empty_like(x) if inplace else"),
        "inplace_copies_back": _contains(act_quant, "x.copy_(y)"),
        "inplace_returns_x": _compact_contains(act_quant, "if inplace:\n        x.copy_(y)\n        return x"),
    }
    for key, passed in checks["act_quant_official_compatible"].items():
        _check(passed, failures, "act_quant_official_compatible." + key)

    checks["qat_simulation_contract"] = {
        "uses_fp32_scales": _contains(qat, "scale_dtype=torch.float32"),
        "uses_inplace_act_quant": _contains(qat, "inplace=True"),
        "does_not_request_ue8m0_scales": '"ue8m0"' not in qat and "'ue8m0'" not in qat,
    }
    for key, passed in checks["qat_simulation_contract"].items():
        _check(passed, failures, "qat_simulation_contract." + key)

    old_compressor_overlap_pattern = "kv[..., : self.nope_head_dim] = fp8_simulate_qat"
    checks["compressor_qat_no_overlap_write"] = {
        "uses_torch_cat": _contains(compressor, "kv = torch.cat("),
        "qats_nope_slice_contiguous": _compact_contains(
            compressor,
            "fp8_simulate_qat(kv[..., : self.nope_head_dim].contiguous(), 64)",
        ),
        "preserves_rope_tail": _compact_contains(compressor, "kv[..., self.nope_head_dim :]"),
        "old_slice_assignment_absent": old_compressor_overlap_pattern not in compressor,
    }
    for key, passed in checks["compressor_qat_no_overlap_write"].items():
        _check(passed, failures, "compressor_qat_no_overlap_write." + key)

    linked_status = {}
    for name in [
        "deepseek-v4-operator-math-20260531.json",
        "deepseek-v4-attention-trace-replay-qatsim-0415-20260531.json",
        "deepseek-v4-proof-ledger-20260531.json",
    ]:
        path = args.artifacts_dir / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        status = payload.get("status") or payload.get("overall_status")
        linked_status[name] = status
        _check(status in {"PASS", "PASS_WITH_DRIFT_RECORDED"}, failures, "linked_artifact." + name)

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 source regression guards for precision fixes",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "linked_artifacts": linked_status,
        "failures": failures,
        "conclusion": (
            "Current source contains the precision fix shapes for non-RoPE KV QAT, "
            "official-compatible in-place act_quant with FP32 scales, no-overlap "
            "compressor QAT, and trace gating; linked numerical artifacts remain passing."
            if not failures
            else "One or more precision fix source guards failed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

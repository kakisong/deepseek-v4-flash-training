#!/usr/bin/env python3
"""校验 DeepSeek-V4 证明环境的来源信息（provenance）。"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(condition: bool, failures: List[str], name: str) -> None:
    if not condition:
        failures.append(name)


def _parse_doc_environment(doc: str) -> Dict[str, str]:
    lines = doc.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip() == "## 环境":
            start = idx
            break
    if start is None:
        return {}
    block_start = None
    for idx in range(start, len(lines)):
        if lines[idx].strip() == "```text":
            block_start = idx + 1
            break
    if block_start is None:
        return {}
    env = {}
    for idx in range(block_start, len(lines)):
        line = lines[idx].strip()
        if line == "```":
            break
        if ":" in line:
            key, value = line.split(":", 1)
            env[key.strip()] = value.strip()
    return env


def _cuda_toolkit_release(text: str) -> str:
    match = re.search(r"release\s+([0-9]+\.[0-9]+)", text)
    return match.group(1) if match else text


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("docs/en/advanced"))
    parser.add_argument("--doc", type=Path, default=Path("docs/en/advanced/deepseek-v4-hyperconnection-runtime.md"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    failures: List[str] = []
    operator = _load(args.artifacts_dir / "deepseek-v4-operator-math-20260531.json")
    env = operator["environment"]
    doc_text = args.doc.read_text(encoding="utf-8")
    doc_env = _parse_doc_environment(doc_text)

    expected_doc = {
        "GPU": env["gpu"],
        "CUDA device count": str(env["cuda_device_count"]),
        "NVIDIA driver": env["nvidia_driver"],
        "NVIDIA-SMI CUDA version": env["nvidia_smi_cuda"],
        "CUDA toolkit": _cuda_toolkit_release(env["cuda_toolkit"]),
        "Python": env["python"],
        "PyTorch": env["torch"],
        "torch CUDA": env["torch_cuda"],
        "megatron-core": env["megatron_core"],
        "mbridge": env["mbridge"],
        "miles package": env["miles"],
        "transformer-engine": env["transformer_engine"],
        "tilelang": env["tilelang"],
    }

    comparisons = {}
    for key, expected in expected_doc.items():
        actual = doc_env.get(key)
        comparisons[key] = {"expected": expected, "actual": actual, "matches": actual == expected}
        _check(actual == expected, failures, "doc_environment." + key)

    runtime_checks = {
        "upstream_hyper_connection_present=False": env["upstream_hyper_connection_present"] is False,
        "transformer_config_has_dsv4_mode=True": env["transformer_config_has_dsv4_mode"] is True,
        "transformer_config_has_experimental_attention_variant=True": env[
            "transformer_config_has_experimental_attention_variant"
        ]
        is True,
    }
    for phrase, expected_bool in runtime_checks.items():
        present = phrase in doc_text
        runtime_checks[phrase] = {"expected": expected_bool, "present_in_doc": present}
        _check(present == expected_bool, failures, "runtime_check." + phrase)

    linked = {}
    for name in [
        "deepseek-v4-operator-math-20260531.json",
        "deepseek-v4-proof-coverage-matrix-20260531.json",
        "deepseek-v4-proof-ledger-20260531.json",
    ]:
        payload = _load(args.artifacts_dir / name)
        status = payload.get("status") or payload.get("overall_status")
        linked[name] = status
        _check(status == "PASS", failures, "linked_artifact." + name)

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 proof environment provenance",
        "status": "PASS" if not failures else "FAIL",
        "source_artifact": "deepseek-v4-operator-math-20260531.json",
        "doc": args.doc.name,
        "environment_from_artifact": {
            "gpu": env["gpu"],
            "cuda_device_count": env["cuda_device_count"],
            "nvidia_driver": env["nvidia_driver"],
            "nvidia_smi_cuda": env["nvidia_smi_cuda"],
            "cuda_toolkit_release": _cuda_toolkit_release(env["cuda_toolkit"]),
            "python": env["python"],
            "torch": env["torch"],
            "torch_cuda": env["torch_cuda"],
            "megatron_core": env["megatron_core"],
            "mbridge": env["mbridge"],
            "miles": env["miles"],
            "transformer_engine": env["transformer_engine"],
            "tilelang": env["tilelang"],
            "upstream_hyper_connection_present": env["upstream_hyper_connection_present"],
            "transformer_config_has_dsv4_mode": env["transformer_config_has_dsv4_mode"],
            "transformer_config_has_experimental_attention_variant": env[
                "transformer_config_has_experimental_attention_variant"
            ],
        },
        "doc_environment_comparisons": comparisons,
        "runtime_structure_checks": runtime_checks,
        "linked_artifacts": linked,
        "failures": failures,
        "conclusion": (
            "The documented validation environment matches the environment recorded in the operator-math artifact, and linked proof artifacts remain passing."
            if not failures
            else "Environment provenance checks failed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

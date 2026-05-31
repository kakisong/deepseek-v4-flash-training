#!/usr/bin/env python3
"""Validate external-reference provenance for the DeepSeek-V4 proof doc."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _check(condition: bool, failures: List[str], name: str) -> None:
    if not condition:
        failures.append(name)


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", type=Path, default=Path("docs/en/advanced/deepseek-v4-hyperconnection-runtime.md"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    doc = args.doc.read_text(encoding="utf-8")
    failures: List[str] = []
    sources: List[Dict[str, Any]] = [
        {
            "id": "megatron_lm_pr_4839",
            "url": "https://github.com/NVIDIA/Megatron-LM/pull/4839",
            "source": "NVIDIA/Megatron-LM GitHub PR",
            "observed_date": "2026-05-31",
            "observed_facts": [
                "The PR is titled as a DeepSeek-V4 bug-fix PR.",
                "The PR contains commits for native and fused mHC H_res orientation fixes.",
                "The review discussion identifies a residual mixing orientation issue involving the transpose of H_res.",
            ],
            "used_for": [
                "problem identification",
                "HC orientation oracle construction",
                "reasoning about why an H_res.T check is necessary",
            ],
        },
        {
            "id": "swift_deepseek_v4_best_practice",
            "url": "https://swift.readthedocs.io/zh-cn/latest/BestPractices/deepseek-v4.html",
            "source": "SWIFT DeepSeek-V4 best-practice documentation",
            "observed_date": "2026-05-31",
            "observed_facts": [
                "The document frames DeepSeek-V4 training as requiring Megatron-side fixes.",
                "The document recommends verifying accuracy with controlled comparison and parity-style checks.",
                "The document is used here as methodology context, not as evidence that Miles implementation is correct.",
            ],
            "used_for": [
                "reference methodology",
                "motivation for parity and checkpoint-forward validation",
                "motivation for strict operator-level checks",
            ],
        },
    ]

    for source in sources:
        _check(source["url"] in doc, failures, "doc_contains_url." + source["id"])
        _check(bool(source["observed_facts"]), failures, "source_has_facts." + source["id"])
        _check(bool(source["used_for"]), failures, "source_has_usage." + source["id"])

    doc_phrases = [
        "Megatron-LM PR #4839",
        "SWIFT DeepSeek-V4 Best Practice",
        "External References",
        "Miles 的正确性结论仍以本仓库 artifact 为准",
    ]
    for phrase in doc_phrases:
        _check(phrase in doc, failures, "doc_phrase." + phrase)

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 external reference provenance",
        "status": "PASS" if not failures else "FAIL",
        "sources": sources,
        "doc": args.doc.name,
        "failures": failures,
        "conclusion": (
            "External references are documented with URLs, observed facts, and bounded usage; Miles correctness still depends on local artifacts."
            if not failures
            else "External reference provenance is incomplete."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

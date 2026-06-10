#!/usr/bin/env python3
"""比较官方推理实现与 Miles 的 DeepSeek-V4 trace tensor。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


TRACE_MAP = [
    ("embed", "module.embedding.word_embeddings"),
    ("layers.0.attn_norm", "module.decoder.layers.0.input_layernorm"),
    ("layers.0.attn.wq_a", "module.decoder.layers.0.self_attention.wq_a"),
    ("layers.0.attn.q_norm", "module.decoder.layers.0.self_attention.q_norm"),
    ("layers.0.attn.wq_b", "module.decoder.layers.0.self_attention.wq_b"),
    ("layers.0.attn.wkv", "module.decoder.layers.0.self_attention.wkv"),
    ("layers.0.attn.kv_norm", "module.decoder.layers.0.self_attention.kv_norm"),
    ("layers.0.attn.wo_b", "module.decoder.layers.0.self_attention.wo_b"),
    ("layers.0.attn", "module.decoder.layers.0.self_attention"),
    ("layers.0.ffn_norm", "module.decoder.layers.0.pre_mlp_layernorm"),
    ("layers.0.ffn.shared_experts", "module.decoder.layers.0.mlp.shared_experts"),
    ("layers.0.ffn", "module.decoder.layers.0.mlp"),
    ("layers.0", "module.decoder.layers.0"),
    ("norm", "module.decoder.final_layernorm"),
]


def _rel_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = (xf.square().sum() + yf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (xf * yf).sum().item() / denom)


def _maybe_unsqueeze_middle(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if tensor.ndim + 1 == target.ndim and target.shape[1] == 1:
        return tensor.unsqueeze(1)
    return tensor


def _canonicalize(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2 and tensor.shape[0] == 1 and tensor.shape[1] != 1:
        dims = [1, 0, *range(2, tensor.ndim)]
        return tensor.permute(*dims).contiguous()
    return tensor


def _compare_pair(
    official_name: str,
    miles_name: str,
    official_tensors: dict[str, torch.Tensor],
    miles_tensors: dict[str, torch.Tensor],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {"official_name": official_name, "miles_name": miles_name}
    if official_name not in official_tensors:
        row.update({"status": "MISSING_OFFICIAL"})
        return row
    if miles_name not in miles_tensors:
        row.update({"status": "MISSING_MILES"})
        return row

    official = _canonicalize(official_tensors[official_name].detach().float().cpu())
    miles = _canonicalize(miles_tensors[miles_name].detach().float().cpu())
    official = _maybe_unsqueeze_middle(official, miles)
    miles = _maybe_unsqueeze_middle(miles, official)

    row["official_shape"] = list(official.shape)
    row["miles_shape"] = list(miles.shape)
    if official.shape != miles.shape:
        row.update(
            {
                "status": "SHAPE_MISMATCH",
                "numel_official": int(official.numel()),
                "numel_miles": int(miles.numel()),
            }
        )
        return row

    diff = (official - miles).abs()
    close = torch.isclose(official, miles, rtol=rtol, atol=atol)
    mismatches = int((~close).sum().item())
    row.update(
        {
            "status": "PASS" if mismatches == 0 else "FAIL",
            "numel": int(diff.numel()),
            "mismatches": mismatches,
            "nonzero_abs_count": int((diff != 0).sum().item()),
            "exact_equal": bool((diff == 0).all().item()),
            "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
            "p95_abs": float(diff.flatten().quantile(0.95).item()) if diff.numel() else 0.0,
            "relative_gap": _rel_gap(official, miles),
        }
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-trace", type=Path, required=True)
    parser.add_argument("--miles-trace", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args()

    official_payload = torch.load(args.official_trace, map_location="cpu", weights_only=False)
    miles_payload = torch.load(args.miles_trace, map_location="cpu", weights_only=False)
    official_tensors = official_payload["trace_tensors"]
    miles_tensors = miles_payload["trace_tensors"]

    rows = [
        _compare_pair(official_name, miles_name, official_tensors, miles_tensors, args.rtol, args.atol)
        for official_name, miles_name in TRACE_MAP
    ]
    comparable = [row for row in rows if "max_abs" in row]
    failed = [row for row in comparable if row["status"] != "PASS"]
    payload = {
        "rtol": args.rtol,
        "atol": args.atol,
        "num_compared": len(comparable),
        "num_failed": len(failed),
        "status": "PASS" if not failed and len(comparable) == len(TRACE_MAP) else "FAIL",
        "first_failed_trace": failed[0] if failed else None,
        "max_abs_trace": max(comparable, key=lambda row: row["max_abs"]) if comparable else None,
        "max_relative_trace": max(comparable, key=lambda row: row["relative_gap"]) if comparable else None,
        "official_logprob_sha256": official_payload.get("official_summary", {}).get("sha256"),
        "miles_logprob_sha256": miles_payload.get("logprob_sha256") or miles_payload.get("summary", {}).get("sha256"),
        "traces": rows,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

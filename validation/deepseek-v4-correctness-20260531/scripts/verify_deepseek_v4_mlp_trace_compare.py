#!/usr/bin/env python3
"""Compare DeepSeek-V4 official and Miles layer-0 MLP trace tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _canonicalize(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach().float().cpu()
    if out.ndim >= 2 and out.shape[0] == 1 and out.shape[1] != 1:
        out = out.permute(1, 0, *range(2, out.ndim)).contiguous()
    if out.ndim == 2 and out.shape == (512, 4096):
        out = out.unsqueeze(1)
    return out


def _maybe_unsqueeze_middle(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if left.ndim + 1 == right.ndim and right.shape[1] == 1:
        left = left.unsqueeze(1)
    if right.ndim + 1 == left.ndim and left.shape[1] == 1:
        right = right.unsqueeze(1)
    return left, right


def _rel_gap(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().flatten().float()
    right = right.detach().flatten().float()
    denom = float((left.square().sum() + right.square().sum()).item())
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (left * right).sum().item() / denom)


def _compare(
    name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    mismatch_limit: int = 0,
) -> dict[str, Any]:
    left = _canonicalize(left)
    right = _canonicalize(right)
    left, right = _maybe_unsqueeze_middle(left, right)
    row: dict[str, Any] = {
        "name": name,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "rtol": rtol,
        "atol": atol,
        "mismatch_limit": mismatch_limit,
    }
    if left.shape != right.shape:
        row["status"] = "SHAPE_MISMATCH"
        return row
    diff = (left - right).abs()
    close = torch.isclose(left, right, rtol=rtol, atol=atol)
    mismatches = int((~close).sum().item())
    row.update(
        {
            "status": "PASS" if mismatches <= mismatch_limit else "FAIL",
            "numel": int(diff.numel()),
            "mismatches": mismatches,
            "nonzero_abs_count": int((diff != 0).sum().item()),
            "exact_equal": bool((diff == 0).all().item()),
            "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
            "p95_abs": float(diff.flatten().quantile(0.95).item()) if diff.numel() else 0.0,
            "p99_abs": float(diff.flatten().quantile(0.99).item()) if diff.numel() else 0.0,
            "relative_l2_gap": _rel_gap(left, right),
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
    official = official_payload["trace_tensors"]
    miles = miles_payload["trace_tensors"]

    official_indices = official["layers.0.ffn.gate.out1"].detach().cpu().long()
    miles_router_onehot = miles["module.decoder.layers.0.mlp.router.out1"].detach().cpu().float()
    router_onehot = torch.zeros_like(miles_router_onehot)
    router_onehot.scatter_(1, official_indices, 1.0)
    miles_router_scores = miles["module.decoder.layers.0.mlp.router.out0"].detach().cpu().float()
    miles_selected_weights = miles_router_scores.gather(1, official_indices)

    official_shared = _canonicalize(official["layers.0.ffn.shared_experts"])
    miles_shared = _canonicalize(miles["module.decoder.layers.0.mlp.shared_experts"])
    official_ffn = _canonicalize(official["layers.0.ffn"])
    miles_ffn = _canonicalize(miles["module.decoder.layers.0.mlp"])
    official_routed = official_ffn - official_shared
    miles_routed = miles_ffn - miles_shared

    comparisons = [
        _compare(
            "ffn_norm_official_vs_miles",
            official["layers.0.ffn_norm"],
            miles["module.decoder.layers.0.pre_mlp_layernorm"],
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "router_indices_onehot_official_vs_miles",
            router_onehot,
            miles_router_onehot,
            rtol=0.0,
            atol=0.0,
        ),
        _compare(
            "router_selected_weights_official_vs_miles",
            official["layers.0.ffn.gate.out0"],
            miles_selected_weights,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "shared_experts_official_vs_miles",
            official_shared,
            miles_shared,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "routed_experts_aggregated_official_vs_miles",
            official_routed,
            miles_routed,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "ffn_output_official_vs_miles",
            official_ffn,
            miles_ffn,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "final_layernorm_official_vs_miles",
            official["norm"],
            miles["module.decoder.final_layernorm"],
            rtol=args.rtol,
            atol=args.atol,
        ),
    ]
    exact_or_threshold_pass = {
        "router_indices_exact": comparisons[1]["status"] == "PASS",
        "router_weights_within_threshold": comparisons[2]["status"] == "PASS",
        "shared_experts_within_threshold": comparisons[3]["status"] == "PASS",
    }
    payload = {
        "status": "PASS_WITH_DRIFT_RECORDED" if all(exact_or_threshold_pass.values()) else "FAIL",
        "official_logprob_sha256": official_payload.get("official_summary", {}).get("sha256"),
        "miles_logprob_sha256": miles_payload.get("logprob_sha256") or miles_payload.get("summary", {}).get("sha256"),
        "router": {
            "num_tokens": int(official_indices.shape[0]),
            "topk": int(official_indices.shape[1]),
            "num_experts": int(miles_router_onehot.shape[1]),
        },
        "key_checks": exact_or_threshold_pass,
        "comparisons": comparisons,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())

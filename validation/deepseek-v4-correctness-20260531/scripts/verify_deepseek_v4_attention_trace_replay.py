#!/usr/bin/env python3
"""Replay DeepSeek-V4 attention from official and Miles trace tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _canonicalize(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach().float().cpu()
    if out.ndim >= 2 and out.shape[0] == 1 and out.shape[1] != 1:
        dims = [1, 0, *range(2, out.ndim)]
        out = out.permute(*dims).contiguous()
    return out


def _maybe_unsqueeze_middle(tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if tensor.ndim + 1 == target.ndim and target.shape[1] == 1:
        return tensor.unsqueeze(1)
    return tensor


def _to_bshd(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _canonicalize(tensor)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor.permute(1, 0, 2, 3).contiguous()
    if tensor.ndim == 4 and tensor.shape[0] == 1:
        return tensor.contiguous()
    raise ValueError(f"expected attention tensor layout, got {tuple(tensor.shape)}")


def _to_bsd(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _canonicalize(tensor)
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        return tensor.permute(1, 0, 2).contiguous()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor.contiguous()
    raise ValueError(f"expected sequence tensor layout, got {tuple(tensor.shape)}")


def _flatten_after_wo_a(tensor: torch.Tensor) -> torch.Tensor:
    tensor = _canonicalize(tensor)
    if tensor.ndim == 4:
        tensor = tensor.flatten(2)
    return tensor


def _rel_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = (xf.square().sum() + yf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (xf * yf).sum().item() / denom)


def _compare(
    name: str,
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    max_abs_limit: float | None = None,
    mean_abs_limit: float | None = None,
    rel_gap_limit: float | None = None,
    mismatch_limit: int | None = None,
) -> dict[str, Any]:
    left = _canonicalize(left)
    right = _canonicalize(right)
    left = _maybe_unsqueeze_middle(left, right)
    right = _maybe_unsqueeze_middle(right, left)
    row: dict[str, Any] = {
        "name": name,
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "rtol": rtol,
        "atol": atol,
    }
    if left.shape != right.shape:
        row.update({"status": "SHAPE_MISMATCH"})
        return row
    diff = (left.float() - right.float()).abs()
    close = torch.isclose(left.float(), right.float(), rtol=rtol, atol=atol)
    row.update(
        {
            "numel": int(diff.numel()),
            "mismatches": int((~close).sum().item()),
            "nonzero_abs_count": int((diff != 0).sum().item()),
            "exact_equal": bool((diff == 0).all().item()),
            "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
            "p95_abs": float(diff.flatten().quantile(0.95).item()) if diff.numel() else 0.0,
            "relative_gap": _rel_gap(left, right),
        }
    )
    failures = []
    if row["mismatches"] != 0:
        failures.append("torch_isclose")
    if max_abs_limit is not None and row["max_abs"] > max_abs_limit:
        failures.append("max_abs_limit")
    if mean_abs_limit is not None and row["mean_abs"] > mean_abs_limit:
        failures.append("mean_abs_limit")
    if rel_gap_limit is not None and row["relative_gap"] > rel_gap_limit:
        failures.append("relative_gap_limit")
    if mismatch_limit is not None and row["mismatches"] > mismatch_limit:
        failures.append("mismatch_limit")
    if mismatch_limit is not None and failures == ["torch_isclose"]:
        failures = []
    row["status"] = "PASS" if not failures else "FAIL"
    if failures:
        row["failed_limits"] = failures
    return row


def _expected_window_topk(batch: int, seqlen: int, window_size: int) -> torch.Tensor:
    base = torch.arange(seqlen).unsqueeze(1)
    k_pos = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
    topk = torch.where(k_pos > base, -1, k_pos)
    return topk.unsqueeze(0).expand(batch, -1, -1).to(torch.int32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-trace", type=Path, required=True)
    parser.add_argument("--miles-trace", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--softmax-scale", type=float, default=512**-0.5)
    args = parser.parse_args()

    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    official_payload = torch.load(args.official_trace, map_location="cpu", weights_only=False)
    miles_payload = torch.load(args.miles_trace, map_location="cpu", weights_only=False)
    official = official_payload["trace_tensors"]
    miles = miles_payload["trace_tensors"]

    comparisons = [
        _compare(
            "q_after_rope_official_vs_miles",
            official["layers.0.attn.q_after_rope"],
            miles["module.decoder.layers.0.self_attention.q_after_rope"],
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=6.25e-2,
            mean_abs_limit=1e-5,
            rel_gap_limit=1e-6,
            mismatch_limit=32,
        ),
        _compare(
            "kv_after_rope_qat_official_vs_miles",
            official["layers.0.attn.kv_after_rope_qat"],
            miles["module.decoder.layers.0.self_attention.kv_vanilla_after_rope_qat"],
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=4e-3,
            mean_abs_limit=1e-6,
            rel_gap_limit=1e-6,
            mismatch_limit=0,
        ),
        _compare(
            "attention_core_official_vs_miles",
            official["layers.0.attn.attention_core"],
            miles["module.decoder.layers.0.self_attention.attention_core"],
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=4e-2,
            mean_abs_limit=2e-4,
            rel_gap_limit=5e-6,
            mismatch_limit=512,
        ),
        _compare(
            "after_wo_a_official_vs_miles",
            official["layers.0.attn.wo_b.input"],
            _flatten_after_wo_a(miles["module.decoder.layers.0.self_attention.after_wo_a"]),
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=4e-2,
            mean_abs_limit=1e-3,
            rel_gap_limit=5e-6,
            mismatch_limit=1024,
        ),
        _compare(
            "after_wo_b_official_vs_miles",
            official["layers.0.attn.wo_b"],
            miles["module.decoder.layers.0.self_attention.after_wo_b"],
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=7e-2,
            mean_abs_limit=3e-3,
            rel_gap_limit=5e-6,
            mismatch_limit=4096,
        ),
    ]

    official_topk = official["layers.0.attn.topk_idxs"].detach().cpu().int().contiguous()
    expected_topk = _expected_window_topk(official_topk.shape[0], official_topk.shape[1], args.window_size)
    topk_exact = bool(torch.equal(official_topk, expected_topk))
    topk_row = {
        "name": "topk_window_indices_official_vs_expected",
        "status": "PASS" if topk_exact else "FAIL",
        "shape": list(official_topk.shape),
        "window_size": args.window_size,
        "exact_equal": topk_exact,
        "mismatches": int((official_topk != expected_topk).sum().item()),
    }

    from miles_plugins.models.deepseek_v4.ops.attention_core import dense_attn_torch, sparse_attn_torch

    attn_sink = official["layers.0.attn.attn_sink"].to(device=device, dtype=torch.float32)
    q_official = _to_bshd(official["layers.0.attn.q_after_rope"]).to(device=device, dtype=torch.bfloat16)
    kv_official = _to_bsd(official["layers.0.attn.kv_after_rope_qat"]).to(device=device, dtype=torch.bfloat16)
    q_miles = _to_bshd(miles["module.decoder.layers.0.self_attention.q_after_rope"]).to(device=device, dtype=torch.bfloat16)
    kv_miles = _to_bsd(miles["module.decoder.layers.0.self_attention.kv_vanilla_after_rope_qat"]).to(
        device=device,
        dtype=torch.bfloat16,
    )
    topk = official_topk.to(device=device)

    with torch.no_grad():
        dense_fp32_from_official = dense_attn_torch(
            q_official.float(), kv_official.float(), attn_sink, topk, args.softmax_scale
        )
        sparse_fp32_from_official = sparse_attn_torch(
            q_official.float(), kv_official.float(), attn_sink, topk, args.softmax_scale
        )
        dense_fp32_from_miles = dense_attn_torch(q_miles.float(), kv_miles.float(), attn_sink, topk, args.softmax_scale)
        sparse_fp32_from_miles = sparse_attn_torch(
            q_miles.float(), kv_miles.float(), attn_sink, topk, args.softmax_scale
        )
        dense_from_official = dense_attn_torch(q_official, kv_official, attn_sink, topk, args.softmax_scale)
        sparse_from_official = sparse_attn_torch(q_official, kv_official, attn_sink, topk, args.softmax_scale)
        dense_from_miles = dense_attn_torch(q_miles, kv_miles, attn_sink, topk, args.softmax_scale)
        sparse_from_miles = sparse_attn_torch(q_miles, kv_miles, attn_sink, topk, args.softmax_scale)

    replay_comparisons = [
        _compare(
            "dense_vs_sparse_fp32_math_replay_from_official_inputs",
            dense_fp32_from_official.cpu(),
            sparse_fp32_from_official.cpu(),
            rtol=1e-5,
            atol=1e-5,
            max_abs_limit=2e-6,
            mean_abs_limit=1e-8,
            rel_gap_limit=1e-8,
            mismatch_limit=0,
        ),
        _compare(
            "dense_vs_sparse_fp32_math_replay_from_miles_inputs",
            dense_fp32_from_miles.cpu(),
            sparse_fp32_from_miles.cpu(),
            rtol=1e-5,
            atol=1e-5,
            max_abs_limit=2e-6,
            mean_abs_limit=1e-8,
            rel_gap_limit=1e-8,
            mismatch_limit=0,
        ),
        _compare(
            "dense_vs_sparse_bf16_production_replay_from_official_inputs",
            dense_from_official.cpu(),
            sparse_from_official.cpu(),
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=6e-2,
            mean_abs_limit=5e-4,
            rel_gap_limit=6e-6,
            mismatch_limit=1200,
        ),
        _compare(
            "dense_vs_sparse_bf16_production_replay_from_miles_inputs",
            dense_from_miles.cpu(),
            sparse_from_miles.cpu(),
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=6e-2,
            mean_abs_limit=5e-4,
            rel_gap_limit=6e-6,
            mismatch_limit=1200,
        ),
        _compare(
            "official_attention_core_vs_sparse_replay",
            official["layers.0.attn.attention_core"],
            sparse_from_official.cpu(),
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=4e-2,
            mean_abs_limit=2e-4,
            rel_gap_limit=5e-6,
            mismatch_limit=1024,
        ),
        _compare(
            "miles_attention_core_vs_sparse_replay",
            miles["module.decoder.layers.0.self_attention.attention_core"],
            sparse_from_miles.cpu(),
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=2e-2,
            mean_abs_limit=1e-5,
            rel_gap_limit=1e-6,
            mismatch_limit=0,
        ),
        _compare(
            "sparse_replay_official_inputs_vs_miles_inputs",
            sparse_from_official.cpu(),
            sparse_from_miles.cpu(),
            rtol=2e-3,
            atol=2e-2,
            max_abs_limit=4e-2,
            mean_abs_limit=2e-4,
            rel_gap_limit=5e-6,
            mismatch_limit=512,
        ),
    ]

    rows = [*comparisons, topk_row, *replay_comparisons]
    failures = [row["name"] for row in rows if row["status"] != "PASS"]
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "device": str(device),
        "attention_dtype": "bfloat16",
        "softmax_scale": args.softmax_scale,
        "window_size": args.window_size,
        "official_logprob_sha256": official_payload.get("official_summary", {}).get("sha256"),
        "miles_logprob_sha256": miles_payload.get("logprob_sha256") or miles_payload.get("summary", {}).get("sha256"),
        "failures": failures,
        "comparisons": rows,
    }
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

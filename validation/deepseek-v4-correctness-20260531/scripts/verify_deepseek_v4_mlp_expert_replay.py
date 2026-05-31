#!/usr/bin/env python3
"""Replay DeepSeek-V4 layer-0 routed experts from official and Miles traces."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _load_official_helper(path: Path):
    spec = importlib.util.spec_from_file_location("official_full_forward_helper", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _canonicalize(tensor: torch.Tensor) -> torch.Tensor:
    out = tensor.detach().cpu()
    if out.ndim >= 2 and out.shape[0] == 1 and out.shape[1] != 1:
        out = out.permute(1, 0, *range(2, out.ndim)).contiguous()
    if out.ndim == 3 and out.shape[1] == 1:
        out = out[:, 0, :]
    return out.contiguous()


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
    left = _canonicalize(left).float()
    right = _canonicalize(right).float()
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


def _selected_weight_hash(weights: torch.Tensor, indices: torch.Tensor, max_experts: int) -> str:
    import hashlib

    h = hashlib.sha256()
    for expert_id in sorted(set(int(x) for x in indices.flatten().tolist()))[:max_experts]:
        expert_weight = weights[expert_id].detach().cpu().contiguous()
        h.update(str(tuple(expert_weight.shape)).encode("ascii"))
        h.update(expert_weight.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _replay_routed_experts(
    *,
    hidden: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    fc1_weight_cpu: torch.Tensor,
    fc2_weight_cpu: torch.Tensor,
    swiglu_limit: float,
    formula: str,
    device: torch.device,
) -> torch.Tensor:
    hidden = hidden.to(device=device, dtype=torch.bfloat16)
    indices = indices.to(device=device, dtype=torch.long)
    weights = weights.to(device=device, dtype=torch.float32)
    out = torch.zeros(hidden.shape[0], fc2_weight_cpu.shape[1], device=device, dtype=torch.float32)
    expert_ids = sorted(set(int(x) for x in indices.detach().cpu().flatten().tolist()))

    for expert_id in expert_ids:
        token_idx, top_idx = torch.where(indices == expert_id)
        if token_idx.numel() == 0:
            continue
        x = hidden[token_idx]
        fc1 = fc1_weight_cpu[expert_id].to(device=device, dtype=torch.bfloat16, non_blocking=True)
        fc2 = fc2_weight_cpu[expert_id].to(device=device, dtype=torch.bfloat16, non_blocking=True)

        if formula == "official_fp32_activation":
            gate_weight, up_weight = fc1.chunk(2, dim=0)
            gate = F.linear(x, gate_weight).float()
            up = F.linear(x, up_weight).float()
            if swiglu_limit > 0:
                up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
                gate = torch.clamp(gate, max=swiglu_limit)
            intermediate = F.silu(gate) * up
            intermediate = weights[token_idx, top_idx, None] * intermediate
            expert_out = F.linear(intermediate.to(torch.bfloat16), fc2).float()
        elif formula == "megatron_bf16_activation":
            fc1_out = F.linear(x, fc1)
            gate, up = fc1_out.chunk(2, dim=-1)
            if swiglu_limit > 0:
                up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
                gate = torch.clamp(gate, max=swiglu_limit)
            intermediate = F.silu(gate) * up
            intermediate = intermediate * weights[token_idx, top_idx, None].to(intermediate.dtype)
            intermediate = intermediate.to(torch.bfloat16)
            expert_out = F.linear(intermediate, fc2).float()
        else:
            raise ValueError(f"unknown formula: {formula}")

        out.index_add_(0, token_idx, expert_out)
        del fc1, fc2, expert_out

    return out.detach().cpu()


def _combine_routed_and_shared(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    return (routed.float() + _canonicalize(shared).float()).to(torch.bfloat16).float()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-trace", type=Path, required=True)
    parser.add_argument("--miles-trace", type=Path, required=True)
    parser.add_argument("--checkpoint-release-dir", type=Path, required=True)
    parser.add_argument("--official-helper", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    official_payload = torch.load(args.official_trace, map_location="cpu", weights_only=False)
    miles_payload = torch.load(args.miles_trace, map_location="cpu", weights_only=False)
    official = official_payload["trace_tensors"]
    miles = miles_payload["trace_tensors"]

    official_indices = official["layers.0.ffn.gate.out1"].detach().cpu().long()
    official_weights = official["layers.0.ffn.gate.out0"].detach().cpu().float()
    miles_router_scores = miles["module.decoder.layers.0.mlp.router.out0"].detach().cpu().float()
    miles_weights = miles_router_scores.gather(1, official_indices)

    official_hidden = _canonicalize(official["layers.0.ffn_norm"])
    miles_hidden = _canonicalize(miles["module.decoder.layers.0.pre_mlp_layernorm"])
    official_ffn = _canonicalize(official["layers.0.ffn"])
    miles_ffn = _canonicalize(miles["module.decoder.layers.0.mlp"])
    official_shared = _canonicalize(official["layers.0.ffn.shared_experts"])
    miles_shared = _canonicalize(miles["module.decoder.layers.0.mlp.shared_experts"])
    official_routed_derived = official_ffn.float() - official_shared.float()
    miles_routed_derived = miles_ffn.float() - miles_shared.float()

    helper = _load_official_helper(args.official_helper)
    raw_state = helper._load_megatron_state(args.checkpoint_release_dir, num_layers=1)
    fc1_weight = raw_state["decoder.layers.0.mlp.experts.experts.linear_fc1.weight"]
    fc2_weight = raw_state["decoder.layers.0.mlp.experts.experts.linear_fc2.weight"]

    official_replay = _replay_routed_experts(
        hidden=official_hidden,
        indices=official_indices,
        weights=official_weights,
        fc1_weight_cpu=fc1_weight,
        fc2_weight_cpu=fc2_weight,
        swiglu_limit=args.swiglu_limit,
        formula="official_fp32_activation",
        device=device,
    )
    miles_official_formula_replay = _replay_routed_experts(
        hidden=miles_hidden,
        indices=official_indices,
        weights=miles_weights,
        fc1_weight_cpu=fc1_weight,
        fc2_weight_cpu=fc2_weight,
        swiglu_limit=args.swiglu_limit,
        formula="official_fp32_activation",
        device=device,
    )
    miles_megatron_formula_replay = _replay_routed_experts(
        hidden=miles_hidden,
        indices=official_indices,
        weights=miles_weights,
        fc1_weight_cpu=fc1_weight,
        fc2_weight_cpu=fc2_weight,
        swiglu_limit=args.swiglu_limit,
        formula="megatron_bf16_activation",
        device=device,
    )
    official_total_replay = _combine_routed_and_shared(official_replay, official_shared)
    miles_official_formula_total_replay = _combine_routed_and_shared(miles_official_formula_replay, miles_shared)
    miles_megatron_formula_total_replay = _combine_routed_and_shared(miles_megatron_formula_replay, miles_shared)

    comparisons = [
        _compare(
            "official_formula_total_replay_vs_official_ffn",
            official_total_replay,
            official_ffn,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "miles_official_formula_total_replay_vs_miles_ffn",
            miles_official_formula_total_replay,
            miles_ffn,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "miles_megatron_bf16_formula_total_replay_vs_miles_ffn",
            miles_megatron_formula_total_replay,
            miles_ffn,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "official_formula_replay_vs_official_routed_derived",
            official_replay,
            official_routed_derived,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "miles_official_formula_replay_vs_miles_routed_derived",
            miles_official_formula_replay,
            miles_routed_derived,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "miles_megatron_bf16_formula_replay_vs_miles_routed_derived",
            miles_megatron_formula_replay,
            miles_routed_derived,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "miles_official_formula_replay_vs_official_formula_replay",
            miles_official_formula_replay,
            official_replay,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "miles_megatron_bf16_formula_replay_vs_miles_official_formula_replay",
            miles_megatron_formula_replay,
            miles_official_formula_replay,
            rtol=args.rtol,
            atol=args.atol,
        ),
        _compare(
            "official_routed_derived_vs_miles_routed_derived",
            official_routed_derived,
            miles_routed_derived,
            rtol=args.rtol,
            atol=args.atol,
        ),
    ]
    key_checks = {
        "official_formula_total_replay_matches_official_ffn": comparisons[0]["status"] == "PASS",
        "megatron_bf16_formula_is_closer_to_miles_than_official_formula": (
            comparisons[2].get("mean_abs", float("inf")) <= comparisons[1].get("mean_abs", float("inf"))
            and comparisons[2].get("relative_l2_gap", float("inf")) <= comparisons[1].get("relative_l2_gap", float("inf"))
        ),
        "official_vs_miles_uses_same_expert_indices": bool(
            torch.equal(
                torch.zeros_like(miles["module.decoder.layers.0.mlp.router.out1"]).scatter(1, official_indices, 1.0),
                miles["module.decoder.layers.0.mlp.router.out1"].detach().cpu().float(),
            )
        ),
    }
    payload = {
        "status": (
            "PASS_WITH_DRIFT_RECORDED"
            if key_checks["official_formula_total_replay_matches_official_ffn"]
            and key_checks["official_vs_miles_uses_same_expert_indices"]
            else "FAIL"
        ),
        "swiglu_limit": args.swiglu_limit,
        "router": {
            "num_tokens": int(official_indices.shape[0]),
            "topk": int(official_indices.shape[1]),
            "num_unique_experts": int(torch.unique(official_indices).numel()),
        },
        "weights": {
            "fc1_shape": list(fc1_weight.shape),
            "fc2_shape": list(fc2_weight.shape),
            "selected_expert_weight_sha256_first64": _selected_weight_hash(fc1_weight, official_indices, max_experts=64),
        },
        "key_checks": key_checks,
        "comparisons": comparisons,
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())

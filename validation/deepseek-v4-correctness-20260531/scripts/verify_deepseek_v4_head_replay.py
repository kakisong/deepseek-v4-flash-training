#!/usr/bin/env python3
"""Replay DeepSeek-V4 output head from official and Miles final hidden traces."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


SEED = 20260531


def _load_official_verifier(script: Path):
    spec = importlib.util.spec_from_file_location("deepseek_v4_official_full_forward_replay", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tensor_sha256(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().float()
    h = hashlib.sha256()
    h.update(str(tuple(data.shape)).encode("ascii"))
    h.update(data.numpy().tobytes())
    return h.hexdigest()


def _compare(name: str, left: torch.Tensor, right: torch.Tensor, *, rtol: float, atol: float) -> dict[str, Any]:
    left = left.detach().cpu().float()
    right = right.detach().cpu().float()
    diff = (left - right).abs()
    close = torch.isclose(left, right, rtol=rtol, atol=atol)
    denom = float((left.square().sum() + right.square().sum()).item())
    rel = 0.0 if denom == 0.0 else float(1.0 - 2.0 * (left * right).sum().item() / denom)
    return {
        "name": name,
        "status": "PASS" if bool(close.all().item()) else "FAIL",
        "shape": list(left.shape),
        "rtol": rtol,
        "atol": atol,
        "mismatches": int((~close).sum().item()),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "p95_abs": float(diff.quantile(0.95).item()) if diff.numel() else 0.0,
        "p99_abs": float(diff.quantile(0.99).item()) if diff.numel() else 0.0,
        "relative_l2_gap": rel,
    }


def _summary(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    x = tensor.detach().cpu().float()
    return {
        "name": name,
        "shape": list(x.shape),
        "mean": float(x.mean().item()) if x.numel() else 0.0,
        "min": float(x.min().item()) if x.numel() else 0.0,
        "max": float(x.max().item()) if x.numel() else 0.0,
        "sha256": _tensor_sha256(x),
    }


def _canonical_seq_hidden(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.ndim == 3 and tensor.shape[1] == 1:
        return tensor[:, 0].contiguous()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor[0].contiguous()
    if tensor.ndim == 2:
        return tensor.contiguous()
    raise ValueError(f"unexpected final hidden shape: {tuple(tensor.shape)}")


def _replay_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    start: int,
    total_length: int,
    mode: str,
) -> torch.Tensor:
    h = hidden[start - 1 : total_length - 1]
    if mode == "fp32_head":
        logits = F.linear(h.float(), weight.float())
    elif mode == "bf16_head":
        logits = F.linear(h.bfloat16(), weight.bfloat16()).float()
    else:
        raise ValueError(f"unsupported replay mode: {mode}")
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, target_tokens.unsqueeze(-1)).squeeze(-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-trace", type=Path, required=True)
    parser.add_argument("--miles-trace", type=Path, required=True)
    parser.add_argument("--miles-output", type=Path, required=True)
    parser.add_argument("--checkpoint-release-dir", type=Path, required=True)
    parser.add_argument("--rollout-data", type=Path, required=True)
    parser.add_argument("--official-full-forward-script", type=Path, default=Path("tools/verify_deepseek_v4_official_full_forward.py"))
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-2)
    args = parser.parse_args()

    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    official_mod = _load_official_verifier(args.official_full_forward_script)
    raw_state = official_mod._load_megatron_state(args.checkpoint_release_dir, num_layers=1)
    weight = raw_state["output_layer.weight"].to(device=device)

    rollout = official_mod._load_rollout_data(args.rollout_data, max_samples=1)
    tokens = rollout["tokens"][0][: rollout["total_lengths"][0]].long().to(device=device)
    total_length = int(rollout["total_lengths"][0])
    response_length = int(rollout["response_lengths"][0])
    start = total_length - response_length
    target_tokens = tokens[start:total_length]

    official_payload = torch.load(args.official_trace, map_location="cpu", weights_only=False)
    miles_trace_payload = torch.load(args.miles_trace, map_location="cpu", weights_only=False)
    miles_output_payload = torch.load(args.miles_output, map_location="cpu", weights_only=False)

    official_hidden = _canonical_seq_hidden(official_payload["trace_tensors"]["norm"]).to(device=device)
    miles_hidden = _canonical_seq_hidden(miles_trace_payload["trace_tensors"]["module.decoder.final_layernorm"]).to(
        device=device
    )
    official_saved = official_payload["log_probs"][0].to(device=device).float()
    miles_saved = miles_output_payload["log_probs"][0].to(device=device).float()

    official_fp32 = _replay_logprobs(
        official_hidden,
        weight,
        target_tokens,
        start=start,
        total_length=total_length,
        mode="fp32_head",
    )
    official_bf16 = _replay_logprobs(
        official_hidden,
        weight,
        target_tokens,
        start=start,
        total_length=total_length,
        mode="bf16_head",
    )
    miles_fp32 = _replay_logprobs(
        miles_hidden,
        weight,
        target_tokens,
        start=start,
        total_length=total_length,
        mode="fp32_head",
    )
    miles_bf16 = _replay_logprobs(
        miles_hidden,
        weight,
        target_tokens,
        start=start,
        total_length=total_length,
        mode="bf16_head",
    )

    comparisons = [
        _compare("official_saved_vs_official_hidden_fp32_head_replay", official_saved, official_fp32, rtol=args.rtol, atol=args.atol),
        _compare("miles_saved_vs_miles_hidden_bf16_head_replay", miles_saved, miles_bf16, rtol=args.rtol, atol=args.atol),
        _compare("official_saved_vs_miles_saved", official_saved, miles_saved, rtol=args.rtol, atol=args.atol),
        _compare("same_fp32_head_official_hidden_vs_miles_hidden", official_fp32, miles_fp32, rtol=args.rtol, atol=args.atol),
        _compare("same_bf16_head_official_hidden_vs_miles_hidden", official_bf16, miles_bf16, rtol=args.rtol, atol=args.atol),
        _compare("official_hidden_fp32_head_vs_bf16_head", official_fp32, official_bf16, rtol=args.rtol, atol=args.atol),
        _compare("miles_hidden_fp32_head_vs_bf16_head", miles_fp32, miles_bf16, rtol=args.rtol, atol=args.atol),
    ]
    replay_exact = {
        "official_saved_replayed_by_fp32_head": comparisons[0]["mismatches"] == 0,
        "miles_saved_replayed_by_bf16_head": comparisons[1]["mismatches"] == 0,
    }
    payload = {
        "seed": SEED,
        "status": "PASS_WITH_DRIFT_RECORDED" if all(replay_exact.values()) else "FAIL",
        "rollout": {
            "num_samples": 1,
            "num_tokens": int(response_length),
            "total_length": int(total_length),
        },
        "output_layer_weight": {
            "shape": list(weight.shape),
            "checkpoint_dtype": str(raw_state["output_layer.weight"].dtype).replace("torch.", ""),
        },
        "summaries": [
            _summary("official_saved", official_saved.cpu()),
            _summary("miles_saved", miles_saved.cpu()),
            _summary("official_hidden_fp32_head_replay", official_fp32.cpu()),
            _summary("official_hidden_bf16_head_replay", official_bf16.cpu()),
            _summary("miles_hidden_fp32_head_replay", miles_fp32.cpu()),
            _summary("miles_hidden_bf16_head_replay", miles_bf16.cpu()),
        ],
        "replay_exact": replay_exact,
        "comparisons": comparisons,
    }

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())

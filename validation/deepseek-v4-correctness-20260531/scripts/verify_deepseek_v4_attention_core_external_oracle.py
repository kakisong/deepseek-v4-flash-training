#!/usr/bin/env python3
"""DeepSeek-V4 第 0 层 attention core 的外部 PyTorch oracle。

本校验器读取已记录的 mini-checkpoint attention trace,将 dense/sparse/tilelang
运行时的 attention_core tensor 与一份独立实现的 PyTorch 掩码 attention 公式
进行对比。它有意不导入 Miles 的 attention_core 算子。
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


TRACE_PREFIX = "module.decoder.layers.{layer}.self_attention"


def _status(payload: Dict[str, Any]) -> str:
    return payload.get("status") or payload.get("overall_status") or ""


def _load_trace(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "trace_tensors" not in payload:
        raise KeyError(f"trace_tensors missing in {path}")
    return payload


def _resolve_iter_dir(checkpoint_dir: Path) -> Path:
    if checkpoint_dir.name.startswith("iter_"):
        return checkpoint_dir
    latest = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if latest.exists():
        raw = latest.read_text(encoding="utf-8").strip()
        if raw.isdigit():
            return checkpoint_dir / f"iter_{int(raw):07d}"
        if raw.startswith("iter_"):
            return checkpoint_dir / raw
    candidates = sorted(checkpoint_dir.glob("iter_*"))
    if not candidates:
        raise FileNotFoundError(f"no iter_* checkpoint found under {checkpoint_dir}")
    return candidates[-1]


def _load_attn_sink(checkpoint_dir: Path, layer: int) -> Tuple[torch.Tensor, str]:
    iter_dir = _resolve_iter_dir(checkpoint_dir)
    key = f"decoder.layers.{layer}.self_attention.attn_sink"
    state = {key: torch.empty(64, dtype=torch.float32)}
    import torch.distributed.checkpoint as dcp

    dcp.load_state_dict(state, storage_reader=dcp.FileSystemReader(str(iter_dir)), no_dist=True)
    return state[key].detach().cpu(), iter_dir.name


def _window_topk(batch: int, seqlen: int, window_size: int) -> torch.Tensor:
    positions = torch.arange(seqlen, dtype=torch.long)
    offsets = torch.arange(window_size, dtype=torch.long)
    first = (positions - window_size + 1).clamp(min=0)
    idx = first[:, None] + offsets[None, :]
    idx = torch.where(idx <= positions[:, None], idx, torch.full_like(idx, -1))
    return idx.unsqueeze(0).expand(batch, -1, -1).contiguous().to(torch.int32)


def _relative_l2_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.detach().flatten().float()
    bf = b.detach().flatten().float()
    denom = (af.square().sum() + bf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (af * bf).sum().item() / denom)


def _compare(name: str, actual: torch.Tensor, expected: torch.Tensor, thresholds: Dict[str, float]) -> Dict[str, Any]:
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    row: Dict[str, Any] = {
        "name": name,
        "actual_shape": list(actual.shape),
        "expected_shape": list(expected.shape),
        "thresholds": thresholds,
    }
    if actual.shape != expected.shape:
        row.update({"status": "FAIL", "failure": "shape_mismatch"})
        return row
    diff = (actual - expected).abs()
    row.update(
        {
            "numel": int(diff.numel()),
            "finite": bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item()),
            "exact_equal": bool(torch.equal(actual, expected)),
            "nonzero_abs_count": int((diff != 0).sum().item()),
            "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
            "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
            "p95_abs": float(diff.flatten().quantile(0.95).item()) if diff.numel() else 0.0,
            "p99_abs": float(diff.flatten().quantile(0.99).item()) if diff.numel() else 0.0,
            "relative_l2_gap": _relative_l2_gap(actual, expected),
        }
    )
    failures = []
    if not row["finite"]:
        failures.append("nonfinite")
    for key, limit in thresholds.items():
        if row[key] > limit:
            failures.append(key)
    row["status"] = "PASS" if not failures else "FAIL"
    if failures:
        row["failures"] = failures
    return row


def _exact_compare(name: str, left: torch.Tensor, right: torch.Tensor) -> Dict[str, Any]:
    diff = (left.detach().float() - right.detach().float()).abs()
    return {
        "name": name,
        "shape": list(left.shape),
        "exact_equal": bool(torch.equal(left.detach().float(), right.detach().float())),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "nonzero_abs_count": int((diff != 0).sum().item()),
    }


def _attention_oracle(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    chunk_size: int,
    softmax_scale: float,
    output_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    qd = q.to(device=device, dtype=dtype)
    kvd = kv.to(device=device, dtype=dtype)
    sink = attn_sink.to(device=device, dtype=dtype)
    topk = topk.to(device=device)
    batch, seqlen, heads, _ = qd.shape
    kv_len = kvd.shape[1]
    out_chunks = []
    for start in range(0, seqlen, chunk_size):
        end = min(start + chunk_size, seqlen)
        q_chunk = qd[:, start:end]
        topk_chunk = topk[:, start:end].long()
        valid = topk_chunk >= 0
        safe_topk = topk_chunk.clamp(min=0)
        attn_mask = torch.zeros((batch, end - start, kv_len), device=device, dtype=torch.bool)
        batch_idx = torch.arange(batch, device=device).view(batch, 1, 1).expand_as(safe_topk)
        seq_idx = torch.arange(end - start, device=device).view(1, end - start, 1).expand_as(safe_topk)
        attn_mask[batch_idx[valid], seq_idx[valid], safe_topk[valid]] = True

        scores = torch.einsum("bchd,bnd->bchn", q_chunk, kvd) * softmax_scale
        scores = scores.masked_fill(~attn_mask.unsqueeze(2), float("-inf"))
        scores_max = scores.max(dim=-1, keepdim=True).values
        scores_max = torch.maximum(scores_max, sink.view(1, 1, heads, 1)).clamp(min=-1e30)
        exp_scores = torch.exp(scores - scores_max)
        numerator = torch.einsum("bchn,bnd->bchd", exp_scores, kvd)
        sum_exp = exp_scores.sum(dim=-1)
        sink_term = torch.exp(sink.view(1, 1, heads) - scores_max.squeeze(-1))
        chunk_out = numerator / (sum_exp + sink_term).unsqueeze(-1)
        if output_dtype is not None:
            chunk_out = chunk_out.to(output_dtype)
        out_chunks.append(chunk_out.detach().cpu())
    return torch.cat(out_chunks, dim=1)


def _trace_tensor(payload: Dict[str, Any], layer: int, suffix: str) -> torch.Tensor:
    key = f"{TRACE_PREFIX.format(layer=layer)}.{suffix}"
    return payload["trace_tensors"][key].detach().cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-trace", type=Path, required=True)
    parser.add_argument("--sparse-trace", type=Path, required=True)
    parser.add_argument("--tilelang-trace", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--softmax-scale", type=float, default=512**-0.5)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dense = _load_trace(args.dense_trace)
    sparse = _load_trace(args.sparse_trace)
    tilelang = _load_trace(args.tilelang_trace)
    attn_sink, checkpoint_iteration = _load_attn_sink(args.checkpoint_dir, args.layer)

    q_dense = _trace_tensor(dense, args.layer, "q_after_rope")
    kv_dense = _trace_tensor(dense, args.layer, "kv_vanilla_after_rope_qat")
    q_sparse = _trace_tensor(sparse, args.layer, "q_after_rope")
    kv_sparse = _trace_tensor(sparse, args.layer, "kv_vanilla_after_rope_qat")
    q_tilelang = _trace_tensor(tilelang, args.layer, "q_after_rope")
    kv_tilelang = _trace_tensor(tilelang, args.layer, "kv_vanilla_after_rope_qat")
    core_dense = _trace_tensor(dense, args.layer, "attention_core")
    core_sparse = _trace_tensor(sparse, args.layer, "attention_core")
    core_tilelang = _trace_tensor(tilelang, args.layer, "attention_core")

    if q_dense.ndim != 4 or kv_dense.ndim != 3:
        raise ValueError(f"unexpected q/kv shapes: {tuple(q_dense.shape)}, {tuple(kv_dense.shape)}")
    topk = _window_topk(q_dense.shape[0], q_dense.shape[1], args.window_size)

    oracle_fp64 = _attention_oracle(
        q_dense,
        kv_dense,
        attn_sink,
        topk,
        dtype=torch.float64,
        device=device,
        chunk_size=args.chunk_size,
        softmax_scale=args.softmax_scale,
        output_dtype=None,
    )
    oracle_fp64_bf16 = oracle_fp64.to(torch.bfloat16).float()
    oracle_fp32 = _attention_oracle(
        q_dense,
        kv_dense,
        attn_sink,
        topk,
        dtype=torch.float32,
        device=device,
        chunk_size=args.chunk_size,
        softmax_scale=args.softmax_scale,
        output_dtype=torch.bfloat16,
    ).float()

    qkv_checks = [
        _exact_compare("q_dense_vs_sparse", q_dense, q_sparse),
        _exact_compare("q_dense_vs_tilelang", q_dense, q_tilelang),
        _exact_compare("kv_dense_vs_sparse", kv_dense, kv_sparse),
        _exact_compare("kv_dense_vs_tilelang", kv_dense, kv_tilelang),
    ]

    fp64_thresholds = {
        "max_abs": 0.065,
        "mean_abs": 0.001,
        "p99_abs": 0.008,
        "relative_l2_gap": 8e-6,
    }
    bf16_thresholds = {
        "max_abs": 0.065,
        "mean_abs": 0.001,
        "p99_abs": 0.008,
        "relative_l2_gap": 8e-6,
    }
    fp32_thresholds = {
        "max_abs": 0.065,
        "mean_abs": 0.001,
        "p99_abs": 0.008,
        "relative_l2_gap": 8e-6,
    }
    comparisons = [
        _compare("dense_core_vs_external_fp64", core_dense, oracle_fp64, fp64_thresholds),
        _compare("sparse_core_vs_external_fp64", core_sparse, oracle_fp64, fp64_thresholds),
        _compare("tilelang_core_vs_external_fp64", core_tilelang, oracle_fp64, fp64_thresholds),
        _compare("dense_core_vs_external_fp64_rounded_bf16", core_dense, oracle_fp64_bf16, bf16_thresholds),
        _compare("sparse_core_vs_external_fp64_rounded_bf16", core_sparse, oracle_fp64_bf16, bf16_thresholds),
        _compare("tilelang_core_vs_external_fp64_rounded_bf16", core_tilelang, oracle_fp64_bf16, bf16_thresholds),
        _compare("dense_core_vs_external_fp32_bf16_formula", core_dense, oracle_fp32, fp32_thresholds),
        _compare("sparse_core_vs_external_fp32_bf16_formula", core_sparse, oracle_fp32, fp32_thresholds),
        _compare("tilelang_core_vs_external_fp32_bf16_formula", core_tilelang, oracle_fp32, fp32_thresholds),
    ]

    failures = []
    for row in qkv_checks:
        if not row["exact_equal"]:
            failures.append(row["name"])
    for row in comparisons:
        if row["status"] != "PASS":
            failures.append(row["name"])

    payload = {
        "date": "2026-06-01",
        "scope": "DeepSeek-V4 mini checkpoint layer-0 attention-core external oracle",
        "status": "PASS" if not failures else "FAIL",
        "method": (
            "Load recorded layer-0 q_after_rope, kv_vanilla_after_rope_qat, and backend "
            "attention_core tensors; load attn_sink directly from the distributed checkpoint; "
            "reconstruct the sliding-window top-k mask; compare dense/sparse/tilelang outputs "
            "against an in-script PyTorch FP64/FP32 masked-attention formula."
        ),
        "boundary": (
            "This is an external formula oracle for layer-0 attention_core. It validates that "
            "all runtime backends stay inside the declared BF16 attention-core envelope; it does "
            "not make full-model strict logprob parity pass."
        ),
        "runtime": {
            "device": str(device),
            "chunk_size": args.chunk_size,
            "softmax_scale": args.softmax_scale,
            "window_size": args.window_size,
            "checkpoint_iteration": checkpoint_iteration,
        },
        "inputs": {
            "dense_trace_name": args.dense_trace.name,
            "sparse_trace_name": args.sparse_trace.name,
            "tilelang_trace_name": args.tilelang_trace.name,
            "checkpoint_name": args.checkpoint_dir.name,
            "layer": args.layer,
            "q_shape": list(q_dense.shape),
            "kv_shape": list(kv_dense.shape),
            "attn_sink_shape": list(attn_sink.shape),
        },
        "qkv_exact_checks": qkv_checks,
        "comparisons": comparisons,
        "failures": failures,
        "conclusion": (
            "Layer-0 Q/KV inputs are exact across dense/sparse/tilelang, and all three "
            "backend attention_core tensors are within the external PyTorch FP64/FP32 "
            "BF16 envelope. The remaining dense/sparse/tilelang strict differences are "
            "therefore backend numerical choices around the same mathematical attention "
            "formula, not a different Q/KV/RoPE/QAT input or a formula-level error."
            if not failures
            else "At least one backend is outside the external attention-core oracle envelope."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

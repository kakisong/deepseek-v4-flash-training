#!/usr/bin/env python3
"""DeepSeek-V4 operator-level precision checks.

This script is intentionally small and deterministic. It checks the math that is
hard to prove from a training loss curve alone:

* HyperConnection post-mix uses H_res.T @ residual, matching Megatron-LM PR #4839.
* RoPE returns a new tensor, preserves the input, and inverse rotation recovers it.
* Dense and sparse PyTorch attention references agree, preserve dtype, and keep
  fully masked rows finite in forward/backward.
* TileLang sparse MLA agrees with the dense PyTorch reference when CUDA+TileLang
  are available.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import os
import platform
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


SEED = 20260531


@dataclass
class CheckResult:
    name: str
    status: str
    details: dict[str, Any]


def _run(cmd: list[str], cwd: str | None = None) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _nvidia_smi_cuda_version() -> str | None:
    output = _run(["nvidia-smi"])
    if output is None:
        return None
    match = re.search(r"CUDA Version:\s*([0-9.]+)", output)
    return match.group(1) if match else None


def collect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "megatron_core": _pkg_version("megatron-core"),
        "mbridge": _pkg_version("mbridge"),
        "miles": _pkg_version("miles"),
        "transformer_engine": _pkg_version("transformer-engine"),
        "tilelang": _pkg_version("tilelang"),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["cuda_device_count"] = torch.cuda.device_count()
        env["nvidia_driver"] = _run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"]
        )
        env["nvidia_smi_cuda"] = _nvidia_smi_cuda_version()
    env["cuda_toolkit"] = _run(["nvcc", "--version"])
    git_root = _run(["git", "rev-parse", "--show-toplevel"])
    if git_root:
        env["git_commit"] = _run(["git", "rev-parse", "HEAD"], cwd=git_root)
        env["git_branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=git_root)
    try:
        from megatron.core.transformer.transformer_config import TransformerConfig

        fields = getattr(TransformerConfig, "__dataclass_fields__", {})
        env["transformer_config_has_dsv4_mode"] = "dsv4_mode" in fields
        env["transformer_config_has_experimental_attention_variant"] = "experimental_attention_variant" in fields
    except Exception as exc:
        env["transformer_config_probe_error"] = repr(exc)
    try:
        import megatron.core.transformer.hyper_connection as upstream_hc  # noqa: F401

        env["upstream_hyper_connection_present"] = True
    except Exception:
        env["upstream_hyper_connection_present"] = False
    return env


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} has non-finite values"


def rel_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = (xf.square().sum() + yf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (xf * yf).sum().item() / denom)


def make_topk(batch: int, seqlen: int, kv_len: int, topk: int, *, device: torch.device) -> torch.Tensor:
    rows = []
    actual_topk = min(kv_len, topk)
    for _ in range(batch):
        batch_rows = []
        for _ in range(seqlen):
            idx = torch.randperm(kv_len, device=device)[:actual_topk]
            if actual_topk < topk:
                pad = torch.full((topk - actual_topk,), -1, device=device, dtype=torch.long)
                idx = torch.cat([idx, pad], dim=0)
            batch_rows.append(idx)
        rows.append(torch.stack(batch_rows, dim=0))
    return torch.stack(rows, dim=0).to(torch.int32)


def check_hyper_connection(device: torch.device) -> CheckResult:
    from miles_plugins.models.deepseek_v4.ops.hyper_connection import DeepSeekV4HyperConnectionUtil

    torch.manual_seed(SEED)
    cfg = SimpleNamespace(
        layernorm_epsilon=1e-6,
        dsv4_hc_mult=3,
        dsv4_hc_sinkhorn_iters=1,
        dsv4_hc_eps=1e-6,
    )
    util = DeepSeekV4HyperConnectionUtil(cfg)

    batch, seqlen, streams, hidden = 2, 4, 3, 5
    x = torch.randn(batch, seqlen, hidden, device=device, dtype=torch.float32)
    residual = torch.randn(batch, seqlen, streams, hidden, device=device, dtype=torch.float32)
    post = torch.randn(batch, seqlen, streams, device=device, dtype=torch.float32)
    comb = torch.randn(batch, seqlen, streams, streams, device=device, dtype=torch.float32)

    miles = util.hc_post_raw(x, residual, post, comb)
    expected = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual)
    wrong = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb, residual)

    # Megatron-LM PR #4839 reference formula uses [S, B, N, C]. Convert layout
    # only, not the math.
    h_res_sbnc = comb.permute(1, 0, 2, 3).contiguous()
    residual_sbnc = residual.permute(1, 0, 2, 3).contiguous()
    post_sbn = post.permute(1, 0, 2).contiguous()
    x_sbc = x.permute(1, 0, 2).contiguous()
    s, b, n, c = residual_sbnc.shape
    fixed_mixed = torch.bmm(
        h_res_sbnc.view(s * b, n, n).transpose(-1, -2),
        residual_sbnc.view(s * b, n, c),
    ).view(s, b, n, c)
    fixed = post_sbn.unsqueeze(-1) * x_sbc.unsqueeze(2) + fixed_mixed
    fixed_bsnc = fixed.permute(1, 0, 2, 3).contiguous()

    max_expected = (miles - expected).abs().max().item()
    max_wrong = (miles - wrong).abs().max().item()
    max_fixed = (miles - fixed_bsnc).abs().max().item()
    # The two expressions reduce products in a different order, so exact
    # bitwise equality is too strict for fp32. The acceptable drift is only the
    # expected single-ULP matmul/sum rounding noise.
    torch.testing.assert_close(miles, expected, rtol=2e-6, atol=3e-7)
    torch.testing.assert_close(miles, fixed_bsnc, rtol=2e-6, atol=3e-7)
    assert max_wrong > 1e-3, "non-symmetric H_res did not distinguish fixed and pre-fix formulas"
    return CheckResult(
        "hyper_connection_pr4839_orientation",
        "PASS",
        {
            "seed": SEED,
            "max_diff_vs_comb_transpose_residual": max_expected,
            "max_diff_vs_megatron_pr4839_fixed_native": max_fixed,
            "max_diff_vs_prefix_wrong_comb_residual": max_wrong,
        },
    )


def check_rope(device: torch.device) -> CheckResult:
    from miles_plugins.models.deepseek_v4.ops.rope import (
        apply_rotary_emb,
        precompute_freqs_cis,
        wrapped_precompute_freqs_cis,
    )

    torch.manual_seed(SEED)
    dim = 8
    seqlen = 7
    freqs = precompute_freqs_cis(
        dim=dim,
        seqlen=seqlen,
        original_seq_len=0,
        base=10000,
        factor=4,
        beta_fast=32,
        beta_slow=1,
    ).to(device)
    x = torch.randn(2, seqlen, 3, dim, device=device, dtype=torch.float32, requires_grad=True)
    before = x.detach().clone()
    y = apply_rotary_emb(x, freqs)
    recovered = apply_rotary_emb(y, freqs, inverse=True)
    loss = y.square().mean()
    loss.backward()
    cache_cfg = SimpleNamespace(
        original_max_position_embeddings=65536,
        rotary_scaling_factor=4,
        beta_fast=32,
        beta_slow=1,
    )
    cached_freqs = wrapped_precompute_freqs_cis(cache_cfg, rope_head_dim=dim, base=10000)

    input_mutation = (x.detach() - before).abs().max().item()
    inverse_error = (recovered.detach() - before).abs().max().item()
    storage_alias = y.data_ptr() == x.data_ptr()
    assert input_mutation == 0.0
    assert inverse_error < 5e-6
    assert not storage_alias
    assert x.grad is not None
    assert_finite("rope_grad", x.grad)
    assert cached_freqs.shape[0] == 131072
    return CheckResult(
        "rope_out_of_place_inverse_grad",
        "PASS",
        {
            "seed": SEED,
            "input_mutation_max_abs": input_mutation,
            "inverse_max_abs": inverse_error,
            "output_aliases_input_storage": storage_alias,
            "grad_finite": True,
            "wrapped_cache_seq_len": cached_freqs.shape[0],
        },
    )


def _load_official_kernel_module(inference_dir: Path):
    kernel_py = inference_dir / "kernel.py"
    if not kernel_py.exists():
        raise FileNotFoundError(f"{inference_dir} must contain kernel.py")

    sys.path.insert(0, str(inference_dir))
    spec = importlib.util.spec_from_file_location("deepseek_v4_official_operator_kernel", kernel_py)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_official_kv_qat_simulation(
    device: torch.device,
    *,
    official_inference_dir: Path | None,
) -> CheckResult:
    if official_inference_dir is None:
        return CheckResult("official_kv_qat_simulation", "SKIP", {"reason": "--official-inference-dir not set"})
    if device.type != "cuda":
        return CheckResult("official_kv_qat_simulation", "SKIP", {"reason": "official TileLang kernel requires CUDA"})

    from miles_plugins.models.deepseek_v4.ops.qat import fp8_simulate_qat

    official_kernel = _load_official_kernel_module(official_inference_dir)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    x = torch.randn(2, 3, 512, device=device, dtype=torch.bfloat16) * 2.0
    x[:, :, :64] = torch.linspace(-9.0, 9.0, 64, device=device, dtype=torch.bfloat16)

    official = x.detach().clone()
    official_kernel.act_quant(official, 64, None, torch.float32, True)
    miles = fp8_simulate_qat(x.detach().clone(), 64)

    diff = (official.float() - miles.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    nonzero = int((diff != 0).sum().item())
    exact_equal = bool((diff == 0).all().item())
    assert exact_equal
    return CheckResult(
        "official_kv_qat_simulation",
        "PASS",
        {
            "seed": SEED,
            "shape": list(x.shape),
            "dtype": "bfloat16",
            "block_size": 64,
            "scale_fmt": None,
            "scale_dtype": "float32",
            "inplace": True,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "nonzero_abs_count": nonzero,
            "exact_equal": exact_equal,
        },
    )


def _dense_reference_with_grad(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
    upstream_grad: torch.Tensor,
):
    from miles_plugins.models.deepseek_v4.ops.attention_core import dense_attn_torch

    q_ref = q.detach().clone().float().requires_grad_(True)
    kv_ref = kv.detach().clone().float().requires_grad_(True)
    sink_ref = attn_sink.detach().clone().float().requires_grad_(True)
    out = dense_attn_torch(q_ref, kv_ref, sink_ref, topk_idxs, sm_scale)
    (out.float() * upstream_grad.float()).sum().backward()
    return out.detach(), q_ref.grad.detach(), kv_ref.grad.detach(), sink_ref.grad.detach()


def check_dense_sparse_torch(device: torch.device) -> CheckResult:
    from miles_plugins.models.deepseek_v4.ops.attention_core import dense_attn_torch, sparse_attn_torch

    torch.manual_seed(SEED)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    batch, seqlen, heads, dim, kv_len, topk = 1, 8, 4, 16, 16, 8
    sm_scale = dim**-0.5
    q = torch.randn(batch, seqlen, heads, dim, device=device, dtype=dtype)
    kv = torch.randn(batch, kv_len, dim, device=device, dtype=dtype)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    topk_idxs = make_topk(batch, seqlen, kv_len, topk, device=device)
    topk_idxs[:, 0, :] = -1

    q_dense = q.detach().clone().requires_grad_(True)
    kv_dense = kv.detach().clone().requires_grad_(True)
    sink_dense = attn_sink.detach().clone().requires_grad_(True)
    q_sparse = q.detach().clone().requires_grad_(True)
    kv_sparse = kv.detach().clone().requires_grad_(True)
    sink_sparse = attn_sink.detach().clone().requires_grad_(True)

    dense_out = dense_attn_torch(q_dense, kv_dense, sink_dense, topk_idxs, sm_scale)
    sparse_out = sparse_attn_torch(q_sparse, kv_sparse, sink_sparse, topk_idxs, sm_scale)
    assert dense_out.dtype == dtype
    assert sparse_out.dtype == dtype
    assert_finite("dense_out", dense_out)
    assert_finite("sparse_out", sparse_out)
    forward_max = (dense_out.float() - sparse_out.float()).abs().max().item()
    forward_rel = rel_diff(dense_out, sparse_out)
    assert forward_max < (1e-5 if dtype == torch.float32 else 2e-2)
    assert forward_rel < 1e-5

    upstream_grad = torch.randn_like(dense_out).float()
    dense_loss = (dense_out.float() * upstream_grad).sum()
    sparse_loss = (sparse_out.float() * upstream_grad).sum()
    dense_loss.backward()
    sparse_loss.backward()

    assert q_dense.grad is not None
    assert kv_dense.grad is not None
    assert sink_dense.grad is not None
    assert q_sparse.grad is not None
    assert kv_sparse.grad is not None
    assert sink_sparse.grad is not None
    for name, tensor in {
        "dense_q_grad": q_dense.grad,
        "dense_kv_grad": kv_dense.grad,
        "dense_attn_sink_grad": sink_dense.grad,
        "sparse_q_grad": q_sparse.grad,
        "sparse_kv_grad": kv_sparse.grad,
        "sparse_attn_sink_grad": sink_sparse.grad,
    }.items():
        assert_finite(name, tensor)
    dq_rel = rel_diff(q_dense.grad, q_sparse.grad)
    dkv_rel = rel_diff(kv_dense.grad, kv_sparse.grad)
    dsink_rel = rel_diff(sink_dense.grad, sink_sparse.grad)
    assert dq_rel < 1e-5
    assert dkv_rel < 1e-5
    assert dsink_rel < 1e-5
    fully_masked_row_abs = sparse_out[:, 0].float().abs().max().item()
    assert fully_masked_row_abs == 0.0

    return CheckResult(
        "dense_sparse_torch_attention",
        "PASS",
        {
            "seed": SEED,
            "dtype": str(dtype).replace("torch.", ""),
            "shape": {"batch": batch, "seqlen": seqlen, "heads": heads, "dim": dim, "kv_len": kv_len, "topk": topk},
            "fully_masked_rows": 1,
            "forward_max_abs": forward_max,
            "forward_rel_diff": forward_rel,
            "dq_rel_diff": dq_rel,
            "dkv_rel_diff": dkv_rel,
            "dattn_sink_rel_diff": dsink_rel,
            "fully_masked_row_max_abs": fully_masked_row_abs,
            "grads_finite": True,
        },
    )


def _tilelang_case(
    device: torch.device,
    *,
    name: str,
    batch: int,
    seqlen: int,
    heads: int,
    dim: int,
    kv_len: int,
    topk: int,
    fully_masked_rows: int,
) -> dict[str, Any]:
    from miles_plugins.models.deepseek_v4.ops.attention_core import sparse_attn_tilelang

    torch.manual_seed(SEED)
    sm_scale = dim**-0.5
    dtype = torch.bfloat16
    q = torch.randn(batch, seqlen, heads, dim, device=device, dtype=dtype)
    kv = torch.randn(batch, kv_len, dim, device=device, dtype=dtype)
    attn_sink = torch.randn(heads, device=device, dtype=torch.float32)
    topk_idxs = make_topk(batch, seqlen, kv_len, topk, device=device)
    if fully_masked_rows:
        topk_idxs[:, :fully_masked_rows, :] = -1
    upstream_grad = torch.randn(batch, seqlen, heads, dim, device=device, dtype=torch.float32)

    ref_o, ref_dq, ref_dkv, ref_dsink = _dense_reference_with_grad(
        q, kv, attn_sink, topk_idxs, sm_scale, upstream_grad
    )

    q_tl = q.detach().clone().requires_grad_(True)
    kv_tl = kv.detach().clone().requires_grad_(True)
    sink_tl = attn_sink.detach().clone().requires_grad_(True)
    tl_o = sparse_attn_tilelang(q_tl, kv_tl, sink_tl, topk_idxs, sm_scale)
    (tl_o.float() * upstream_grad).sum().backward()

    assert q_tl.grad is not None
    assert kv_tl.grad is not None
    assert sink_tl.grad is not None
    for tensor_name, tensor in {
        "tilelang_out": tl_o,
        "tilelang_dq": q_tl.grad,
        "tilelang_dkv": kv_tl.grad,
        "tilelang_dattn_sink": sink_tl.grad,
        "dense_ref_out": ref_o,
        "dense_ref_dq": ref_dq,
        "dense_ref_dkv": ref_dkv,
        "dense_ref_dattn_sink": ref_dsink,
    }.items():
        assert_finite(tensor_name, tensor)

    fwd_max = (ref_o.float() - tl_o.float()).abs().max().item()
    fwd_rel = rel_diff(ref_o, tl_o)
    dq_rel = rel_diff(ref_dq, q_tl.grad)
    dkv_rel = rel_diff(ref_dkv, kv_tl.grad)
    dsink_rel = rel_diff(ref_dsink, sink_tl.grad)
    fully_masked_row_abs = (
        tl_o[:, :fully_masked_rows].float().abs().max().item() if fully_masked_rows else 0.0
    )
    assert fwd_rel < 1e-3
    assert fwd_max < 0.1
    assert dq_rel < 5e-2
    assert dkv_rel < 5e-2
    assert dsink_rel < 5e-2
    assert fully_masked_row_abs == 0.0

    return {
        "name": name,
        "seed": SEED,
        "dtype": "bfloat16",
        "shape": {"batch": batch, "seqlen": seqlen, "heads": heads, "dim": dim, "kv_len": kv_len, "topk": topk},
        "fully_masked_rows": fully_masked_rows,
        "forward_max_abs": fwd_max,
        "forward_rel_diff": fwd_rel,
        "dq_rel_diff": dq_rel,
        "dkv_rel_diff": dkv_rel,
        "dattn_sink_rel_diff": dsink_rel,
        "fully_masked_row_max_abs": fully_masked_row_abs,
        "grads_finite": True,
    }


def check_tilelang_sparse_attention(device: torch.device, *, skip_tilelang: bool) -> CheckResult:
    if skip_tilelang:
        return CheckResult("tilelang_sparse_attention", "SKIP", {"reason": "--skip-tilelang"})
    if device.type != "cuda":
        return CheckResult("tilelang_sparse_attention", "SKIP", {"reason": "CUDA unavailable"})
    try:
        import tilelang  # noqa: F401
        import miles_plugins.models.deepseek_v4.ops.attention_core  # noqa: F401
    except Exception as exc:
        return CheckResult("tilelang_sparse_attention", "SKIP", {"reason": repr(exc)})

    cases = [
        _tilelang_case(
            device,
            name="production_shape",
            batch=1,
            seqlen=128,
            heads=8,
            dim=512,
            kv_len=160,
            topk=64,
            fully_masked_rows=0,
        ),
        _tilelang_case(
            device,
            name="padded_topk_fully_masked_row",
            batch=1,
            seqlen=64,
            heads=8,
            dim=512,
            kv_len=96,
            topk=96,
            fully_masked_rows=1,
        ),
    ]
    return CheckResult(
        "tilelang_sparse_attention",
        "PASS",
        {
            "cases": cases,
            "max_forward_max_abs": max(case["forward_max_abs"] for case in cases),
            "max_forward_rel_diff": max(case["forward_rel_diff"] for case in cases),
            "max_dq_rel_diff": max(case["dq_rel_diff"] for case in cases),
            "max_dkv_rel_diff": max(case["dkv_rel_diff"] for case in cases),
            "max_dattn_sink_rel_diff": max(case["dattn_sink_rel_diff"] for case in cases),
            "grads_finite": all(case["grads_finite"] for case in cases),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--skip-tilelang", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--official-inference-dir", type=Path)
    args = parser.parse_args()

    if args.device == "cuda":
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    results: list[CheckResult] = []
    failures: list[str] = []
    checks = [
        ("hyper_connection_pr4839_orientation", lambda: check_hyper_connection(device)),
        ("rope_out_of_place_inverse_grad", lambda: check_rope(device)),
        (
            "official_kv_qat_simulation",
            lambda: check_official_kv_qat_simulation(device, official_inference_dir=args.official_inference_dir),
        ),
        ("dense_sparse_torch_attention", lambda: check_dense_sparse_torch(device)),
        ("tilelang_sparse_attention", lambda: check_tilelang_sparse_attention(device, skip_tilelang=args.skip_tilelang)),
    ]
    for check_name, check in checks:
        try:
            result = check()
        except Exception as exc:
            result = CheckResult(check_name, "FAIL", {"error": repr(exc)})
            failures.append(result.name)
        results.append(result)
        print(f"[{result.status}] {result.name}")
        for key, value in result.details.items():
            print(f"  {key}: {value}")

    payload = {
        "seed": SEED,
        "device": str(device),
        "environment": collect_environment(),
        "results": [asdict(result) for result in results],
        "overall_status": "PASS" if not failures else "FAIL",
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json_output}")

    print(f"overall_status={payload['overall_status']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

"""
DeepSeek-V4-Flash  unpacked BF16 HF  →  MegaBlocks (dpsk) FP8/FP4 converter.

This is the EXACT INVERSE of ``megablocks_to_hf_bf16.py``:
that tool dequantized the official MegaBlocks (dpsk) FP8/FP4 checkpoint into an
unpacked BF16 HF dir; this tool re-quantizes an unpacked BF16 HF dir back into
the MegaBlocks (dpsk) FP8/FP4 layout (same tensor names, dtypes, shapes, and
46-shard split as the original ``DeepSeek-V4-Flash``).

It is driven by a *template* (the original ``DeepSeek-V4-Flash`` dir): for every
tensor in the template we look up its dpsk name + target dtype + shard, fetch
the matching BF16 source tensor from the HF dir (via the authoritative
megablocks→HF name map), and re-quantize per the recipe below.

Quantization recipes (must invert dequant_fp8 / dequant_fp4 exactly):
  - tensors WITHOUT a ``.scale`` sibling in the template  (norms, embed, head,
    hc_*, attn_sink, gate.weight/bias, compressor.ape/norm/wgate/wkv, ...):
        copied as-is, cast to the template dtype (bf16 / f32 / i64).
  - FP8 e4m3fn weights (template dtype F8_E4M3: attn wq_*/wkv/wo_*,
    shared_experts w1/w2/w3, mtp e_proj):
        block 128x128; scale = round_up_pow2(absmax / 448) stored as
        float8_e8m0fnu; weight = round_e4m3(w / scale).
  - FP4 e2m1fn weights (template dtype I8: routed experts ffn.experts.N.wN):
        per-row, group-32 along K; scale = round_up_pow2(absmax / 6) stored as
        float8_e8m0fnu; 2 nibbles per byte along K (even idx -> low nibble,
        odd idx -> high nibble), encoded via the FP4 e2m1 table.

Idempotency (why this is a *lossless* round-trip on the dequant side):
A BF16 value produced by dequant is exactly ``q * 2^e`` where ``q`` is on the
e4m3/e2m1 grid and ``2^e`` is the (power-of-2) scale. Re-quantizing with a
power-of-2 (round-up) scale reproduces the SAME real value (possibly with a
different scale exponent + mantissa when the original block did not use the full
dynamic range), so ``dequant(requant(bf16)) == bf16`` bit-for-bit.

Usage:
  python hf_bf16_to_megablocks.py \
      --hf-src   /.../DeepSeek-V4-Flash-bf16-unpacked \
      --template /.../DeepSeek-V4-Flash \
      --dst      /.../DeepSeek-V4-Flash-roundtrip
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# Reuse the authoritative name-map + dequant helpers from the forward tool.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megablocks_to_hf_bf16 import (  # noqa: E402
    FP4_GROUP,
    FP8_BLOCK,
    _is_routed_expert,
    map_name_megablocks_to_hf,
)

FP8_MAX = 448.0
FP4_MAX = 6.0
E8M0_MIN_K = -127
E8M0_MAX_K = 127

# e2m1 magnitude grid + decision midpoints (matches FP4_TABLE in the forward tool).
_FP4_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def round_up_pow2(x: torch.Tensor) -> torch.Tensor:
    """Smallest power of two >= x, as fp32.  x must be >= 0.

    Uses frexp so exact powers of two map to themselves (no log2 rounding error).
    For x == 0 returns 2^E8M0_MIN_K (harmless: the block is all-zero, q == 0).
    """
    mant, exp = torch.frexp(x)  # x = mant * 2**exp, mant in [0.5, 1) for x>0
    # ceil(log2(x)): exp-1 when x is an exact power of two (mant == 0.5), else exp.
    k = torch.where(mant <= 0.5, exp - 1, exp)
    k = k.clamp(E8M0_MIN_K, E8M0_MAX_K)
    return torch.ldexp(torch.ones_like(x), k)


def quant_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 (M,K) -> (e4m3fn weight (M,K), e8m0 scale (M//128, K//128))."""
    w = w.to(torch.float32)
    M, K = w.shape
    assert M % FP8_BLOCK == 0 and K % FP8_BLOCK == 0, f"{w.shape} not /{FP8_BLOCK}"
    bM, bK = M // FP8_BLOCK, K // FP8_BLOCK
    wb = w.view(bM, FP8_BLOCK, bK, FP8_BLOCK).transpose(1, 2)  # (bM,bK,128,128)
    amax = wb.abs().amax(dim=(-1, -2))                          # (bM,bK)
    scale = round_up_pow2(amax / FP8_MAX)                       # (bM,bK) pow2 fp32
    q = (wb / scale[..., None, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    q = q.transpose(1, 2).reshape(M, K).contiguous()
    s = scale.to(torch.float8_e8m0fnu).contiguous()
    return q, s


def quant_fp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 (M,K) -> (int8 packed weight (M,K//2), e8m0 scale (M, K//32))."""
    w = w.to(torch.float32)
    M, K = w.shape
    assert K % FP4_GROUP == 0, f"K={K} not /{FP4_GROUP}"
    g = K // FP4_GROUP
    wg = w.view(M, g, FP4_GROUP)
    amax = wg.abs().amax(dim=-1)                    # (M,g)
    scale = round_up_pow2(amax / FP4_MAX)           # (M,g) pow2 fp32
    v = (wg / scale[..., None]).reshape(M, K)       # scaled, on/near e2m1 grid

    sign = (v < 0).to(torch.uint8)
    a = v.abs().clamp(0.0, FP4_MAX)
    mids = _FP4_MIDS.to(v.device)
    m = torch.bucketize(a, mids).to(torch.uint8)    # magnitude index 0..7
    nib = (sign << 3) | m
    nib = torch.where(m == 0, torch.zeros_like(nib), nib)  # canonical +0

    nib = nib.view(M, K // 2, 2)
    packed = (nib[..., 0] | (nib[..., 1] << 4)).to(torch.uint8)  # (M,K//2)
    qw = packed.view(torch.int8).contiguous()
    s = scale.to(torch.float8_e8m0fnu).contiguous()
    return qw, s


def _read_header(path: str) -> dict:
    """Return {tensor_name: {'dtype':..., 'shape':...}} from a safetensors file."""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _group_bases(names: list[str]) -> dict:
    """Group template tensor names into base -> {'weight':name, 'scale':name|None}."""
    groups: dict[str, dict] = {}
    for k in names:
        if k.endswith(".scale"):
            groups.setdefault(k[: -len(".scale")], {})["scale"] = k
        elif k.endswith(".weight"):
            groups.setdefault(k[: -len(".weight")], {})["weight"] = k
        else:  # standalone tensor (attn_sink, hc_*, ape, tid2eid, ...)
            groups.setdefault(k, {})["weight"] = k
    return groups


def convert(hf_src: str, template: str, dst: str, device: str = "cuda", fill_missing: bool = False):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(dst, exist_ok=True)

    tmpl_index = json.load(open(os.path.join(template, "model.safetensors.index.json")))
    tmpl_map = tmpl_index["weight_map"]
    hf_index = json.load(open(os.path.join(hf_src, "model.safetensors.index.json")))
    hf_map = hf_index["weight_map"]

    # shard file -> [dpsk tensor names], preserving the template's exact split.
    shard_to_keys: dict[str, list[str]] = {}
    for name, shard in tmpl_map.items():
        shard_to_keys.setdefault(shard, []).append(name)

    # lazy safe_open cache over the HF (bf16) source shards.
    hf_handles: dict[str, object] = {}

    def hf_get(hf_name: str) -> torch.Tensor:
        shard = hf_map[hf_name]
        if shard not in hf_handles:
            hf_handles[shard] = safe_open(os.path.join(hf_src, shard), framework="pt")
        return hf_handles[shard].get_tensor(hf_name)

    _DTYPE = {
        "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
        "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    }

    n_filled = 0
    for shard in tqdm(sorted(shard_to_keys), desc="shards", unit="shard"):
        hdr = _read_header(os.path.join(template, shard))
        groups = _group_bases(shard_to_keys[shard])
        out_state: dict[str, torch.Tensor] = {}
        tmpl_handle = safe_open(os.path.join(template, shard), framework="pt") if fill_missing else None

        for base, g in groups.items():
            wn = g["weight"]
            sn = g.get("scale")
            w_dtype = hdr[wn]["dtype"]
            hf_name = map_name_megablocks_to_hf(wn)

            # If the HF source lacks this tensor (e.g. training didn't save the MTP
            # layer), optionally copy the already-quantized tensor straight from the
            # template (original dpsk ckpt) instead of re-quantizing.
            if fill_missing and hf_name not in hf_map:
                out_state[wn] = tmpl_handle.get_tensor(wn)
                if sn is not None:
                    out_state[sn] = tmpl_handle.get_tensor(sn)
                n_filled += 1
                continue

            src = hf_get(hf_name).to(dev)

            if sn is None:
                # No scale in template -> copy as-is, cast to the template dtype.
                out_state[wn] = src.to(_DTYPE[w_dtype]).cpu()
            elif w_dtype == "F8_E4M3":
                qw, s = quant_fp8(src)
                out_state[wn] = qw.cpu()
                out_state[sn] = s.cpu()
            elif w_dtype == "I8" and _is_routed_expert(wn):
                qw, s = quant_fp4(src)
                out_state[wn] = qw.cpu()
                out_state[sn] = s.cpu()
            else:
                raise RuntimeError(f"Unexpected scaled tensor {wn} dtype={w_dtype}")

            # Hard invariants: produced tensors must match the template exactly.
            assert tuple(out_state[wn].shape) == tuple(hdr[wn]["shape"]), (
                f"{wn}: shape {tuple(out_state[wn].shape)} != template {hdr[wn]['shape']}")
            if sn is not None:
                assert tuple(out_state[sn].shape) == tuple(hdr[sn]["shape"]), (
                    f"{sn}: shape {tuple(out_state[sn].shape)} != template {hdr[sn]['shape']}")

        # sanity: same key set as the template shard
        assert set(out_state) == set(hdr), (
            f"{shard}: key mismatch "
            f"missing={set(hdr)-set(out_state)} extra={set(out_state)-set(hdr)}")
        save_file(out_state, os.path.join(dst, shard), metadata={"format": "pt"})
        del out_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Copy index verbatim (identical key->shard layout) + all non-weight aux files.
    shutil.copyfile(
        os.path.join(template, "model.safetensors.index.json"),
        os.path.join(dst, "model.safetensors.index.json"),
    )
    for fname in os.listdir(template):
        sp = os.path.join(template, fname)
        dp = os.path.join(dst, fname)
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        if os.path.isdir(sp):
            if not os.path.exists(dp):
                shutil.copytree(sp, dp, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copyfile(sp, dp)
    print(f"[ok] wrote {len(shard_to_keys)} shards to {dst}"
          + (f"  ({n_filled} tensors copied from template for missing HF sources)" if fill_missing else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-src", required=True, help="unpacked BF16 HF dir (e.g. DeepSeek-V4-Flash-bf16-unpacked)")
    ap.add_argument("--template", required=True, help="original dpsk FP8/FP4 dir (DeepSeek-V4-Flash) used as layout template")
    ap.add_argument("--dst", required=True, help="output dpsk FP8/FP4 dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fill-missing-from-template", action="store_true",
                    help="for tensors absent in --hf-src (e.g. an MTP layer the training didn't save), "
                         "copy the already-quantized tensor verbatim from --template instead of re-quantizing")
    args = ap.parse_args()
    convert(args.hf_src, args.template, args.dst, args.device, args.fill_missing_from_template)

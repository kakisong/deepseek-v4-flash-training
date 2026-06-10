"""
DeepSeek-V4-Flash  未打包 BF16 HF  →  MegaBlocks (dpsk) FP8/FP4 转换器。

本工具是 ``megablocks_to_hf_bf16.py`` 的精确逆操作:
那个工具把官方 MegaBlocks (dpsk) FP8/FP4 checkpoint 反量化为
未打包 BF16 HF 目录;本工具则把未打包 BF16 HF 目录重新量化回
MegaBlocks (dpsk) FP8/FP4 布局(与原始 ``DeepSeek-V4-Flash`` 相同的
tensor 名称、dtype、shape 以及 46 个 shard 的切分)。

它由一个*模板*(原始 ``DeepSeek-V4-Flash`` 目录)驱动:对模板中的每个
tensor,查出其 dpsk 名称 + 目标 dtype + 所在 shard,再(通过权威的
megablocks→HF 名称映射)从 HF 目录取出对应的 BF16 源 tensor,
并按下述方案重新量化。

量化方案(必须精确反演 dequant_fp8 / dequant_fp4):
  - 模板中没有 ``.scale`` 伴随项的 tensor (norms、embed、head、
    hc_*、attn_sink、gate.weight/bias、compressor.ape/norm/wgate/wkv 等):
        原样复制,转换为模板的 dtype(bf16 / f32 / i64)。
  - FP8 e4m3fn 权重(模板 dtype 为 F8_E4M3:attn wq_*/wkv/wo_*、
    shared_experts w1/w2/w3、mtp e_proj):
        128x128 分块;scale = round_up_pow2(absmax / 448),存为
        float8_e8m0fnu;weight = round_e4m3(w / scale)。
  - FP4 e2m1fn 权重(模板 dtype 为 I8:路由专家 ffn.experts.N.wN):
        按行、沿 K 每 32 个一组;scale = round_up_pow2(absmax / 6),存为
        float8_e8m0fnu;沿 K 每字节存 2 个半字节(偶数 idx -> 低半字节,
        奇数 idx -> 高半字节),按 FP4 e2m1 表编码。

幂等性(为何在 dequant 一侧这是*无损*往返):
dequant 产出的 BF16 值恰好是 ``q * 2^e``,其中 ``q`` 落在
e4m3/e2m1 网格上,而 ``2^e`` 是(2 的幂的)scale。用 2 的幂(向上取整)的
scale 重新量化会复现完全相同的实数值(当原始 block 没有用满
动态范围时,scale 的指数 + 尾数可能与原来不同),
因此 ``dequant(requant(bf16)) == bf16`` 逐位相等。

用法:
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

# 复用正向工具中权威的名称映射 + 反量化辅助函数。
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

# e2m1 幅值网格 + 判定中点(与正向工具中的 FP4_TABLE 一致)。
_FP4_MAGS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def round_up_pow2(x: torch.Tensor) -> torch.Tensor:
    """返回 >= x 的最小 2 的幂,以 fp32 表示。  要求 x >= 0。

    使用 frexp,使精确的 2 的幂映射为其自身(没有 log2 舍入误差)。
    当 x == 0 时返回 2^E8M0_MIN_K(无害:该 block 全为零,q == 0)。
    """
    mant, exp = torch.frexp(x)  # x = mant * 2**exp,x>0 时 mant 落在 [0.5, 1) 内
    # ceil(log2(x)):当 x 恰为 2 的幂(mant == 0.5)时取 exp-1,否则取 exp。
    k = torch.where(mant <= 0.5, exp - 1, exp)
    k = k.clamp(E8M0_MIN_K, E8M0_MAX_K)
    return torch.ldexp(torch.ones_like(x), k)


def quant_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 (M,K) -> (e4m3fn 权重 (M,K), e8m0 scale (M//128, K//128))。"""
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
    """BF16 (M,K) -> (int8 打包权重 (M,K//2), e8m0 scale (M, K//32))。"""
    w = w.to(torch.float32)
    M, K = w.shape
    assert K % FP4_GROUP == 0, f"K={K} not /{FP4_GROUP}"
    g = K // FP4_GROUP
    wg = w.view(M, g, FP4_GROUP)
    amax = wg.abs().amax(dim=-1)                    # (M,g)
    scale = round_up_pow2(amax / FP4_MAX)           # (M,g) pow2 fp32
    v = (wg / scale[..., None]).reshape(M, K)       # 已缩放,落在/接近 e2m1 网格

    sign = (v < 0).to(torch.uint8)
    a = v.abs().clamp(0.0, FP4_MAX)
    mids = _FP4_MIDS.to(v.device)
    m = torch.bucketize(a, mids).to(torch.uint8)    # 幅值索引 0..7
    nib = (sign << 3) | m
    nib = torch.where(m == 0, torch.zeros_like(nib), nib)  # 规范化为 +0

    nib = nib.view(M, K // 2, 2)
    packed = (nib[..., 0] | (nib[..., 1] << 4)).to(torch.uint8)  # (M,K//2)
    qw = packed.view(torch.int8).contiguous()
    s = scale.to(torch.float8_e8m0fnu).contiguous()
    return qw, s


def _read_header(path: str) -> dict:
    """从 safetensors 文件返回 {tensor_name: {'dtype':..., 'shape':...}}。"""
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _group_bases(names: list[str]) -> dict:
    """将模板 tensor 名分组为 base -> {'weight':name, 'scale':name|None}。"""
    groups: dict[str, dict] = {}
    for k in names:
        if k.endswith(".scale"):
            groups.setdefault(k[: -len(".scale")], {})["scale"] = k
        elif k.endswith(".weight"):
            groups.setdefault(k[: -len(".weight")], {})["weight"] = k
        else:  # 独立 tensor(attn_sink、hc_*、ape、tid2eid 等)
            groups.setdefault(k, {})["weight"] = k
    return groups


def convert(hf_src: str, template: str, dst: str, device: str = "cuda", fill_missing: bool = False):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    os.makedirs(dst, exist_ok=True)

    tmpl_index = json.load(open(os.path.join(template, "model.safetensors.index.json")))
    tmpl_map = tmpl_index["weight_map"]
    hf_index = json.load(open(os.path.join(hf_src, "model.safetensors.index.json")))
    hf_map = hf_index["weight_map"]

    # shard 文件 -> [dpsk tensor 名列表],保持与模板完全一致的切分。
    shard_to_keys: dict[str, list[str]] = {}
    for name, shard in tmpl_map.items():
        shard_to_keys.setdefault(shard, []).append(name)

    # 针对 HF(bf16)源 shard 的惰性 safe_open 缓存。
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

            # 若 HF 源缺少该 tensor(例如训练没有保存 MTP 层),
            # 可选择直接从模板(原始 dpsk ckpt)复制
            # 已量化好的 tensor,而不是重新量化。
            if fill_missing and hf_name not in hf_map:
                out_state[wn] = tmpl_handle.get_tensor(wn)
                if sn is not None:
                    out_state[sn] = tmpl_handle.get_tensor(sn)
                n_filled += 1
                continue

            src = hf_get(hf_name).to(dev)

            if sn is None:
                # 模板中无 scale -> 原样复制,转换为模板的 dtype。
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

            # 硬性不变量:产出的 tensor 必须与模板完全一致。
            assert tuple(out_state[wn].shape) == tuple(hdr[wn]["shape"]), (
                f"{wn}: shape {tuple(out_state[wn].shape)} != template {hdr[wn]['shape']}")
            if sn is not None:
                assert tuple(out_state[sn].shape) == tuple(hdr[sn]["shape"]), (
                    f"{sn}: shape {tuple(out_state[sn].shape)} != template {hdr[sn]['shape']}")

        # 健全性检查:键集合须与模板 shard 相同
        assert set(out_state) == set(hdr), (
            f"{shard}: key mismatch "
            f"missing={set(hdr)-set(out_state)} extra={set(out_state)-set(hdr)}")
        save_file(out_state, os.path.join(dst, shard), metadata={"format": "pt"})
        del out_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 原样复制 index(key->shard 布局完全相同)+ 所有非权重辅助文件。
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

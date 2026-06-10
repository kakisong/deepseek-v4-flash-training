"""V4 sparse-MLA 反向的直接实测开销拆解。

在空闲 H200 上,以生产形状用 CUDA event 分别独立计时 3 个 bwd kernel
(preprocess / bwd-main / postprocess)以及 fwd。目标是直接测量
bwd 时间花在哪里(尤其是 bwd-main 中 fp32 atomic_addx4 的 dKV scatter),
而不是用排除法推断。

在 GPU pod 内运行,需把 fsx 上的 miles 加入 PYTHONPATH:
    PYTHONPATH=/mnt/fsx-cdsn/.../train/miles python3 tools/v4_bwd_profile.py [S] [topk] [dist]
      S    : query/kv 长度(默认 4096)
      topk : 默认 640(window 128 + compress 512)
      dist : 'rand'(在 S_kv 上均匀分布)| 'local'(window 占比高,竞争激烈)
"""
from __future__ import annotations

import sys

import torch

from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_bwd as bwd_mod
from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_fwd as fwd_mod


def make_indices(B, S, S_kv, topk, dist, device, gen):
    """每个 query 的 topk 索引。'rand' = 均匀分布;'local' = 128 窗口(最近位置)+ 其余随机。"""
    if dist == "local":
        win = 128
        idx = torch.empty(B, S, topk, dtype=torch.int32, device=device)
        pos = torch.arange(S, device=device).view(1, S, 1)
        # window 部分:query 位置之前最近的 `win` 个位置(因果、连续 -> 竞争激烈)
        woff = torch.arange(win, device=device).view(1, 1, win)
        widx = (pos - woff).clamp(min=0)
        idx[:, :, :win] = widx.int()
        # compress 部分:随机的更早位置
        rest = topk - win
        idx[:, :, win:] = torch.randint(0, S_kv, (B, S, rest), dtype=torch.int32, device=device, generator=gen)
        return idx
    return torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=device, generator=gen)


def cuda_time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # 毫秒


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    dist = sys.argv[3] if len(sys.argv) > 3 else "rand"
    B, H, D = 1, 64, 512
    S_kv = S
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(0)

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device, generator=gen)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=device, generator=gen)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)
    topk_idxs = make_indices(B, S, S_kv, topk, dist, device, gen)

    # 跑一次前向,得到 o + lse
    o, lse = fwd_mod.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs)
    do = torch.randn_like(o)

    # 一次性 JIT 编译三个 bwd kernel
    pre_k = bwd_mod.preprocess(B, S, H, D)
    bwd_k = bwd_mod.bwd(B, S, S_kv, H, D, topk)
    post_k = bwd_mod.postprocess(B, S_kv, D)
    delta = pre_k(o, do)
    dkv_f32 = torch.zeros_like(kv, dtype=torch.float32)
    d_attn_sink = torch.zeros_like(attn_sink)

    def run_fwd():
        fwd_mod.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs)

    def run_pre():
        pre_k(o, do)

    def run_bwd_main():
        dkv_f32.zero_()
        d_attn_sink.zero_()
        bwd_k(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv_f32, d_attn_sink)

    def run_post():
        post_k(dkv_f32)

    t_fwd = cuda_time(run_fwd)
    t_pre = cuda_time(run_pre)
    t_main = cuda_time(run_bwd_main)
    t_post = cuda_time(run_post)
    t_bwd = t_pre + t_main + t_post

    tokens = B * S
    # 原子 dKV scatter(bwd-main)的 HBM 流量模型:
    #   每个 query token:NH 个 head-block * topk 个 key * D 个 fp32 值,每次原子操作 = RMW(读+写)= 8 字节
    NH = max(2 ** (H.bit_length() - 1 if H & (H - 1) == 0 else H.bit_length()), 16) // min(32, max(16, H))
    # NH = padded_H//block_H;H=64 时 -> padded 64,block_H 32 -> NH=2
    NH = 2 if H == 64 else None
    scatter_bytes = tokens * (NH or 1) * topk * D * 8  # fp32 的 RMW
    hbm_TBs = 4.8e12
    scatter_floor_ms = scatter_bytes / hbm_TBs * 1e3

    print(f"\n=== V4 sparse-MLA bwd cost  (B={B} S={S} S_kv={S_kv} H={H} D={D} topk={topk} dist={dist}) ===")
    print(f"  fwd                 : {t_fwd:8.3f} ms")
    print(f"  bwd.preprocess      : {t_pre:8.3f} ms  ({100*t_pre/t_bwd:5.1f}% of bwd)")
    print(f"  bwd.MAIN (atomic)   : {t_main:8.3f} ms  ({100*t_main/t_bwd:5.1f}% of bwd)  <-- target")
    print(f"  bwd.postprocess     : {t_post:8.3f} ms  ({100*t_post/t_bwd:5.1f}% of bwd)")
    print(f"  bwd TOTAL           : {t_bwd:8.3f} ms")
    print(f"  fwd+bwd             : {t_fwd + t_bwd:8.3f} ms")
    print(f"  --- atomic-scatter HBM model ---")
    print(f"  dKV atomic RMW bytes: {scatter_bytes/1e9:8.2f} GB   (NH={NH} * topk={topk} * D={D} * 8B/tok)")
    print(f"  HBM-floor @4.8TB/s  : {scatter_floor_ms:8.3f} ms   ({100*scatter_floor_ms/t_main:5.1f}% of bwd-main)")
    print(f"  tokens={tokens}  bwd-main per-token={1e3*t_main/tokens:.3f} us")


if __name__ == "__main__":
    main()

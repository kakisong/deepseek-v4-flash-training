"""tilelang_sparse_mla_bwd NaN bug 的最小复现脚本。

运行方式(在已安装 tilelang 的容器内):
    python3 examples/deepseek_v4_sft/tools/tilelang_sparse_mla_repro.py

本脚本把 V4 稀疏注意力的各实现(tilelang/sparse_torch/dense_torch)单独抽出来,
使 kernel 迭代只需数秒,而不是 Stage A 的 70+ 秒/iter 或 Stage B0 的 10+ 分钟。

复现 V4-Flash 生产形状(config.kv_lora_rank=512):
    B=1, S=1280, S_kv=1280, H=64, D=512, topk=640 (window 128 + compress 512)
    并包含会产生全 (-1) topk 行的早期因果 query。

对比 tilelang、sparse_attn_torch、dense_attn_torch 三者的梯度。
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.attention_core import (
    dense_attn_torch,
    sparse_attn_tilelang,
    sparse_attn_torch,
)


def make_inputs(*, B=1, S=1280, S_kv=1280, H=64, D=512, topk=640, device="cuda", seed=0):
    """V4-Flash 生产形状,带贴近真实的因果 mask topk 模式。"""
    g = torch.Generator(device=device).manual_seed(seed)

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=device, generator=g)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)

    # 早期 query 的因果 mask:位置 q < compress_ratio * topk 的 query 带有部分 -1。
    # 位置 0..2 被完全 mask(模拟 V4 indexer + clean_logits 的行为)。
    topk_idxs = torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=device, generator=g)
    for s_pos in range(S):
        # 模拟 _make_causal_cu_seqlens(seq_len_q, seq_len_kv, compress_ratio=4)
        valid_count = (s_pos + 1) // 4
        if valid_count == 0:
            topk_idxs[:, s_pos, :] = -1
        elif valid_count < topk:
            topk_idxs[:, s_pos, valid_count:] = -1
            topk_idxs[:, s_pos, :valid_count] = torch.randint(
                0, valid_count, (B, valid_count), dtype=torch.int32, device=device, generator=g
            )

    return q, kv, attn_sink, topk_idxs


def run(impl_fn, name, q, kv, attn_sink, topk_idxs):
    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    attn_sink = attn_sink.detach().clone().requires_grad_(True)
    o = impl_fn(q, kv, attn_sink, topk_idxs)
    n_nan_o = torch.isnan(o).sum().item()
    n_inf_o = torch.isinf(o).sum().item()
    finite_o = o[torch.isfinite(o)]
    if finite_o.numel() > 0:
        mn_o, mx_o = finite_o.min().item(), finite_o.max().item()
    else:
        mn_o = mx_o = float("nan")
    print(f"  [{name}] o: nan={n_nan_o} inf={n_inf_o} min={mn_o:.3e} max={mx_o:.3e}")

    do = torch.ones_like(o)
    o.backward(do)

    def stat(t, label):
        if t is None or t.grad is None:
            print(f"  [{name}] {label}: None")
            return
        g = t.grad
        n_nan = torch.isnan(g).sum().item()
        n_inf = torch.isinf(g).sum().item()
        finite = g[torch.isfinite(g)]
        if finite.numel() > 0:
            mn, mx = finite.min().item(), finite.max().item()
        else:
            mn = mx = float("nan")
        print(f"  [{name}] {label}: nan={n_nan} inf={n_inf} min={mn:.3e} max={mx:.3e}")

    stat(q, "dq")
    stat(kv, "dkv")
    stat(attn_sink, "d_attn_sink")


def main():
    assert torch.cuda.is_available(), "needs GPU"
    q, kv, attn_sink, topk_idxs = make_inputs()
    print(f"shapes: q={tuple(q.shape)} kv={tuple(kv.shape)} topk={topk_idxs.shape[-1]}")
    n_minus_one = (topk_idxs == -1).sum().item()
    n_total = topk_idxs.numel()
    print(f"causal -1 entries: {n_minus_one}/{n_total} ({100 * n_minus_one / n_total:.1f}%)")

    print("\n--- dense_attn_torch (reference) ---")
    run(dense_attn_torch, "dense_torch", q, kv, attn_sink, topk_idxs)

    print("\n--- sparse_attn_torch (reference) ---")
    run(sparse_attn_torch, "sparse_torch", q, kv, attn_sink, topk_idxs)

    print("\n--- sparse_attn_tilelang (broken) ---")
    run(sparse_attn_tilelang, "tilelang", q, kv, attn_sink, topk_idxs)


if __name__ == "__main__":
    main()

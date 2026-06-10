"""测试:完全不带 mask 的 tilelang bwd(所有 topk 索引均有效)。

若 NaN 仍然出现,则该 bug 与 -1 mask 无关 — 是 kernel 在 V4 形状下
的固有问题。若 NaN 消失,则与 mask 相关。
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.attention_core import sparse_attn_tilelang


def main():
    B, S, S_kv, H, D, topk = 1, 1280, 1280, 64, 512, 640
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(0)

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=device, generator=g)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=device)

    # 全部为有效索引(没有 -1)
    topk_idxs = torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=device, generator=g)

    print(f"shapes: q={tuple(q.shape)} kv={tuple(kv.shape)} topk={topk_idxs.shape[-1]}")
    print(f"-1 entries: {(topk_idxs == -1).sum().item()} (should be 0)")

    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    attn_sink = attn_sink.detach().clone().requires_grad_(True)

    o = sparse_attn_tilelang(q, kv, attn_sink, topk_idxs)
    print(f"o nan = {torch.isnan(o).sum().item()}")

    do = torch.ones_like(o)
    o.backward(do)

    for t, label in [(q, "dq"), (kv, "dkv"), (attn_sink, "d_attn_sink")]:
        g_t = t.grad
        n_nan = torch.isnan(g_t).sum().item()
        n_inf = torch.isinf(g_t).sum().item()
        print(f"  {label}: nan={n_nan} inf={n_inf} numel={g_t.numel()}")


if __name__ == "__main__":
    main()

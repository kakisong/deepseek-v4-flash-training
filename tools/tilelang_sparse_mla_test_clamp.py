"""假设检验:NaN 是否由读取 KV[by, -1, :] 的脏内存导致?

在 kernel 启动前,于 wrapper 内把 -1 索引替换为 0。这样 kernel 里的
`!= -1` mask 判断恒为 True(mask 语义因此是错的,输出也会错),
但不会再读到脏内存。如果 NaN 消失,即可确认根因是脏内存读取。
"""

from __future__ import annotations

import torch

from miles_plugins.models.deepseek_v4.ops.kernel import (
    tilelang_sparse_mla_bwd as bwd_mod,
    tilelang_sparse_mla_fwd as fwd_mod,
)
from tilelang_sparse_mla_repro import make_inputs


def sparse_attn_tilelang_clamped(q, kv, attn_sink, topk_idxs, sm_scale=None):
    """与 sparse_attn_tilelang 相同,但在调用 kernel 前把 -1 钳制为 0。"""
    safe_topk_idxs = topk_idxs.clamp(min=0).contiguous()

    class _F(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, kv, attn_sink, topk_idxs):
            o, lse = fwd_mod.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)
            ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o.clone(), lse)
            return o

        @staticmethod
        def backward(ctx, do):
            q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
            dq, dkv, d_attn_sink = bwd_mod.sparse_mqa_bwd_interface(
                q, kv, attn_sink, o, do, topk_idxs, lse, sm_scale=sm_scale
            )
            return dq, dkv, d_attn_sink, None

    return _F.apply(q, kv, attn_sink, safe_topk_idxs)


def run(impl_fn, name, q, kv, attn_sink, topk_idxs):
    q = q.detach().clone().requires_grad_(True)
    kv = kv.detach().clone().requires_grad_(True)
    attn_sink = attn_sink.detach().clone().requires_grad_(True)
    o = impl_fn(q, kv, attn_sink, topk_idxs)
    n_nan_o = torch.isnan(o).sum().item()
    print(f"  [{name}] o nan = {n_nan_o}")

    do = torch.ones_like(o)
    o.backward(do)

    for t, label in [(q, "dq"), (kv, "dkv"), (attn_sink, "d_attn_sink")]:
        g = t.grad
        n_nan = torch.isnan(g).sum().item()
        n_inf = torch.isinf(g).sum().item()
        print(f"  [{name}] {label}: nan={n_nan} inf={n_inf}")


def main():
    q, kv, attn_sink, topk_idxs = make_inputs()
    print(f"shapes: q={tuple(q.shape)} kv={tuple(kv.shape)} topk={topk_idxs.shape[-1]}")
    print(f"-1 entries: {(topk_idxs == -1).sum().item()}")

    print("\n--- sparse_attn_tilelang_clamped (-1 -> 0) ---")
    run(sparse_attn_tilelang_clamped, "clamped", q, kv, attn_sink, topk_idxs)


if __name__ == "__main__":
    main()

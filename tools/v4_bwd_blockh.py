"""以 query 为中心的快速调优杠杆:block_H(每 block 的 head 数)决定 NH = head-block 数量,
而每个 head-block 都会发出自己的原子 dKV scatter -> NH 会成倍放大 op_red 次数。
block_H=64(NH=1)相比生产配置 block_H=32(NH=2)能把原子归约操作减半,
代价是约 2 倍的 shared memory(可能被迫降到 num_stages=1)。该 kernel 受
L2 归约带宽限制,因此这有望是免费收益。直接测量验证。

Run: PYTHONPATH=<fsx miles> python3 tools/v4_bwd_blockh.py [S] [topk]
"""
import sys

import torch

from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_fwd as fwd_mod
from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_bwd import (
    bwd as prod_bwd,
    preprocess,
)

from v4_bwd_ablate import cuda_time, make_bwd  # noqa: E402


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    B, H, D = 1, 64, 512
    S_kv = S
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=dev, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=dev, generator=g)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=dev)
    topk_idxs = torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=dev, generator=g)
    o, lse = fwd_mod.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs)
    do = torch.randn_like(o)
    delta = preprocess(B, S, H, D)(o, do)
    d_attn_sink = torch.zeros_like(attn_sink)
    dkv = torch.zeros(B, S_kv, D, dtype=torch.float32, device=dev)

    print(f"\n=== block_H / num_stages sweep, atomic mode (B={B} S={S} H={H} D={D} topk={topk}) ===")
    print(f"  {'block_H':>8} {'NH':>3} {'ns':>3} {'direct_dq':>10} {'time(ms)':>10}  note")
    #     block_H, ns, direct_dq, split_store, note
    configs = [
        (32, 2, False, 2, "production"),
        (64, 1, True, 2, "NH=1 ns=1 +freed dQ"),
        (64, 2, True, 4, "NH=1 ns=2 +freed dQ +split4 (TARGET)"),
        (64, 2, True, 8, "NH=1 ns=2 +freed dQ +split8"),
        (64, 3, True, 4, "NH=1 ns=3 +freed dQ +split4"),
        (32, 2, True, 4, "NH=2 +freed dQ +split4"),
    ]
    base = None
    for bh, ns, ddq, ss, note in configs:
        try:
            k = make_bwd(B, S, S_kv, H, D, topk, "atomic", num_stages=ns, block_H_cap=bh,
                         direct_dq=ddq, split_store_n=ss)

            def run(k=k):
                dkv.zero_(); d_attn_sink.zero_()
                k(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink)
            t = cuda_time(run)
            if bh == 32 and ns == 2 and not ddq:
                base = t
            spd = f"{base / t:.2f}x" if base else ""
            print(f"  {bh:>8} {64 // bh:>3} {ns:>3} ddq={str(ddq):>5} ss={ss} {t:>9.3f}  {note}  {spd}")
        except Exception as e:
            print(f"  {bh:>8} {64 // bh:>3} {ns:>3} ddq={str(ddq):>5} ss={ss} {'FAIL':>9}  {note}: {str(e)[:45]}")

    # ---- 正确性:候选配置(NH=1, direct_dq)对比生产 bwd kernel,输入相同 ----
    print("\n=== correctness: candidate block_H=64/direct_dq atomic vs production block_H=32 ===")
    ref_k = prod_bwd(B, S, S_kv, H, D, topk)          # 生产 kernel
    cand_k = make_bwd(B, S, S_kv, H, D, topk, "atomic", num_stages=1, block_H_cap=64,
                      direct_dq=True, split_store_n=2)
    dkv_ref = torch.zeros(B, S_kv, D, dtype=torch.float32, device=dev)
    das_ref = torch.zeros_like(attn_sink)
    dq_ref = ref_k(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv_ref, das_ref)
    dkv_c = torch.zeros(B, S_kv, D, dtype=torch.float32, device=dev)
    das_c = torch.zeros_like(attn_sink)
    dq_c = cand_k(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv_c, das_c)
    for name, a, b in [("dq", dq_ref.float(), dq_c.float()),
                       ("dkv", dkv_ref, dkv_c), ("dAttnSink", das_ref, das_c)]:
        denom = a.abs().max().clamp(min=1e-8)
        print(f"  {name:10s}: max|ref|={a.abs().max():.4e}  max abs diff={ (a-b).abs().max():.3e}  "
              f"rel={(a-b).abs().max()/denom:.2e}  nan_c={torch.isnan(b).sum().item()}")


if __name__ == "__main__":
    main()

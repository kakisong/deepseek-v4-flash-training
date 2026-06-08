"""Causal ablation of the dKV atomic scatter in the V4 sparse-MLA backward.

Replicates the production bwd kernel EXACTLY except for the dKV store, which is
parametrized by `store_mode`:
  atomic     : production -- fp32 T.atomic_addx4 into dKV[Indices] (gather-scatter, RMW)
  coalesced  : non-atomic vectorized T.copy of the SAME bytes into a contiguous fp32
               scratch (no gather, no atomic, no contention). Keeps all GEMMs.
  coalesced16: same but bf16 scratch (half the store bytes) -- the de-atomic target dtype
  nostore    : skip the global store (GEMMs may be DCE'd; lower-bound sanity only)

delta(atomic - coalesced) = the direct, causal cost of doing dKV as an fp32 atomic
gather-scatter. This is the measurement, not an inference-by-elimination.

Run:  PYTHONPATH=<fsx miles> python3 tools/v4_bwd_ablate.py [S] [topk] [dist]
"""
import sys

import tilelang
import torch
from tilelang import language as T

from miles_plugins.models.deepseek_v4.ops.kernel import tilelang_sparse_mla_fwd as fwd_mod
from miles_plugins.models.deepseek_v4.ops.kernel.tilelang_sparse_mla_bwd import preprocess


@tilelang.jit(out_idx=[-3], pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
})
def make_bwd(B, S, S_kv, H, D, topk, store_mode, sm_scale=None, block_size=32,
             num_stages=2, threads=128, block_H_cap=32, direct_dq=False, split_store_n=2):
    # Single @tilelang.jit level (matches production bwd()): the shape locals live in
    # the SAME frame as @T.prim_func so the annotation strings resolve. store_mode is a
    # trace-time arg -> jit caches one compiled kernel per mode.
    assert topk % block_size == 0
    if sm_scale is None:
        sm_scale = D ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504
    q_shape = [B, S, H, D]
    kv_shape = [B, S_kv, D]
    o_shape = [B, S, H, D]
    indices_shape = [B, S, topk]
    delta_shape = [B, S, H]
    lse_shape = [B, S, H]
    attn_sink_shape = [H]
    padded_H = max(tilelang.math.next_power_of_2(H), 16)
    block_H = min(block_H_cap, padded_H)
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)
    split_store = split_store_n
    scratch_rows = NS * BS  # contiguous destination for the coalesced modes
    store_dtype = T.bfloat16 if store_mode == "coalesced16" else T.float32
    # atomic scatters into the full S_kv-length dKV; coalesced writes the contiguous scratch
    if store_mode == "atomic":
        dkv_rows, dkv_dtype = S_kv, T.float32
    else:
        dkv_rows, dkv_dtype = scratch_rows, store_dtype

    @T.prim_func
    def k(
        Q: T.Tensor(q_shape, T.bfloat16),
        KV: T.Tensor(kv_shape, T.bfloat16),
        dO: T.Tensor(o_shape, T.bfloat16),
        AttnSink: T.Tensor(attn_sink_shape, T.float32),
        Indices: T.Tensor(indices_shape, T.int32),
        Lse: T.Tensor(lse_shape, T.float32),
        Delta: T.Tensor(delta_shape, T.float32),
        dQ: T.Tensor(q_shape, T.bfloat16),
        dKV: T.Tensor([B, dkv_rows, D], dkv_dtype),
        dAttnSink: T.Tensor(attn_sink_shape, T.float32),
    ):
        with T.Kernel(S, B, NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, D], T.bfloat16)
            KV_shared = T.alloc_shared([BS, D], T.bfloat16)
            dO_shared = T.alloc_shared([block_H, D], T.bfloat16)
            P_shared_cast = T.alloc_shared([block_H, BS], T.bfloat16)
            dP_shared_cast = T.alloc_shared([block_H, BS], T.bfloat16)
            if not direct_dq:
                dQ_shared = T.alloc_shared([block_H, D], T.bfloat16)
            acc_p = T.alloc_fragment([block_H, BS], T.float32)
            acc_dp = T.alloc_fragment([block_H, BS], T.float32)
            acc_dq = T.alloc_fragment([block_H, D], T.float32)
            acc_dkv = T.alloc_fragment([BS, D], T.float32)
            acc_dkv_shared = T.alloc_shared([BS // split_store, D], store_dtype)

            T.copy(Q[by, s_i, bz * block_H:(bz + 1) * block_H, :D], Q_shared)
            T.copy(dO[by, s_i, bz * block_H:(bz + 1) * block_H, :D], dO_shared)
            T.clear(acc_dq)

            for i_i in T.Pipelined(NS, num_stages=num_stages):
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(
                        Indices[by, s_i, i_i * BS + bi_i] != -1, 0, -T.infinity(acc_p.dtype))
                for bi_i, d_i in T.Parallel(BS, D):
                    KV_shared[bi_i, d_i] = KV[by, T.if_then_else(
                        Indices[by, s_i, i_i * BS + bi_i] != -1,
                        Indices[by, s_i, i_i * BS + bi_i], 0), d_i]
                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(
                        acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i])
                T.copy(acc_p, P_shared_cast)
                T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True,
                       policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = (acc_p[h_i, bi_i]
                                         * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale)
                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, Q_shared, acc_dkv, transpose_A=True,
                       policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                # ---- the only part that varies ----
                if store_mode == "atomic":
                    for s in range(split_store):
                        for bi_i, d_i in T.Parallel(BS, D):
                            if bi_i < BS // split_store:
                                acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]
                        for bi_i, d_i in T.Parallel(BS // split_store, D // 4):
                            T.atomic_addx4(
                                dKV[by, Indices[by, s_i, i_i * BS + bi_i + s * (BS // split_store)], d_i * 4],
                                acc_dkv_shared[bi_i, d_i * 4])
                elif store_mode in ("coalesced", "coalesced16"):
                    for s in range(split_store):
                        for bi_i, d_i in T.Parallel(BS, D):
                            if bi_i < BS // split_store:
                                acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]
                        # non-atomic, contiguous, coalesced store of the SAME bytes
                        T.copy(acc_dkv_shared,
                               dKV[by, i_i * BS + s * (BS // split_store):
                                   i_i * BS + s * (BS // split_store) + BS // split_store, :])
                # nostore: nothing

            if direct_dq:
                T.copy(acc_dq, dQ[by, s_i, bz * block_H:(bz + 1) * block_H, :D])
            else:
                T.copy(acc_dq, dQ_shared)
                T.copy(dQ_shared, dQ[by, s_i, bz * block_H:(bz + 1) * block_H, :D])

    return k


def cuda_time(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    dist = sys.argv[3] if len(sys.argv) > 3 else "rand"
    B, H, D = 1, 64, 512
    S_kv = S
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=dev, generator=g)
    kv = torch.randn(B, S_kv, D, dtype=torch.bfloat16, device=dev, generator=g)
    attn_sink = torch.zeros(H, dtype=torch.float32, device=dev)
    if dist == "local":
        win = 128
        topk_idxs = torch.empty(B, S, topk, dtype=torch.int32, device=dev)
        pos = torch.arange(S, device=dev).view(1, S, 1)
        woff = torch.arange(win, device=dev).view(1, 1, win)
        topk_idxs[:, :, :win] = (pos - woff).clamp(min=0).int()
        topk_idxs[:, :, win:] = torch.randint(0, S_kv, (B, S, topk - win), dtype=torch.int32, device=dev, generator=g)
    else:
        topk_idxs = torch.randint(0, S_kv, (B, S, topk), dtype=torch.int32, device=dev, generator=g)

    o, lse = fwd_mod.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs)
    do = torch.randn_like(o)
    pre_k = preprocess(B, S, H, D)
    delta = pre_k(o, do)
    d_attn_sink = torch.zeros_like(attn_sink)

    NS = (topk + 31) // 32
    print(f"\n=== dKV atomic-scatter ABLATION (B={B} S={S} H={H} D={D} topk={topk} dist={dist}) ===")
    res = {}
    for mode in ["atomic", "coalesced", "coalesced16", "nostore"]:
        k = make_bwd(B, S, S_kv, H, D, topk, mode)
        scratch_rows = NS * 32
        sdt = torch.bfloat16 if mode == "coalesced16" else torch.float32
        dkv = torch.zeros(B, scratch_rows, D, dtype=sdt, device=dev) if mode != "atomic" \
            else torch.zeros(B, S_kv, D, dtype=torch.float32, device=dev)

        def run(k=k, dkv=dkv):
            dkv.zero_(); d_attn_sink.zero_()
            k(q, kv, do, attn_sink, topk_idxs, lse, delta, dkv, d_attn_sink)
        t = cuda_time(run)
        res[mode] = t
        print(f"  {mode:12s}: {t:8.3f} ms")
    print(f"  --------")
    print(f"  atomic - coalesced(fp32) = {res['atomic'] - res['coalesced']:7.3f} ms"
          f"  ({100*(res['atomic'] - res['coalesced'])/res['atomic']:5.1f}% of bwd-main is the atomic scatter)")
    print(f"  atomic - coalesced16(bf16)= {res['atomic'] - res['coalesced16']:7.3f} ms"
          f"  ({100*(res['atomic'] - res['coalesced16'])/res['atomic']:5.1f}% removable via de-atomic+bf16)")
    print(f"  GEMM/compute floor (nostore) = {res['nostore']:7.3f} ms ({100*res['nostore']/res['atomic']:5.1f}%)")


if __name__ == "__main__":
    main()

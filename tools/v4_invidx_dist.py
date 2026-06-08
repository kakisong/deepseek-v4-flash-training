"""Characterize the inverse-index load distribution for the V4 sparse-MLA backward.

A de-atomic KV-centric backward loops, per kv position j, over the queries that selected j.
The viability (ragged-GEMM / load-balance cost) depends on the distribution of n_j =
#queries referencing key j. Measure it for the realistic window(128 local)+compress(512)
index pattern, and report what a KV-centric tiling would cost vs the query-centric one.

Run: python3 tools/v4_invidx_dist.py [S] [topk]  (pure torch, any GPU/CPU)
"""
import sys

import torch


def main():
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    win = 128
    S_kv = S
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=dev).manual_seed(0)

    # realistic-ish: window = last `win` causal positions (contiguous), compress = random earlier
    pos = torch.arange(S, device=dev).view(S, 1)
    woff = torch.arange(win, device=dev).view(1, win)
    widx = (pos - woff).clamp(min=0)  # [S, win]
    cidx = torch.randint(0, S_kv, (S, topk - win), device=dev, generator=g)
    idx = torch.cat([widx, cidx], dim=1)  # [S, topk]

    counts = torch.bincount(idx.reshape(-1), minlength=S_kv).float()  # n_j per key
    nz = counts[counts > 0]
    print(f"\n=== inverse-index load distribution (S={S} S_kv={S_kv} topk={topk} win={win}) ===")
    print(f"  keys referenced            : {(counts>0).sum().item()}/{S_kv} "
          f"({100*(counts>0).float().mean().item():.1f}%)")
    print(f"  queries-per-key n_j  mean  : {nz.mean().item():.1f}")
    for p in [50, 90, 99, 99.9, 100]:
        print(f"                       p{p:<5}: {torch.quantile(nz, p/100).item():.0f}")
    print(f"  total valid pairs L         : {int(counts.sum().item())}  (= S*topk = {S*topk})")

    # KV-centric tiling cost model: pad each key's query list up to a multiple of BQ.
    for BQ in [16, 32, 64, 128]:
        tiles = torch.ceil(counts / BQ).sum().item()  # padded tiles across all keys
        ideal = counts.sum().item() / BQ
        waste = tiles * BQ / counts.sum().item()
        print(f"  BQ={BQ:>3}: padded query-tiles={int(tiles):>8}  (ideal {ideal:.0f}) "
              f"-> {waste:.2f}x compute vs query-centric (1.0 = no waste)")
    # window-only keys are dense (n_j ~ win); compress-only keys are the ragged tail
    win_keys = torch.bincount(widx.reshape(-1), minlength=S_kv).float()
    cmp_keys = torch.bincount(cidx.reshape(-1), minlength=S_kv).float()
    print(f"  window contributes mean n_j={win_keys[win_keys>0].mean().item():.1f} "
          f"(dense/contiguous), compress mean n_j={cmp_keys[cmp_keys>0].mean().item():.1f} (sparse)")


if __name__ == "__main__":
    main()

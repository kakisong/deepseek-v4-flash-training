"""刻画 V4 sparse-MLA 反向的逆索引负载分布。

去原子化的 KV 中心反向会对每个 kv 位置 j,遍历所有选中 j 的 query。
其可行性(ragged-GEMM / 负载均衡开销)取决于 n_j 的分布,n_j =
引用 key j 的 query 数。本脚本针对贴近真实的 window(128 local)+compress(512)
索引模式测量该分布,并报告 KV 中心 tiling 相对 query 中心方案的代价。

Run: python3 tools/v4_invidx_dist.py [S] [topk]  (纯 torch,任意 GPU/CPU)
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

    # 接近真实场景:window = 最近的 `win` 个因果位置(连续),compress = 随机的更早位置
    pos = torch.arange(S, device=dev).view(S, 1)
    woff = torch.arange(win, device=dev).view(1, win)
    widx = (pos - woff).clamp(min=0)  # [S, win]
    cidx = torch.randint(0, S_kv, (S, topk - win), device=dev, generator=g)
    idx = torch.cat([widx, cidx], dim=1)  # [S, topk]

    counts = torch.bincount(idx.reshape(-1), minlength=S_kv).float()  # 每个 key 的 n_j
    nz = counts[counts > 0]
    print(f"\n=== inverse-index load distribution (S={S} S_kv={S_kv} topk={topk} win={win}) ===")
    print(f"  keys referenced            : {(counts>0).sum().item()}/{S_kv} "
          f"({100*(counts>0).float().mean().item():.1f}%)")
    print(f"  queries-per-key n_j  mean  : {nz.mean().item():.1f}")
    for p in [50, 90, 99, 99.9, 100]:
        print(f"                       p{p:<5}: {torch.quantile(nz, p/100).item():.0f}")
    print(f"  total valid pairs L         : {int(counts.sum().item())}  (= S*topk = {S*topk})")

    # KV 中心 tiling 的代价模型:把每个 key 的 query 列表填充到 BQ 的整数倍。
    for BQ in [16, 32, 64, 128]:
        tiles = torch.ceil(counts / BQ).sum().item()  # 所有 key 填充后的 tile 总数
        ideal = counts.sum().item() / BQ
        waste = tiles * BQ / counts.sum().item()
        print(f"  BQ={BQ:>3}: padded query-tiles={int(tiles):>8}  (ideal {ideal:.0f}) "
              f"-> {waste:.2f}x compute vs query-centric (1.0 = no waste)")
    # 仅 window 的 key 是稠密的(n_j ~ win);仅 compress 的 key 是参差不齐的长尾
    win_keys = torch.bincount(widx.reshape(-1), minlength=S_kv).float()
    cmp_keys = torch.bincount(cidx.reshape(-1), minlength=S_kv).float()
    print(f"  window contributes mean n_j={win_keys[win_keys>0].mean().item():.1f} "
          f"(dense/contiguous), compress mean n_j={cmp_keys[cmp_keys>0].mean().item():.1f} (sparse)")


if __name__ == "__main__":
    main()

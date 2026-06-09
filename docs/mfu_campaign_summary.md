# V4-Flash H200 MFU 优化总结 (2026-06-05)

42×8 H200 (336 GPU) 上 DeepSeek-V4-Flash 128K 长上下文 SFT。目标：最大化训练吞吐/效率（MFU 目标 40%+）。

## 0. 结论速览

| 阶段 | MFU | TFLOPS/GPU | 说明 |
|---|---|---|---|
| 起点 (codex 配置, TCP) | ~2.4% | 23 | NCCL 走 TCP，整集群被通信卡死 |
| **EFA 修复后 (已上线)** | **7.2%** | **71** | **3× 吞吐，单卡反超 H20(~53)** |
| 配置层面天花板 | ~7.2% | ~71 | 已榨干，所有偏离更差或撞 bug |
| 40% 目标 | 需 kernel 工程 | — | 非配置可达 |

## 1. 前期 codex 探索 (docs/kaynzhang_077_h200_exploration.md, 06-04)

codex 在**不知道 NCCL 走 TCP** 的前提下系统扫了配置（全程 ~2% MFU，当时归因为"算力没发挥"但没定位到网络）：
- R1: no-offload CP6 full/uniform/1 → 19.4 TFLOPS, 60GB。
- P1/P1b/P1r: selective / 提 max_tokens → OOM；顺手修了 packed-THD zigzag CP 的 RoPE shape bug。
- P2: CP3/DP2 → 更差，停。
- **P3: full/block/4 + max_tokens=32768 → 23.3 TFLOPS（TCP 下最优，+20% vs R1）** ← 成为正式 run 默认。
- P4: block3 → -1.4%；P5: tok36864 → -45%；P6: VPP2 → -43%（都更差）。
- 关键遗漏：GPU util 99% 但只有 19-23 TFLOPS、节点网络仅 ~4GB/s——这其实就是 TCP 通信瓶颈的特征，但当时没往 EFA/NCCL 插件方向查。

## 2. 本次：根因 = NCCL 没走 EFA（已修复，3×）

**铁证**（在运行中的 worker 上实测）：GPU 100% util 但仅 120W（TDP 700W）；训练进程没加载 `libnccl-net.so`/`libfabric.so`；ENA 网卡 11GB/s 而 16 张 EFA 网卡 0 流量。镜像 `e1/deepseek-v4-flash:sft-only-20260609` 没打 aws-ofi-nccl 插件 → NCCL 静默退回 TCP socket。

**2 节点 nccl-test 对照**：all_reduce **5.6→410 GB/s (73×)**，all_to_all **1.3→82.8 GB/s (64×)**。

**修复**（免重建镜像）：把 AWS efa-installer 的 libfabric 2.4 + aws-ofi-nccl 1.19 + 匹配 rdma-core + libhwloc15 staging 到 fsx `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/efa/root`，`run.sh` 经 fleet env `V4_EFA_ENABLE=1` 注入 `NCCL_NET_PLUGIN/FI_PROVIDER=efa/FI_EFA_USE_DEVICE_RDMA=1/NCCL_PROTO=simple/NCCL_SOCKET_IFNAME=enp` + EFA-first `LD_LIBRARY_PATH`。`V4_EFA_ENABLE=0` 可回退。
**效果（真实负载 smoke）**：23→71 TFLOPS，2.4%→7.2% MFU，actor_train 626→206s，tok/s 11.4k→34.7k。

## 3. 本次：数据与工具

- **数据 ≤128K**：`albaliang_077_le128k.jsonl`（49667 条，剔除 332 条 >128K 的长尾 straggler，0.66%）。每条已有 `token_length` 字段，直接过滤。seq_length 134136→**131208**（可整除 2·CP，CP∈{2,3,6,7,14}）。
- **run.sh 新增**：`HW_FP8_ENABLED`（TE blockwise FP8 MoE GEMM）、`--cpu-offload`（重开优化器 offload）；新建 `tp8pp6cp7ep8` / `tp8pp3cp14ep8` scale。

## 4. 本次：MFU 配置探索（EFA 之后），全部探针

| # | 配置 | TFLOPS | 结果/原因 |
|---|---|---|---|
| E0 | PP7/CP6 block4 tok32768 | 71 | EFA 基线 = 配置最优 |
| P1 | +FP8 (MoE GEMM) | 71 | **零增益**——128K 下不是 GEMM 瓶颈；已按用户要求搁置 FP8 |
| P2 | CP7 selective | OOM | 激活显存太重 |
| P3 | block2 | 崩 | hash-router `input_ids` 断言（前3层hash路由，Megatron block-recompute 没穿 input_ids；block≥3 才侥幸过） |
| P4 | selective+offload | 崩 | `CheckpointWithoutOutput` 反向 bug（moe_act/layernorm 都中招，仅 core_attn 安全但会 OOM） |
| P5 | block4 **tok22528** | 14 | **慢 4.5×**——microbatch 变多，每-microbatch 固定开销巨大 |

**配置结论**：
- max_tokens=32768 是**尖锐最优**（两边都更差）。
- recompute 锁死 block4：减 recompute 的两条路（block<3、selective）都撞 Megatron bug；即便修了，recompute 仅占 ~18%。
- 激活显存 ≈ 43·tokens/(PP·CP) **与拓扑无关**，换 CP/PP 不腾显存。
- MFU 公式按 **dense O(n²)** 算注意力，实际 kernel 稀疏（top-512，~0.5%）→ 真实 FLOPs 极小，负载是**开销/带宽受限**，不是算力受限。

## 5. 本次：kernel 定位（进行中）

单卡微基准各算子耗时 → **稀疏注意力反向异常慢**：fwd 250ms vs fwd+bwd 12,603ms（**bwd ≈ 50× fwd**，正常应 ~2×）。指向 `tilelang_sparse_mla_bwd` 的 `num_stages=0`（无软流水）+ `block_size=32` 没为 H200 调。indexer 13.8ms、compressor 8.8ms、MoE 6.5ms 相对小。
（注：微基准的 topk_idxs 宽度建错了——用了 ~32K 而非真实 top-512 ~640，绝对值约放大 50×；**bwd≫fwd 的比值是真信号**，正在修宽度重测拿准确量级。）

## 6. 到 40% 的路线（kernel 工程，按 ROI）

kernel 都在 fsx（`miles/miles_plugins/models/deepseek_v4/ops/`），**改了不用重建镜像**，可快速迭代。
1. **修 sparse-MLA 反向 kernel**：`num_stages>0` 软流水 + 为 H200(132 SM) 调 block size（`HW_TILELANG_*` 现在是死参数，需接线）。← 最大嫌疑。
2. 优化 V4 indexer/compressor 的每-microbatch 开销（P5 的 4.5× 慢化指向这里）。
3. 打 Bug1 补丁（block-recompute 穿 input_ids）→ 解锁 block/3-2（小，需改 Megatron patch + 重建镜像）。
4. （后续）FP8 attention（kernel 工作量大、ROI 低，已搁置）。

## 7. 当前状态

- 正式 3-epoch run 已在**最优配置（EFA + le128k + block4/tok32768）**上验证可跑（3× 吞吐）；为做 kernel 探针当前已停集群。
- 下一步：修微基准 topk 宽度拿准量级 → 改 sparse-MLA bwd kernel → smoke 验证 MFU。

## 8. 2026-06-05 续: 40% MFU 不可达 + H20 对照真相 (wandb 实证)

**核心结论:40% MFU 对 H200 128K 是结构性不可达;H200 已是 H20 的 2.2× 单卡绝对吞吐,差距全在分母。**

实测数据点 (actor_train_tflops = 3·fwd_flops/world/time, fwd_flops 含 dense O(S²) 注意力):

| run | HW | nodes | TP/PP/CP | context | recompute | actor TFLOPS/GPU | MFU |
|---|---|---|---|---|---|---|---|
| lr0mjb4o (wandb) | H20 | 16 | 8/4/4 | 4K (seq_length=4096) | full | 49.8 | **33.7%** (÷148) |
| 本会话最优 | H200 | 42 | 8/7/6 | 128K | block/4 | **110.7** | **11.2%** (÷989) |
| 4K 稀疏(maxtok4096,none) | H200 | 16 | 8/4/4 | 4K | none | 5.46 | 0.55% (欠packing,overhead-bound) |
| 4K 密集(maxtok32768,none) | H200 | 16 | 8/4/4 | 4K | none | OOM | — |

**为什么 40% 不可达 (三重结构性原因):**
1. **分母**: H200 BF16 峰值 989 = H20 的 6.7×。H200 单卡 110 TFLOPS = H20(49.8) 的 2.2× 绝对吞吐,但 110/989 = 11% vs 49.8/148 = 34%。40% = 396 TFLOPS = H20 绝对吞吐的 8×,这个显存/通信/atomic 受限的 workload 给不出。
2. **激活显存 ∝ tokens/microbatch,与序列长度无关** → 128K **强制** ~42 路切分(PP7×CP6);PP4×CP4 在 128K **实测 OAOM**(135/140GB)。密集打包短样本到 32768 tok/microbatch 撞**同一道显存墙**(4K dense 也 OOM)。少切分装不下,多切分 = CP-ring 通信 + PP 气泡 → MFU 被压。
3. **sparse-MLA 反向 = 58% 算力**,9× 正向,fp32 dKV atomic scatter 受限,bf16 调优做不到 GEMM-bound(num_stages=2 已榨 -18%)。

**短上下文不救 MFU**: MFU 分子被 phantom O(S²) 注意力 credit 主导(128K 占 94%,4K 占 ~12%)。短上下文 = 同样的真实算力 + 更少 phantom credit = **更低** MFU(实测 4K 欠packing 仅 0.55%)。H20 的 4K 34% 是"小峰值好填满",不是"短上下文高效"。

**40% 路线(对抗验证后,bf16 上限 ~15%,+FP8+反向重写 ~20%):** 见 outputs/_mfu_ceiling_roadmap.json。便宜栈(threads=256 + PP3CP14)→ ~13%;de-atomic 反向重写(1-2 周)→ ~15%。

**正确的指标是绝对吞吐 / tok-s / GPU-hours,H200 已胜 H20 2.2×。MFU 不是跨不同峰值硬件的可比指标。**

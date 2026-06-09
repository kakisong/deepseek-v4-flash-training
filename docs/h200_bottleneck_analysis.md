# H200 vs H20:为什么 MFU% 偏低,以及真正的瓶颈在哪里

**TL;DR(基于对通信/计算的因果节流 + 对 bwd 的 DIRECT 内核 profiling,2026-06-06):**
在 V4-Flash 128K SFT 工作负载上,**comm ≈ 0%**(NCCL 扫描曲线平坦),且该步骤**从不是计算受限的**(SM ~16%)。主要开销来自 **sparse-MLA backward**,它是**单个内核**(bwd-main = backward 的 99%,是 forward 的 9×),并且**受限于其 fp32 dKV `atomic_addx4` scatter 的内存层级**:直接测得为 **L2-reduction-throughput 受限(~60% L2、5–33% DRAM、16% SM)**。一次去除该原子操作的因果 store-ablation 可恢复 **bwd-main 的 31.5%(fp32)– 43.4%(bf16)**。最大的单一杠杆是**对 backward 进行去原子化重写**(§7)。H200 的 6.7× tensor-FLOP 峰值在结构上无法被利用,因为瓶颈是一个非 tensor 的原子归约操作(由 L2/SM 发射,只随 ~1.7× 缩放而非 6.7×)。**这是直接验证的结论,而非排除法得出——见 §7。**
在相同规模下"H200 MFU% < H20 MFU%"是**合理且符合预期的**:H20 是一块刻意削减计算但保留带宽的芯片,因此在内存带宽受限的工作负载上,两块芯片的*绝对*吞吐相近(H200 约为 H20 的 1.2–1.86×),而 MFU%——它要除以大 6.7× 的 H200 计算峰值——会让 H200 看起来差 3×。**请用 tokens/sec/GPU、GPU-hours 和 $/token,而不是 MFU%,来比较这两块芯片。**

---

## 1. 硬件(这就是问题的全部)

| metric | H20 | H200 | ratio |
|---|---|---|---|
| BF16 compute | 148 TFLOPS | 989 TFLOPS | **6.7×** |
| HBM bandwidth | 4.0 TB/s | 4.8 TB/s | **1.2×** |
| NVLink | 900 GB/s | 900 GB/s | **1.0×** |
| TDP | ~400 W | 700 W | 1.75× |

H20 是符合出口管制的芯片:**核心削减约 41%(78/144 SMs),但保留了内存带宽和互连**(其 4.0 TB/s 甚至超过 H100 的 3.35)。它的设计就是为了在内存受限的工作负载上看起来很高效。来源:Tom's Hardware、Wccftech、NVIDIA/厂商规格表。

## 2. 测量结果(tokens/sec/GPU——无虚影、无峰值依赖)

相同工作负载(albaliang ≤128K 长上下文 SFT,gb=128,max_tokens=32768),来自 wandb 的稳态数据:

| config | tok/s total | **tok/s/GPU** | tflops/GPU | MFU% |
|---|---|---|---|---|
| H20 16-node TP8 PP4 CP4 +offload | 12,319 | **96** | 51 | 34.5% |
| H200 16-node TP8 PP4 CP4 +offload | ~13,860 | **~108** | 71 | 7.2% |
| H200 42-node TP8 PP7 CP6 no-offload | 57,800 | **172** | 114 | 11.5% |

- **在相同的 16-node 配置下,H200(108)≥ H20(96)。** H200 并不更慢。
- **在其最佳的 42-node 配置下,H200 = 172 tok/s/GPU = H20 的 96 的 1.79×。** 注意这 ≈ HBM
  带宽比,而非 6.7× 的计算比。
- MFU% 的反转(H200 11.5%"对" H20 34.5%)纯粹是 6.7× 峰值分母造成的。

(16-node 低于 42-node 是因为 PP4 强制启用 `--optimizer-cpu-offload`,其 PCIe D2H/H2D 税
在 H200 更快的步骤中占比更大——Amdahl 效应。42-node no-offload 才是更好的 H200 配置。)

## 3. 瓶颈三角定位(3 个独立实验)

| experiment (causal throttle) | result | conclusion |
|---|---|---|
| **NCCL channel sweep**(full → 8 → 2 channels) | tok/s 57.8k → 56.5k → 56.2k(**平坦,~3%**) | comm ≈ 关键路径的 **0%** |
| **SM-clock throttle**(1980 → 990 MHz,减半) | tok/s 56.2k → 36.1k(**−36%**) | **~56% 受 SM-throughput 限制** |
| **FP8 on MoE GEMMs** | 无加速 | 那 56% **不是** MoE-GEMM(tensor)计算 |
| **→ 分解** | comm 0% / SM-bound 56% / HBM 44% | 那 56% = **fp32 atomic-scatter**(sparse-MLA bwd) |

SM-clock 测试是决定性的:把计算时钟减半会损失 36% 吞吐 → 步骤中约 56% 对 SM-throughput
敏感。FP8 无加速排除了它是 MoE-GEMM,因此它是 sparse-MLA backward 中的 **fp32 dKV
`atomic_addx4` scatter**(原子操作由 SM 发射 → 随 SM 数量 ~1.7× H200/H20 缩放,而非 tensor
FLOPs 6.7×)。剩余约 44% 是 HBM 带宽(重计算/激活的读写)。这与最初的 micro-bench 一致
(sparse-MLA bwd = 计算的 58%,原子受限)。**这里通信确实是免费的**——余量巨大(2 channels 就够)。

NCCL 扫描在通信问题上是决定性的:把 collective 带宽节流到 **2 channels**
几乎不影响吞吐(~3% 下降,在工作点处斜率平坦)→ 通信有巨大余量,**不在**关键路径上。这**纠正了**早前(基于文献/理论的)
"communication-bound"的说法——这对 DeepSeek 的 H800(NVLink 削减到 400 GB/s)是成立的,但
在**这里是错误的**:完整的 900 GB/s NVLink + EFA + DeepEP overlap 让通信资源充裕且被隐藏。

因此瓶颈在于步骤中**HBM 流量密集**的部分:sparse-MLA gather/scatter(尤其是
fp32 dKV `atomic_addx4` scatter)、重计算(full)的激活读写、MLA、normalization、
elementwise。它们搬运的是字节,而非 FLOPs → H200 只能兑现其 1.2× 的 HBM 优势。

## 4. 这如何回答"H200 MFU < H20 是否合理"

合理。MFU% = achieved_tflops / peak。对一个 HBM 受限的工作负载,achieved tflops 跟随 HBM
带宽(两块芯片之间约 1.2×),但 H200 的 peak 大 6.7× → MFU% ≈ H20_MFU × 1.2/6.7
≈ H20 的 1/5.6。这块芯片做的真实工作相等或更多;只是这个指标除以了一个
无法利用的计算峰值。**在 H200 上达到 40% MFU 对此工作负载在物理上是不可达的**(= 396
tflops/GPU;HBM 受限的上限约为 114–170)。

## 5. 优化启示(这会重定向策略)

因为 bwd-main 在 dKV scatter 上受 L2-atomic-reduction 限制(comm/compute 已被实验排除,§3;
资源由 ncu 确认,§7):

- ❌ **FP8 / 计算优化**——无济于事(实测无加速;SM 仅 16%)。
- ❌ **Comm 优化**(更好的 DeepEP / overlap)——无济于事(已有余量)。
- ❌ **bf16 dKV reduction**——在 L2 常驻规模下无济于事:我们受限于原子**操作数**,而
  bf16x2 对相同数值需要 2× 的原子操作(只在 large-S DRAM-spill 区间帮助字节数)。
- ⚠️ **通过 KV-centric inversion 去原子化**——这是显而易见的想法,但在**这里适得其反**(见 §9):这是
  MQA(1 个 KV head 被 64 个 query head 共享),因此 KV-centric 会以 **64× head 代价**去 gather Q/dO——
  比它移除的原子操作流量大得多。§7 ablation 的"31–43% 可移除"是一个
  *理想化下限*(它保留了高效的 query-centric GEMMs 和一个免费的写操作);它**无法**直接
  实现。真实的去原子化上限 ≈ 6–15%,需通过 window-dense 混合 / query-block grouping(困难)。
- ✅ **现可用(已验证、低风险、~6%)**:`block_H=64`(NH=1,将 dKV op_red 数量减半)+
  直接 `acc_dq`→global store(释放 64 KB 的 `dQ_shared` 使 smem 容得下)。实测 12.90→12.11 ms
  (1.06×),S=4096;dq 梯度逐位精确,dkv 相对生产值为 rel 1.8e-6。见 §9。

## 6. H200-vs-H20 的正确指标(弃用 MFU%)

1. **tokens/sec/GPU**(稳态)——H200 为 H20 的 1.2–1.79×。
2. **GPU-hours / epoch**——H200 约为 H20 的 55–83%。
3. **$/trained-token** = GPU-hours × $/GPU-hour——业务决策指标。

## 7. DIRECT 内核测量(2026-06-06)——关闭遗留项,细化 §3

第 3 节是通过*排除法*三角定位瓶颈的。随后我们在一块空闲的 H200 上**直接**测量了 sparse-MLA
backward 内核(单 GPU,tilelang JIT,CUDA-event 计时 +
nsight-compute + 一次因果 store-mode ablation)。工具:`tools/v4_bwd_profile.py`、
`tools/v4_bwd_ablate.py`。Shapes:B=1,H=64,D=512(kv_lora),topk=640(window128+compress512)。

**(a) backward 是单个内核。** 在 S=4096:fwd 1.34 ms;bwd = preprocess 0.13 + **bwd-main
12.43** + postprocess 0.01 ms。**bwd-main 占 backward 的 98.9%,是 forward 的 9.3×。**
下面所有内容都是关于 bwd-main 的。

**(b) 因果 ablation——只替换 dKV store,保持每个 GEMM 完全相同**(S=4096,topk=640):

| bwd-main store variant | time | interpretation |
|---|---|---|
| `atomic`(production:fp32 `atomic_addx4` gather-scatter) | 12.89 ms | — |
| `coalesced`(非原子的连续 fp32 store,相同字节数) | 8.84 ms | **atomic scatter = 4.06 ms = bwd-main 的 31.5%** |
| `coalesced16`(非原子的 bf16 store) | 7.29 ms | **去原子化+bf16 移除 5.60 ms = 43.4%** |
| `nostore`(GEMM/计算下限) | 3.26 ms | 纯计算 = 25% |

`local`(window 密集、最大竞争)的索引分布给出**相同**的原子开销
(4.16 ms),与均匀的 `rand`(4.06 ms)一致 → 原子代价是**结构性的(RMW + 非合并 gather +
L2 reduction throughput),而非锁竞争。** 去原子化无论索引模式如何都获胜。

**(c) nsight-compute——实际被饱和的资源是什么。** `atomic_addx4` 被降级为一个全局
**reduction**(`lts__t_sectors_op_red`,而非 `op_atom`):

| metric | S=2048 (dKV 4MB, L2-resident) | S=32768 (dKV 64MB > 60MB L2) |
|---|---|---|
| L2 throughput | **59.6%**(最高) | **62.7%**(最高) |
| DRAM throughput | 5.1% | **33.4%** |
| Compute (SM) throughput | 15.6% | 15.9% |
| L2 reduction sectors | 251.7 M = 8.05 GB = 全部 L2 流量的 20% | (随规模缩放) |
| top warp stall | long-scoreboard 3.26 inst(global/L2 延迟) | — |

**结论(取代 §3 的"44% HBM-bound"):** bwd-main 受限于 fp32
dKV 原子归约 scatter 的内存层级——在所有规模下都受 L2-reduction-throughput 限制(L2 ~60%),
只有当 dKV 超过 60 MB 的 L2(large S)时才溢出到 DRAM(33%)。它**从不是计算受限的**(SM 钉在
~16%)。早前的 SM-clock 节流显示 56% 的敏感度,是因为降低 SM 时钟同时也
节流了*原子归约的发射速率*(reduction 由 SM 发射)——这是一致的,不矛盾。
去原子化重写被确认为唯一的杠杆,其**实测上限为 bwd-main 的 31.5%(fp32)–
43.4%(bf16)**,即整个 backward 的约 31–43%。

(Memory-clock 节流 `-lmc` 在这些 H200 上不受支持——只能延后处理——但 ncu 的直接
`gpu__dram_throughput`/`lts__throughput` 计数器让它变得不必要:它们表明 L2 而非 DRAM 才是墙。)

## 8. 经验配置扫描(2026-06-06)——PP7CP6 no-offload 是最优

测试过的每个替代方案都更差或不可行;当前生产配置胜出:

| config (42-node, TP8 EP8) | steady tok/s | tflops/GPU | verdict |
|---|---|---|---|
| **PP7 CP6, no-offload, recompute full/block4** | **56,226** | **114** | ✅ **最优** |
| PP7 CP6 **VPP2** | 51,300 | 101 | −9% —— interleave 的 P2P+调度开销 > 它节省的 4.7% bubble |
| PP7 CP3 **DP2** | (从未达到稳态) | — | gb=128/DP2 → 41 µbatch → bubble 14.6%(3×)+ packing 0.82 vs 0.94;慢到不可行 |
| PP6 CP7 | **OOM** | — | 更低的 PP → 7.2 layers/stage → 更多权重内存 |
| PP3 CP14 | OOM(受 PP6 限制) | — | 14 layers/stage |
| PP14 CP3 + recompute selective | **OOM**(在 step 上) | — | high-PP 的权重余量无法覆盖 selective 在 128K 下的激活 |
| recompute selective(任意 PP) | **OOM** | — | activation ∝ 43/(PP×CP)=const,在 128K 下太大 |
| DeepGEMM | n/a | — | 未安装;反正 MoE-GEMM 不是瓶颈(FP8 无加速) |
| cpu-offload | 已 OFF | — | 关闭 = 最优(16-node 强制开启 → 71 tflops,差得多) |

**为什么配置已经穷尽:** atomic-scatter 对配置免疫;comm 已经免费(无可调);
bubble 很小,而 VPP/DP 只会让它更糟;内存被双重封死——更低的 PP 在
权重上 OOM,减少重计算又在激活上 OOM。剩余的杠杆全都在 bwd 内核里(§9)。

## 9. 原子操作真的能去掉吗?(2026-06-06)——MQA 使之困难

§7 的 ablation 说去掉原子操作可恢复 bwd-main 的 31–43%。随后我们尝试*实现*它,
发现那个数字是一个**理想化下限,不可达**,因为这是 **MQA**(一个共享的 KV head
供全部 H=64 个 query head 使用)。流量统计(每 query,S=4096,topk=640,D=512):

| layout | gather traffic / query | dKV write | net traffic/query |
|---|---|---|---|
| **query-centric(当前)** | KV gather **0.65 MB**(topk·D·2;无 head 维——MQA) | atomic-scatter 5.2 MB(topk·D·NH·8,fp32 RMW) | ~6 MB |
| KV-centric(朴素去原子化) | Q+dO gather **~84 MB**(topk·H·D·2·2——那个 ×64 的 head 维!) | ~1 KB,无原子 | **~84 MB → ~14× 更差** |

因此为了消除原子而反转成 KV-centric 循环**适得其反**:它移除了一个 5 MB 的原子写,却
增加了一个 **84 MB 的 head 放大 Q/dO gather**(是当前 0.65 MB KV gather 的 128×)。原子操作是
流量最小的 MQA 布局的*正确*代价。

**尝试过的方案(均在空闲 H200 上测量,S=4096):**

| lever | result | why |
|---|---|---|
| `block_H=64` → NH=1(dKV op_red 数量减半) | 12.61 ms,1.02× | 强制 num_stages=1;pipeline 损失 ≈ 抵消原子节省 |
| NH=1 + **直接 `acc_dq`→global**(释放 64 KB 的 `dQ_shared`) | **12.11 ms,1.06×** ✅ | 保留 NH=1 节省而无 smem 墙;**dq 逐位精确,dkv 相对生产值 rel 1.8e-6** |
| NH=1 + ns=2(需 split_store=4 才能容下 smem) | 12.99 ms,0.99× | 额外的 shared-staging passes 抵消了更深的 pipe |
| NH=4(block_H=16) | 17.5 ms,0.74× | 更多 head-blocks = 更多冗余 scatter → 直接更差(确认 op-count 是墙) |
| bf16 dKV reduction(bf16x2) | 没有更快 | 受 L2-op-count 限制,而非字节;bf16x2 = 2× ops |

**结论。** 生产的 query-centric 内核**接近最优**。唯一免费、已验证的收益是
**NH=1 + direct-dq ≈ bwd-main 的 6%**(通过设置 `block_H=64` 并把 `acc_dq` 直接存入
`dQ` global、去掉 `dQ_shared` 来应用;重新加回 `dAttnSink` block;重跑 NaN 复现套件)。除此之外,
唯一 >10% 的路径是 **window-dense + compress-sparse 混合**(仅对 128 个连续
window keys 去原子化,通过 sliding-window backward——无 inverse index、无 H 代价;~6–10%,因为 window 占
keys 的 20%)或 **query-block grouping**(一个 block 拥有 G 个相邻 query,并在
scatter 前于 shared memory 中预归约它们共享的 window keys;~10–15%)——两者都是多日的内核项目,且受 MQA 限制
上限有限,**不是**理想化 ablation 所暗示的 31–43%。inverse-index 负载几乎是均匀的
(`tools/v4_invidx_dist.py`:n_j 均值 640,p99 694,BQ=64 padding 浪费 1.05×),因此负载均衡*不是*
瓶颈——MQA 的 head 放大才是。

## 10. 每 rank 序列长度驱动 MFU(2026-06-08)——PP3 的 CP 扫描

在启动 256K continued-SFT 运行时,浮现出一个此前被低估的杠杆:
**MFU 随每 rank 序列长度(= seq / CP)强烈上升**,因为更长的每 rank
序列使 attention 计算更密集、给出更胖的 GEMM,并更好地摊销
sparse-MLA 的 atomic-scatter 开销(§7)。

**两个实测锚点(均为 PP7/CP6,即生产布局):**
| run | per-rank tokens (max_tokens) | recompute | MFU (actor_train_tflops/989) |
|---|---|---|---|
| 128K SFT (kaynzhang_077_134k) | 32,768 | block-4 | **11.6%**(115 tflops) |
| 256K SFT (kaynzhang_077_256k) | 43,691 | full/uniform-1 | **17.6%**(174 tflops) |

per-rank 32K→44K 带来了 **+52% 的相对 MFU**——而且 256K 这个点是在*更重*的
(full/uniform)重计算下做到的,所以每 rank 长度效应甚至比原始数字更强。
这意味着并行化目标**不是"最大化 CP 以消除 PP bubble"**,而是**"在管理 bubble 的同时
保持每 rank 序列足够长(高 MFU)"**——高 CP 会缩短每 rank 序列并侵蚀 MFU。

**扫描(256K 数据,PP3,DP1,从 iter_1164 初始化,full/uniform-1 重计算):** 改变 CP 以改变
每 rank 序列,测量 MFU + 峰值内存。
| scale | CP | per-rank tokens | nodes (DP1) | result |
|---|---|---|---|---|
| tp8pp3cp4ep8  | 4  | 65,604 | 12 | **OOM** —— peak 99.5%(reserved 130GB),DeepEP 在内存墙处非法访问。PP3 的 14-layer/stage 权重 + CP4 的高激活超过 140GB。 |
| (baseline PP7) | 6 | 43,691 | 42 | 容得下(64% / 89GB),**MFU 17.6%** |
| tp8pp3cp7ep8  | 7  | 37,488 | 21 | 容得下但**临界边缘——in-step 约 99.5%**(127.8GB "before-clear"/91%,nvidia-smi in-step 约 143GB),**MFU 20.0%**(197.7 tflops,step-1 稳态) |
| tp8pp3cp14ep8 | 14 | 18,744 | 42 | 容得很宽裕(**84% / 117GB**),**MFU 15.0%**(~148 tflops,steps 1-5 稳态) |
| tp8pp3cp7ep8 (DP2) | 7 | 37,488 | 42 | 容得下但**临界 in-step 约 99.5%**(92% before-clear),**MFU 18.9%**(~187 tflops,~224s/step) |

**FINAL 结论(生产对比,全部 42-node 256K 配置):**
| config | MFU | actor_train/step | memory | scaling 21→42 |
|---|---|---|---|---|
| PP7/CP6(当前基线) | 17.6% | ~243s | 64%(安全) | — |
| **PP3/CP7 DP2** | **18.9%** | **~224s** | ~99.5%(临界) | **1.91×**(CP7-DP1 428s → 224s,接近理想) |
| PP3/CP14 DP1 | 15.0% | ~293s | 84%(安全) | 1.46×(差——短序列扼杀 MFU) |

- **PP3/CP7-DP2 是唯一击败 PP7/CP6 基线的 PP3 配置**(18.9% vs 17.6% → wall-clock 快约 8%,
  在一次 52h 的 2-epoch 运行上约 4h)。它保持每 rank 序列长(37.5K → 高 MFU),并从 21→42 节点
  **缩放 1.91×**(对比 CP14 的 1.46×),证实:**保持每 rank 序列长,不要把 GPU 花在 CP 上。**
- **DP2 的代价真实但很小**:CP7-DP1 20.0% → CP7-DP2 18.9%(−1.1pp,翻倍的 bubble + 284B
  的 DP all-reduce),如预测。
- **PP3/CP14-DP1 是死路**:15.0% < 基线。通过高 CP 用满全部 42 节点(DP1 的唯一方式)
  把每 rank 序列缩短到了 MFU 拐点以下。
- **生产的隐患**:CP7-DP2 在 in-step 内存约 99.5% 下运行——多日运行有在
  动态 batch 方差上 OOM 的风险。缓解:把 `max-tokens-per-gpu` 从 37488 削减到约 36000(in-step ~95%)以保留大部分
  +8% 同时留有余量。**决策 = +8% 速度(CP7-DP2,临界)vs 安全(PP7/CP6 基线)。**

**两个效应,都真实,而且会叠加(CP7 结果澄清了 §10):**
1. **per-rank-seq → MFU**(在固定 PP7 内是干净的):32K→11.6%,44K→17.6%。
2. **PP3 bubble 减少 → MFU**:PP3 CP7 达到 **20.0% > PP7 CP6 的 17.6%**,*尽管每 rank 序列更短*
   (37.5K < 43.7K)——即 PP3 更小的 bubble(~8% vs PP7 的 ~12-16%)足以补偿
   每 rank 序列的损失。所以当 PP3 能容下时,它是真正的胜利。

**内存是 PP3 的约束条件**(14-15 layers/stage 权重,无法由 CP/DP 分片):
- CP4(64K/rank):OOM。CP6(43.7K)在 PP3 下也会超。**CP7(37.5K)是实际的 PP3
  下限——已经 in-step 约 99.5%。** CP14(18.7K)是宽裕的 PP3 配置。
- 因此 256K 下可行的 PP3 区间是 **CP7–CP14(每 rank 18.7K–37.5K)**;CP7 最快但临界,CP14
  安全但序列更短。

**在固定 PP3 下对每 rank 序列效应的干净隔离**(决定性结果):CP7(37.5K/rank)=
**20.0%** vs CP14(18.7K/rank)= **15.0%**——同样 PP3,每 rank 序列减半 → **MFU 相对 −24%**。
每 rank 序列效应强到足以**压过 PP3 的 bubble 收益**:CP14 尽管有 PP3 的
小 bubble,却落在 PP7/CP6 基线(17.6%)*之下*,因为它 18.7K/rank 的序列太
短而无法计算密集。所以"用 PP3 在 DP1 下用满 42 节点"(强制 CP14)是 **MFU 负向的**。
因此生产问题归结为 **CP7-DP2(42 节点,保持 37.5K/rank,+DP2 bubble/comm,临界 99.5%
内存)vs CP14-DP1(42 节点,18.7K/rank,安全 84% 内存,15.0%)**——见 CP7-DP2 那一行。复现:`--workload sft_pp3_scan_smoke --scale tp8pp3cp<N>ep8
--fleet h200_k8s_<12|21|42>node --max-tokens-per-gpu <seq/N>`(seq 262416;一个 workload,CLI 覆盖)。

## 11. 轻 PP 解锁长 CP——MFU 在约 32K/rank 以上 SATURATES 于 ~20%(2026-06-09)

PP3 的内存墙(14-layer/stage 权重)把每 rank 序列封顶在 37.5K(CP7,临界)。改用**轻
PP**(PP8 = 5-6 layers/stage,init 后约 33GB 权重;PP4 = 11 layers/stage)释放内存以把
CP 推*低*(每 rank 序列推*高*),在 **DP1、无 DP 惩罚**下,全部在 32 节点(256 GPU)上。实测(同样的
`sft_pp3_scan_smoke` workload,init iter_1164,full/uniform-1 重计算,EFA on):

| scale | PP | CP | per-rank | nodes | MFU (perf1) | tflops/GPU | peak mem | actor_train/step |
|---|---|---|---|---|---|---|---|---|
| tp8pp4cp8ep8 | 4 | 8 | 32,802 | 32 | **19.4%** | 191.8 | ~110GB (78%) | 290s |
| tp8pp8cp4ep8 | 8 | 4 | 65,604 | 32 | **20.0%** | 197.3 | ~106GB (76%) | 282s |

**决定性的细化——MFU-vs-每 rank 序列曲线 SATURATES。** 在整个扫描中归一化到 DP1(干净、
无惩罚的点):

| per-rank | config | DP | nodes | MFU |
|---|---|---|---|---|
| 18.7K | PP3 CP14 | 1 | 42 | 15.0% |
| **32K** | **PP4 CP8** | 1 | 32 | **19.4%** |
| 37.5K | PP3 CP7 | 1 | 21 | 20.0% |
| 64K | PP8 CP4 | 1 | 32 | 20.0% |

拐点在 **~32K/rank**:18.7K→32K 很陡(**+4.4pp**),但 32K→37.5K→64K 是**平的(+0.6pp,然后 0)**。
**64K 相对 37.5K 不带来任何 MFU**——早前"MFU 在 64K 还会继续上升吗?"的问题解答为*不会,它
饱和*。所以长的每 rank 序列只在约 32K 拐点之前是必要的;超过之后,sparse-MLA 原子
下限(§7)无论如何把每 GPU 效率封顶在 ~20%(≈198 tflops,即 HBM 受限上限)。

**绝对效率(真正挑选生产的指标——不是 MFU%)。** 在单一 data/seq 内,
`tflops/GPU ∝ tok/s/GPU ∝ MFU`(FLOPs/token 固定),所以 MFU 在这里*确实*是一个有效的每 GPU 代理(§1–4 的"MFU
误导"警告**只针对跨芯片**)。MFU% 隐藏的是**节点数**。同样的 256K 数据,
GBS=128(~12.5M tok/step):

| config | nodes | tok/s/GPU | cluster tok/s | **GPU-h/step**(成本) | 2-epoch ETA¹(wall-clock) |
|---|---|---|---|---|---|
| **PP8 CP4** | 32 | 173 | 44.3k | **20.1** ✅ 最便宜 | ~61h |
| PP4 CP8 | 32 | 168 | 43.1k | 20.6 | ~63h |
| PP3 CP7-DP1 | 21 | 174 | 29.2k | 20.0 | ~92h |
| PP3 CP7-DP2 | 42 | 166 | 55.8k | 20.9 | **~48h** ✅ 最快 |
| PP3 CP14 | 42 | 127 | 42.6k | 27.3 | ~63h |
¹ ~778 steps(2×49,785 samples / GBS 128),仅计算;真实 wall-clock 更高(周期性 saves/eval)。

- **最便宜(最少 GPU-hours):PP8 CP4(32 节点)**——最高的每 GPU 效率 + 最少 GPU → ~15,600
  GPU-h,并释放 10 个节点。安全的 76% 内存。**MFU% 和 GPU-hours 一致认为它是每 GPU 最优。**
- **最快 wall-clock:CP7-DP2(42 节点)**——但*仅靠*多投入 31% 的 GPU;它比 PP8 CP4 *低* 4% 的
  GPU 效率,且在 99.5% 内存下运行临界。速度是用 GPU 买来的,而非效率。
- **PP4 CP8 ≈ PP8 CP4** 在成本上(都是 32 节点,~20 GPU-h/step);PP4 的 32K/rank 已经过了拐点。

**待解的生产杠杆(需要 40 节点 fleet):** ~20% 的上限是每 GPU 的;wall-clock = 20% × N_gpu。
DP1 下 ≥20% 的配置封顶在 32 节点(PP8CP4)或 21(CP7)。**PP8 CP5(51.2K/rank,40 节点,DP1)** 处于
32K 拐点之上 → 应在 **40 节点上、PP8 安全内存、无 DP 惩罚下保持 ~20%**——即 CP7-DP2 的
wall-clock *而无* 其 DP2 惩罚或内存临界。这是唯一一个未测试、可能同时击败
PP8CP4(更多节点)和 CP7-DP2(更高 MFU + 安全)的配置。**PP4 CP10(25.6K/rank,40 节点)** 在
拐点之下 → 预计 ~17-18%,是对 40 节点更差的利用。两者都需要 seq÷(2·5),(2·10):seq 262416 缺少一个因子
5 → 改用 **seq 262400**(÷80 → CP5 52480 / CP10 26240)+ scales `tp8pp8cp5ep8`/`tp8pp4cp10ep8` + 一个
`h200_k8s_40node` fleet。

### 40-node 实测结果(2026-06-09,进行中)
同样的 `sft_pp3_scan_smoke`,seq 262400,init iter_1164,full/uniform-1,EFA on,除非另注均为 DP1。每个都是
一次性 smoke(读取 perf 1-2 后删除 ckpt);这些数字是唯一的持久记录。

| scale | PP | CP | DP | per-rank | nodes | MFU | tflops/GPU | tok/s (total) | peak mem | step |
|---|---|---|---|---|---|---|---|---|---|---|
| tp8pp8cp5ep8 | 8 | 5 | 1 | 52,480 | 40 | **18.4%** | 182 | 50,829 | **~93GB (66%, safe)** | ~245s |
| tp8pp4cp10ep8 | 4 | 10 | 1 | 26,240 | 40 | **17.1%** | 169 | 46,968 | ~104GB (74%, safe) | ~267s |
| tp8pp4cp5ep8 | 4 | 5 | 2 | 52,480 | 40 | **20.5%** | 203 | ~57,000 | ~122GB (88%, ragged) | ~217s |

(PP4CP5DP2 稳态 = perf 2/3 = 21.0%/20.1%;perf 1 异常为 10.6%/424s——强制 save 的
尾部渗入了该步的计时。perf 0/1 的 tok/s 32.0k/29.4k 来自那些被污染的步骤。)

**PP8 CP5 推翻了 ~20% 的预测——它落在 18.4%,低于相同 PP8 下 PP8 CP4 的 20.0%。** 所以
"在 32K 以上平坦饱和"的图景过于简单:固定 PP8,51.2K(CP5)= 18.4% < 64K(CP4)=
20.0%。内存如预测(安全 66%,PP8 的轻权重),但 MFU 没有。

### 最终综合——三个假设被推翻,以及生产赢家(2026-06-09)
PP4CP5DP2 的 A/B 给出了结论。**在相同 40 节点的相同 51.2K/rank 下:PP4 CP5 DP2 = 20.5% 击败
PP8 CP5 DP1 = 18.4%。** 这单一对比推翻了三个早前的猜测:

1. **"轻 PP(PP8)更好"——被推翻。** 在相同每 rank 序列下,PP8 的 8-stage pipeline 比 PP4 *多花* ~1.6pp
   (18.4% vs 20.5%)。它的权重内存余量是真实的,但不值更深 pipeline 的
   开销。PP8CP5 的低 MFU 是 **PP8 造成的,而非 CP5。**
2. **"CP5(非 2 的幂)有 comm 惩罚"——被推翻。** PP4 *CP5* DP2 达到 20.5%——CP5 没问题。
   前一条消息的 CP5-惩罚猜测是错的;CP10/CP14 低只是因为它们每 rank 序列短。
3. **"DP2 花费 ~1.1pp"——被推翻(基本上)。** PP4CP5DP2(DP2)= 20.5%,完全处于 ~20% 饱和。DP2 的
   bubble/all-reduce 在 microbatch 保持高时是便宜的(GBS128/DP2 = 64 µb → bubble 4.7%)。早前的
   CP7 DP1→DP2 −1.1pp 下降是 **CP7-DP2 的 99.5% 内存临界**(重计算/batch 抖动),而非 DP2。

**每 PP 的干净饱和曲线(每 rank → MFU,每个 PP 单调,都饱和于 ~20%):**
- PP3:18.7K→15.0%,37.5K→20.0%   · PP4:25.6K→17.1%,32K→19.4%,51.2K→20.5%   · PP8:51.2K→18.4%,64K→20.0%

拐点 ~32K;上限 ~20%(≈198 tflops,即 sparse-MLA 的 atomic/HBM 墙)。超过 32K/rank
收益很小;**PP 选择(≤4)和让内存远离临界边缘,比追逐序列长度更重要。**

### 生产结论(256K 2-epoch)——绝对效率,而非 MFU%
GPU-h/step = step·GPU/3600(成本);2-epoch ETA = 778 steps × step,仅计算。

| config | nodes | MFU | tok/s | **GPU-h/step** | step | peak mem | 2-ep ETA |
|---|---|---|---|---|---|---|---|
| **PP4 CP5 DP2** | 40 | **20.5%** | **~57.0k** | **19.3** ✅ | **217s** ✅ | 88%(临界) | **~47h** ✅ |
| PP8 CP4 | 32 | 20.0% | 44.3k | 20.1 | 282s | 76%(安全) | ~61h |
| PP3 CP7 DP2 | 42 | 18.9% | 55.8k | 20.9 | 224s | 99.5%(临界) | ~48h |
| PP8 CP5 | 40 | 18.4% | 50.8k | 21.8 | 245s | 66%(安全) | ~53h |
| PP7 CP6(旧基线) | 42 | 17.6% | 51.4k | 22.7 | 243s | 64%(安全) | ~53h |

- **最快且最便宜:PP4 CP5 DP2(40 节点)**——最高 tok/s(57k),最低 GPU-h/step(19.3),
  最短 wall-clock(~47h)。它**严格支配旧的 CP7-DP2 候选**(更高 MFU,更多内存
  余量 88% vs 99.5%,更快)。唯一隐患是临界的 88% 内存 → 把 `max-tokens-per-gpu` 从 52480 削减到 ~50000
  以在多日运行上为动态 batch 留余量(小幅 MFU 代价)。
- **最安全的便宜选择:PP8 CP4(32 节点)**——在安全 76% 内存下 20.0%,20.1 GPU-h/step,并释放 10
  个节点;~61h(更慢只是因为它跑在更少 GPU 上,而非每 GPU 效率更低)。
- **建议:** 生产运行采用 **PP4 CP5 DP2(40 节点,max_tokens ~50000)**,以最低 GPU-hours 运行 ~47h;
  若 88% 内存被证实易 OOM 或别处需要这 10 个节点,则回退到 **PP8 CP4(32 节点)**。PP7CP6(原始基线)
  和 PP8CP5 都被支配。所有这些都处于由 sparse-MLA 原子内核(§7–9)封顶的 ~15–20%
  MFU 带内——真正的上限是那个内核,而非配置。

## 附录——可复现性
- 16-node 启动 + 4 个基础设施阻塞项/修复:见 memory `h200-k8s-16node-runbook`。
- NCCL 扫描旋钮:`V4_NCCL_MAX_NCHANNELS`(接入 run.sh 的 runtime_env)。
- wandb runs:H20 `lr0mjb4o`/`wkf2d2lc`;H200 42-node `jvp2ovpl`;NCCL 扫描 `yyrm0qlz`(8ch)/`simb8dhd`(2ch)。
- **bwd 内核 microbench 工具**(在空闲 H200 pod 上运行,`PYTHONPATH=<fsx miles>`,`CUDA_VISIBLE_DEVICES=0`):
  - `tools/v4_bwd_profile.py [S topk dist]` —— 逐内核 CUDA-event 分解(fwd/preprocess/bwd-main/postprocess)+ atomic HBM 模型。
  - `tools/v4_bwd_ablate.py [S topk dist]` —— 因果 store-mode ablation(atomic / coalesced-fp32 / coalesced-bf16 / nostore);也参数化 block_H / num_stages / direct_dq / split_store。
  - `tools/v4_bwd_blockh.py [S topk]` —— block_H/NH/ns/direct_dq/split 扫描 + 对生产内核的正确性检查。
  - `tools/v4_invidx_dist.py [S topk]` —— inverse-index 负载分布(每 key 的 query 数),用于去原子化可行性。
  - ncu:`ncu -k regex:sparse_mqa_bwd_kernel -c 1 -s 20 --section SpeedOfLight --section MemoryWorkloadAnalysis ... python3 tools/v4_bwd_profile.py 2048 640 rand`。

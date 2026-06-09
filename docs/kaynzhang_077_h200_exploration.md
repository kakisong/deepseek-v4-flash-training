# kaynzhang_077 H200 训练探索

日期:2026-06-04

## 目标

- 最大化训练 token 吞吐,以 `actor_train_tok_per_s`、`step_time` 以及 TFLOPS/MFU 作为主要决策指标。
- 仅当能提升训练 token 吞吐,或能在不 OOM 的前提下减少 recompute 时,才提高 H200 的 GRAM 利用率。
- 让 42 节点 / 336 GPU 资源池在生产运行中保持满占用。
- 在改动长跑配置前,优先采用单步 smoke 任务的测量结果。

## 集群

- Ray dashboard:`http://10.3.234.60:8201`
- Prometheus:`http://10.3.234.60:40001/promql`
- Grafana:`http://10.3.234.60:7777/grafana`
- GPU 资源池:42 个 H200 节点,336 个 GPU。
- CPU 节点:Ray head 加上一个 CPU worker。它们不应被训练 actor 占用。
- 已知基础设施风险:`10.3.22.244` 的 `/ray_local` 占用超过 95%,Ray 警告 object spilling 可能失败。

## 模型与数据

- 模型 checkpoint:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-FP8_torch_dist`
- SFT 数据:`$V4_DATA/albaliang_077_le134k.jsonl`
- 序列长度:`134136`
- 工具 key:`tools`

## 需记录的指标

- Ray job id、输出目录、W&B run URL。
- 规模:`TP`、`PP`、`CP`、`EP`、`ETP`、`DP`、pipeline layout。
- 内存旋钮:CPU offload、recompute 模式、`max_tokens_per_gpu`、`global_batch_size`。
- 通信旋钮:CP degree、TP locality、PP depth、DP degree、DeepEP、EP overlap。这些是诊断性指标,除非它们能推动 token 吞吐,否则不作为优化目标。
- 运行时指标:`actor_train_time`、`step_time`、tokens/GPU/s、有效 tokens/GPU/s、TFLOPS/MFU。
- Prometheus 指标:`DCGM_FI_DEV_FB_USED` 的最大值与平均值、`DCGM_FI_DEV_GPU_UTIL` 的平均值,以及在需要时查询的节点网络吞吐。
- 故障信号:CUDA OOM、Ray pending demands、dead actors、本地磁盘告警、checkpoint 保存失败。

## 运行记录

### R1:no-offload CP6 基线,成功

- 提交时间:2026-06-04 19:31 CST
- Job id:`raysubmit_nM7fY8TYCnciGCTC`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOff-H200-20260604-193125`
- Workload:`sft_kaynzhang_077_134k_smoke_no_offload`
- 规模:`tp8pp7cp6ep8`
- GPU 数:336
- 并行配置:`TP=8`、`PP=7`、`CP=6`、`EP=8`、`ETP=1`、有效 `DP=1`
- Layout:`Et*6|t*6|t*6|t*6|t*6|t*6|t*7L`
- CPU offload:禁用
- DeepEP:启用
- Router dtype:`fp32`
- Attention:`tilelang`
- Recompute:full、uniform、1 layer
- `max_tokens_per_gpu`:`32768`
- `global_batch_size`:`128`
- `rollout_batch_size`:`128`
- 调试模式:仅训练的单步 smoke
- Dump 细节:禁用
- Checkpoint optimizer 保存:通过 `--no-save-optim` 禁用
- W&B:启用,run URL `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/dp3jmf0n`

目前观测到:

- Ray 状态:`RUNNING`,无 pending demands,近期无 Ray 故障。
- Checkpoint 加载于 19:36:34-19:36:36 CST 完成。
- 模型与 optimizer 初始化时,采样 rank 上约 34.1GB GPU 内存。
- Rollout 于 19:37:23 CST 采集了 128 个样本。
- Rollout 指标:
  - response length 均值 `4719.26`
  - response length 最大值 `29050`
  - rollout 时间 `43.39s`
  - rollout tokens/GPU/s `185.62`
  - rollout 有效 tokens/GPU/s `41.43`
- 训练步于 19:37:25 CST 开始。
- 训练步于 19:49:49 CST 结束。
- 训练指标,记录于 19:49:59 CST:
  - `perf/update_weights_time`:`0.000033s`
  - `perf/data_preprocess_time`:`0.846s`
  - `perf/train_wait_time`:`45.917s`
  - `perf/actor_train_time`:`752.889s`
  - `perf/train_time`:`754.174s`
  - `perf/actor_train_tflops`:`19.411`
  - `perf/actor_train_tok_per_s`:`9493.726`
  - `perf/step_time`:`800.091s`
  - `perf/wait_time_ratio`:`0.0574`
- Checkpoint 保存于 19:49:59 CST 开始;首次保存于 19:50:24 CST 完成,随后启动了第二次终态保存。
- 第二次 checkpoint 保存于 19:50:47 CST 完成。
- Ray job 状态:于 19:51:24 CST 成功。
- 关闭告警:W&B 拆解时在 `teardown_atexit` 中抛出 `ConnectionResetError: Connection lost`;Ray job 仍然成功,且指标已发出。
- Smoke checkpoint 大小:`531G`;收集指标后已删除。
- 训练期间的 Prometheus,19:46 CST:
  - 最大 `DCGM_FI_DEV_FB_USED`:`63962 MiB`
  - 平均 `DCGM_FI_DEV_FB_USED`:`59624 MiB`
  - 平均 `DCGM_FI_DEV_GPU_UTIL`:约 `99.4%`
- Prometheus 滚动窗口,19:48 CST:
  - 10 分钟最大 `DCGM_FI_DEV_FB_USED`:`63976 MiB`
  - 10 分钟平均 `DCGM_FI_DEV_FB_USED`:`58748 MiB`
  - 5 分钟平均 `DCGM_FI_DEV_GPU_UTIL`:`99.63%`
- Ray 网络速率,19:50 CST:
  - 节点平均发送:`3.97e9 B/s`
  - 节点平均接收:`3.95e9 B/s`
  - 节点最大发送:`9.64e9 B/s`
  - 节点最大接收:`9.65e9 B/s`
  - 最热的节点包括 `10.3.28.82`、`10.3.77.237`、`10.3.75.76`、`10.3.94.204`、`10.3.22.244`

阶段性解读:

- 禁用 CPU offload 在 `32768` tokens/GPU 下配合 full recompute 可以从容容纳。
- 当前 GRAM 利用率对于 H200 而言仍然偏低,在观测的训练窗口内平均约 60GB、最大约 64GB。
- 该任务处于计算活跃状态,而非 Ray-pending:整个资源池的 GPU util 接近 99%。
- Full recompute 很可能让单步开销很大;下一步探测应减少 recompute 并/或提高本地 token 量。
- 实测的 `19.4 TFLOPs/GPU` 远低于 H200 BF16 峰值,因此后续探测应聚焦于降低 recompute 开销、提升训练 token 吞吐,而不仅仅是提高原始 GPU 占用率。

## 候选后续探测

### P1:CP6,更高 GRAM,更少 recompute

- Workload:`sft_kaynzhang_077_134k_smoke_no_offload_mem`
- 规模:`tp8pp7cp6ep8`
- 提交时间:2026-06-04 19:52 CST
- Job id:`raysubmit_nfVtsSDzmTcdCgLH`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffMem-H200-20260604-195257`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/n8sve6pk`
- 状态:失败
- 目标:在保持相同通信拓扑的前提下,提高 GRAM 利用率并降低 full recompute 开销。
- 相对 R1 的改动:
  - `max_tokens_per_gpu=49152`
  - `recompute_granularity=selective`
  - CPU offload 仍然禁用
- 启动校验:
  - `log_probs_max_tokens_per_gpu=49152`
  - `max_tokens_per_gpu=49152`
  - `recompute_granularity=selective`
  - `recompute_method=None`
  - entrypoint 中无 optimizer CPU offload 相关 flag
- 失败情况:
  - 在 rollout 和 data preprocess 之后不久,于 actor 训练开始处失败。
  - 错误:`RuntimeError: shape '[1, 49152, 1, 32]' is invalid for input of size 524288`
  - 堆栈指向 `miles_plugins/models/deepseek_v4/ops/rope.py:64`,`freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))`。
  - 这不是 OOM;看起来是由 `max_tokens_per_gpu=49152` 触发的 RoPE/动态 batch 形状假设问题。
  - 输出目录保持较小,约 `3.5M`;无需清理 checkpoint。
- 与 R1 的对比项:
  - 峰值和平均 GRAM
  - actor 训练时间
  - tokens/GPU/s 与 TFLOPS
  - 任何 OOM 或 checkpoint 行为

### P1b:CP6,在已知可行 token 上限下做 selective recompute,因 OOM 失败

- Workload:`sft_kaynzhang_077_134k_smoke_no_offload_selective`
- 规模:`tp8pp7cp6ep8`
- 提交时间:2026-06-04 20:01 CST
- Job id:`raysubmit_v8gihpHAHy7H2gAA`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffSel-H200-20260604-200141`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/i1gp1dnn`
- 状态:约于 2026-06-04 20:09 CST 失败
- 目标:在 R1 已知可行的 token 上限下隔离 selective recompute。
- 相对 R1 的改动:
  - `recompute_granularity=selective`
  - `max_tokens_per_gpu=32768`
  - CPU offload 仍然禁用
- 原因:P1 的 `49152` token 上限在还没来得及测量内存或吞吐之前,就在 RoPE 形状处理中失败了。
- 启动校验:
  - `log_probs_max_tokens_per_gpu=32768`
  - `max_tokens_per_gpu=32768`
  - `recompute_granularity=selective`
  - `recompute_method=None`
- 观测:
  - Checkpoint 加载约于 20:06 CST 完成。
  - 模型与 optimizer 初始化时,采样 rank 上约 34.1GB GPU 内存,与 R1 相同。
  - Rollout 于 20:07:31 CST 采集了 128 个样本。
  - Rollout 指标:
    - response length 均值 `4719.26`
    - response length 最大值 `29050`
    - rollout 时间 `43.36s`
    - rollout tokens/GPU/s `185.75`
    - rollout 有效 tokens/GPU/s `41.46`
  - `train_wait` 耗时 `46.7s`。
  - `data_preprocess` 耗时 `0.8s`。
  - Actor 训练于 20:07:32 CST 开始,在 `49.4s` 后失败。
- 失败情况:
  - 错误:`torch.OutOfMemoryError: CUDA out of memory`
  - 尝试分配:`6.00 GiB`
  - 日志中失败 GPU 状态:total `139.80 GiB`,free `3.49 GiB`,进程占用内存 `136.28 GiB`,PyTorch 已分配 `131.49 GiB`,PyTorch 已 reserved 但未分配 `401.73 MiB`。
  - Ray 还在 `10.3.22.244` 上重复了已有的 `/ray_local` 占用超过 95% 的告警;这是基础设施风险,但直接故障是 CUDA OOM。
- 失败前后的 Prometheus,约 20:09 CST 的 10 分钟滚动查询:
  - 最大 `DCGM_FI_DEV_FB_USED`:`139175 MiB`
  - 每 GPU 滚动最大值的平均 `DCGM_FI_DEV_FB_USED`:`67006 MiB`
  - 最大 `DCGM_FI_DEV_GPU_UTIL`:`100%`
  - 每 GPU 滚动最大值的平均 `DCGM_FI_DEV_GPU_UTIL`:`69.6%`
- 解读:
  - P1b 证明 P1 中的 RoPE 形状失败是由 `49152` token 上限引起的,而非仅由 selective recompute 引起。
  - 在 `32768` tokens/GPU 下,selective recompute 可以进入训练,但激活内存最终会触及 H200 上限并 OOM。
  - 稳定的 no-offload 区间介于 R1 full recompute 约 64GB 峰值与 P1b selective 约 139GB 峰值之间;下一步探测应保持无 CPU offload,但采用更温和的 recompute 缩减,或降低每 rank 的 token 压力。

### 面向 packed THD + zigzag CP 的 RoPE 形状补丁

- 约于 2026-06-04 20:18-20:21 CST 应用。
- 涉及文件:
  - `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles/miles_plugins/models/deepseek_v4/ops/cp_utils.py`
  - `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles/miles_plugins/models/deepseek_v4/deepseek_v4.py`
  - `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles/miles_plugins/models/deepseek_v4/ops/v4_indexer.py`
  - `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles/miles_plugins/models/deepseek_v4/ops/compressor.py`
- 根因:
  - Miles THD CP 切分在每个 CP rank 上为每个样本使用两个 zigzag chunk。
  - V4 attention 路径假设 CP 是连续的,并用 `cp_rank * local_seqlen` 来切分 RoPE 频率。
  - 当 `local_seqlen=49152` 时,CP rank 5 从长度为 `262144` 的 RoPE 表中切出 `[245760:294912]`,仅剩下 `16384` 行。这产生了观测到的 `shape '[1, 49152, 1, 32]' is invalid for input of size 524288`。
- 补丁行为:
  - 从 `packed_seq_params.cu_seqlens_q` 构建显式的逐 token 位置。
  - 在 packed 样本边界处重置 RoPE 位置。
  - 将每个样本的两个本地 zigzag chunk 映射到该样本内部的真实全局位置。
  - 在 attention、V4 indexer 和 compressor 中按显式位置索引 RoPE 频率。
  - 若 `cu_seqlens` 无法映射回本地 CP chunk,或任何位置仍超过 RoPE 表范围,则添加显式的运行时错误。
- 验证:
  - 本地用 `compile()`、Ray worker 容器内用 `compileall` 做语法检查。
  - 容器内针对失败形状的 shape 测试通过:
    - `cu_seqlens=[0, 98304, 196608, 294912]`
    - `cp_size=6`、伪 `cp_rank=5`、`local_seqlen=49152`
    - 生成的位置长度 `49152`,min `40960`,max `57343`
    - 索引后的 RoPE shape `(49152, 32)`
    - `apply_rotary_emb` 接受 `x.shape=(1, 49152, 1, 64)` 并返回相同 shape。
- 残留的语义注意点:
  - 该补丁修复了 RoPE 形状以及逐 token 的 RoPE 位置来源。
  - V4 sparse-attention 的 topk/KV 排序仍然沿用现有的 CP helper 假设;下一轮正确性审查应检查 packed zigzag KV gather/topk 索引,尤其是在信任新的拓扑改动之前。

### P1r:打过 RoPE 补丁后的 P1,形状已修复,因 OOM 失败

- Workload:`sft_kaynzhang_077_134k_smoke_no_offload_mem`
- 规模:`tp8pp7cp6ep8`
- 提交时间:2026-06-04 20:21 CST
- Job id:`raysubmit_4hWMANkpHG14ZcRA`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffMem-H200-20260604-202124`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/86tkdwa8`
- 状态:约于 2026-06-04 20:28 CST 失败
- 配置与 P1 相同:
  - `max_tokens_per_gpu=49152`
  - `recompute_granularity=selective`
  - CPU offload 禁用
  - `--no-save-optim`
- 启动/运行时校验:
  - `optimizer_cpu_offload=False`
  - `use_precision_aware_optimizer=False`
  - `recompute_granularity=selective`
  - `max_tokens_per_gpu=49152`
- 观测:
  - 全部 336 个 `MegatronTrainRayActor` actor 均达到 `ALIVE`;无 Ray pending demands。
  - Checkpoint 加载约于 20:26:35 CST 完成。
  - 模型与 optimizer 初始化时,大多数采样 rank 约 34.1GB GPU 内存,其中一个采样 rank 为 40.0GB。
  - Rollout 于 20:27:23 CST 采集了 128 个样本。
  - Rollout 指标:
    - response length 均值 `4719.26`
    - response length 最大值 `29050`
    - rollout 时间 `42.93s`
    - rollout tokens/GPU/s `187.62`
    - rollout 有效 tokens/GPU/s `41.88`
  - `train_wait` 耗时 `44.6s`。
  - `data_preprocess` 耗时 `0.8s`。
  - Actor 训练于 20:27:25 CST 开始。
  - 进入 actor 训练后未再出现 RoPE 形状错误;旧的 P1 失败点已通过。
- 失败情况:
  - Actor 训练在 `47.4s` 后失败。
  - 错误:`torch.OutOfMemoryError: CUDA out of memory`
  - 尝试分配:`12.94 GiB`
  - 日志中失败 GPU 状态:total `139.80 GiB`,free `12.34 GiB`,进程占用内存 `127.42 GiB`,PyTorch 已分配 `123.09 GiB`,PyTorch 已 reserved 但未分配 `201.55 MiB`。
  - 输出目录保持较小,约 `3.5M`;无需清理 checkpoint。
- Prometheus:
  - 失败附近的瞬时查询:最大 `DCGM_FI_DEV_FB_USED` 约 `124788 MiB`,平均约 `46761 MiB`。
  - 失败后的 8 分钟滚动查询:最大 `DCGM_FI_DEV_FB_USED` 约 `141025 MiB`,每 GPU 滚动最大值的平均约 `56139 MiB`。
  - 8 分钟滚动最大 GPU util `100%`,每 GPU 滚动最大值的平均 GPU util 约 `67.65%`。
- 解读:
  - 对于该失败配置,RoPE 形状 bug 已修复。
  - `49152 + selective` 在 H200 的 CP6 下不可行,因为它触及内存上限。
  - 可用的 no-offload 搜索空间位于 P1r/P1b selective 内存之下,很可能通过带 partial/block 分组的 full recompute,或通过降低每 rank 的 token 压力实现。

### P2:CP3 / DP2 拓扑探测,因吞吐为负而停止

- 新 scale 文件:`cluster/scale/tp8pp7cp3ep8.env`
- Workload:`sft_kaynzhang_077_134k_smoke_no_offload_cp3_full`
- 提交时间:2026-06-04 20:33 CST
- Job id:`raysubmit_C2JajppPqERxMNJL`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-cp3Full-H200-20260604-203350`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/uvrfqdap`
- 状态:于 2026-06-04 20:58 CST 手动停止,因为 actor 训练步在尚未完成时就已超过 CP6 基线时间。
- 目标:测试 CP3/DP2 在仍然占满全部 336 个 GPU 的前提下,是否是可行的吞吐拓扑。
- 并行配置:`TP=8`、`PP=7`、`CP=3`、`EP=8`、`ETP=1`、有效 `DP=2`
- Recompute:full、uniform、1 layer
- `max_tokens_per_gpu`:`49152`
- CPU offload:禁用
- 观测:
  - 全部 336 个 `MegatronTrainRayActor` actor 均达到 `ALIVE`;无 Ray pending demands。
  - Checkpoint 加载约于 20:38 CST 完成。
  - 模型与 optimizer 初始化时,采样 rank 上大约 `34-40GB` GPU 内存。
  - Rollout 于 20:39:55 CST 采集了 128 个样本。
  - Rollout 指标:
    - response length 均值 `4719.26`
    - response length 最大值 `29050`
    - rollout 时间 `44.97s`
    - rollout tokens/GPU/s `179.12`
    - rollout 有效 tokens/GPU/s `39.98`
  - `train_wait` 耗时 `48.1s`。
  - `data_preprocess` 耗时 `0.4s`。
  - Actor 训练于 20:39:56 CST 开始,到 20:57:57 CST 仍在运行。
  - 停止时,actor 训练已运行约 `1081s`,已经差于 R1 的 `752.889s`,因此最佳可能的 token 吞吐已经低于基线。
  - 训练期间各 rank 反复出现 `miles/backends/training_utils/loss.py:927` 的张量拷贝告警;这是日志噪声和潜在的 Python 开销,而非直接的停止原因。
- Prometheus:
  - 观测到的峰值最大 `DCGM_FI_DEV_FB_USED`:约 `66124 MiB`
  - 平均 `DCGM_FI_DEV_FB_USED`:约 `60.4GB`
  - 平均 `DCGM_FI_DEV_GPU_UTIL`:在后期训练窗口内约 `75.8%` 到 `93.7%` 之间波动。
  - 运行期间的 Ray 节点网络速率:平均发送/接收约 `1.1e9 B/s`,最大发送/接收约 `4.1e9 B/s`。
- 解读:
  - CP3/DP2 是稳定的,在观测窗口内未触发 RoPE 形状失败或 CUDA OOM。
  - 但它对该 workload 不是好的吞吐方向:在相同 batch size 下,它在完成前就已经超过了 CP6 基线的 actor 训练时间。
  - 单纯降低通信在这里没有用;后续探测应保持在 CP6/DP1,并在把内存压在 OOM 边界以下的同时降低 recompute 开销。

### P3:CP6 block recompute 吞吐探测

- Workload:`sft_kaynzhang_077_134k_smoke_no_offload_block4`
- 规模:`tp8pp7cp6ep8`
- 提交时间:2026-06-04 21:01 CST
- Job id:`raysubmit_xtVcwemdkHk8mamq`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffBlock4-H200-20260604-210121`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/kf7sarwk`
- 状态:于 2026-06-04 21:19 CST 成功
- 目标:通过降低 recompute 开销,在保持低于 selective-recompute OOM 边界的同时,把 `actor_train_tok_per_s` 提升到 R1 之上。
- 相对 R1 的改动:
  - `recompute_granularity=full`
  - `recompute_method=block`
  - `recompute_num_layers=4`
  - `max_tokens_per_gpu=32768`
  - CPU offload 仍然禁用
- 成功判据:
  - 必须在不 OOM 的前提下,超过 R1 的 `actor_train_tok_per_s=9493.726`,并/或缩短 R1 的 `actor_train_time=752.889s`。
- 观测:
  - Checkpoint 加载约于 21:06:32 CST 完成。
  - 模型与 optimizer 初始化时,采样 rank 上约 `34-40GB` GPU 内存,与 R1 相同。
  - Rollout 于 21:07:20 CST 采集了 128 个样本。
  - Rollout 指标:
    - response length 均值 `4719.26`
    - response length 最大值 `29050`
    - rollout 时间 `43.18s`
    - rollout tokens/GPU/s `186.51`
    - rollout 有效 tokens/GPU/s `41.63`
  - `train_wait` 耗时 `46.0s`。
  - `data_preprocess` 耗时 `0.8s`。
  - Actor 训练从 21:07:22 运行到 21:17:38 CST。
  - 训练指标:
    - `perf/update_weights_time`:`0.000026s`
    - `perf/data_preprocess_time`:`0.848s`
    - `perf/train_wait_time`:`45.638s`
    - `perf/actor_train_time`:`626.223s`
    - `perf/train_time`:`626.227s`
    - `perf/actor_train_tflops`:`23.337`
    - `perf/actor_train_tok_per_s`:`11414.023`
    - `perf/step_time`:`671.864s`
    - `perf/wait_time_ratio`:`0.0679`
  - 尽管设了 `--no-save-optim`,checkpoint 仍保存了两次;smoke checkpoint 大小为 `531G`,已删除。
  - W&B 拆解时在 atexit 中抛出 `ConnectionResetError: Connection lost`,与 R1 相同;Ray job 仍然成功。
- Prometheus:
  - 观测到的最大 `DCGM_FI_DEV_FB_USED`:`116852 MiB`
  - 每 GPU 滚动最大值的平均 `DCGM_FI_DEV_FB_USED`:`88562 MiB`
  - 最大滚动 GPU util:`100%`
  - 平均滚动 GPU util:`100%`
- 与 R1 对比:
  - `actor_train_tok_per_s`:`11414.023` vs `9493.726`,`+20.2%`
  - `actor_train_time`:`626.223s` vs `752.889s`,`-16.8%`
  - `step_time`:`671.864s` vs `800.091s`,`-16.0%`
  - `actor_train_tflops`:`23.337` vs `19.411`,`+20.2%`
  - 峰值 GPU 内存从约 `64GB` 上升到约 `117GB`,仍低于 P1b/P1r 约 `139-141GB` 的 OOM 边界。
- 解读:
  - `full/block/4` 是当前目标下第一个优于基线的配置。
  - 目标应继续保持 token 吞吐;该结果有价值,因为更高的内存利用率降低了 recompute 开销并提升了 tokens/s。
  - 在观测到的 selective-recompute OOM 边界之下仍有一些余量,因此下一步吞吐探测应在相同的 CP6/DP1 拓扑下尝试 `full/block/3`。

### P4:CP6 block3 recompute 吞吐探测

- 新 workload 文件:`cluster/workload/sft_kaynzhang_077_134k_smoke_no_offload_block3.env`
- Workload:`sft_kaynzhang_077_134k_smoke_no_offload_block3`
- 规模:`tp8pp7cp6ep8`
- 提交时间:2026-06-04 21:23 CST
- Job id:`raysubmit_5ksA9BaWTSyBayqp`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffBlock3-H200-20260604-212259`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/0z9ndwhf`
- 状态:于 2026-06-04 21:41 CST 成功
- 目标:相对 P3 再减少一档 recompute,测试 token 吞吐是否提升。
- 相对 P3 的改动:
  - `recompute_num_layers=3`
  - `max_tokens_per_gpu=32768`
  - CP6/DP1 拓扑不变
  - CPU offload 仍然禁用
- 观测:
  - Checkpoint 加载约于 21:28:16 CST 完成。
  - 模型与 optimizer 初始化时,采样 rank 上约 `34GB` GPU 内存。
  - Rollout 于 21:29:06 CST 采集了 128 个样本。
  - Rollout 指标:
    - response length 均值 `4719.26`
    - response length 最大值 `29050`
    - rollout 时间 `43.31s`
    - rollout tokens/GPU/s `185.98`
    - rollout 有效 tokens/GPU/s `41.51`
  - `train_wait` 耗时 `46.9s`。
  - `data_preprocess` 耗时 `0.8s`。
  - Actor 训练从 21:29:07 运行到 21:39:33 CST。
  - 训练指标:
    - `perf/update_weights_time`:`0.000031s`
    - `perf/data_preprocess_time`:`0.839s`
    - `perf/train_wait_time`:`46.259s`
    - `perf/actor_train_time`:`635.051s`
    - `perf/train_time`:`635.080s`
    - `perf/actor_train_tflops`:`23.012`
    - `perf/actor_train_tok_per_s`:`11255.360`
    - `perf/step_time`:`681.339s`
    - `perf/wait_time_ratio`:`0.0679`
  - 尽管设了 `--no-save-optim`,checkpoint 仍保存了两次;smoke checkpoint 大小为 `531G`,已删除。
  - W&B 拆解时在 atexit 中抛出 `ConnectionResetError: Connection lost`;Ray job 仍然成功。
- Prometheus:
  - 观测到的最大 `DCGM_FI_DEV_FB_USED`:约 `140330 MiB`
  - 每 GPU 滚动最大值的平均 `DCGM_FI_DEV_FB_USED`:约 `102170 MiB`
  - 训练窗口内的平均 GPU util:约 `99.4-99.6%`
- 对比:
  - 相对 R1,P4 仍然快得多:`11255.360` vs `9493.726` actor train tok/s,`+18.6%`。
  - 相对 P3,P4 略慢:`11255.360` vs `11414.023` actor train tok/s,`-1.4%`。
  - P4 比 P3 用的内存多得多:峰值约 `140GB` vs `117GB`,几乎没有 OOM 安全余量。
- 解读:
  - 在 block4 之后,更高的内存利用率不再有帮助;block3 处于吞吐/内存权衡的错误一侧。
  - 当前最佳单步配置为 P3:CP6/DP1、无 CPU offload、`full/block/4`、`max_tokens_per_gpu=32768`。
  - 下一步吞吐探测应保持 block4,并测试适度提高 `max_tokens_per_gpu` 是否能在不越过 OOM 边界的情况下改善动态 batch 的 packing。

### P5:CP6 block4 搭配 36864 token 上限

- 新 workload 文件:`cluster/workload/sft_kaynzhang_077_134k_smoke_no_offload_block4_tok36k.env`
- Workload:`sft_kaynzhang_077_134k_smoke_no_offload_block4_tok36k`
- 规模:`tp8pp7cp6ep8`
- 提交时间:2026-06-04 21:43 CST
- Job id:`raysubmit_LK7ELWtBKR3pEZ8F`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffBlock4Tok36k-H200-20260604-214329`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/l873om32`
- 状态:于 2026-06-04 22:10 CST 成功
- 目标:保持 P3 最佳 recompute 设置,提高动态 batch 的 token 上限,看 packing/token 吞吐是否改善。
- 相对 P3 的改动:
  - `max_tokens_per_gpu=36864`
  - `recompute_granularity=full`
  - `recompute_method=block`
  - `recompute_num_layers=4`
  - CP6/DP1 拓扑不变
  - CPU offload 仍然禁用
- 观测:
  - Checkpoint 加载约于 21:48:39 CST 完成。
  - Rollout 于 21:49:28 CST 采集了 128 个样本。
  - Rollout 指标:
    - response length 均值 `4719.26`
    - response length 最大值 `29050`
    - rollout 时间 `43.36s`
    - rollout tokens/GPU/s `185.75`
    - rollout 有效 tokens/GPU/s `41.46`
  - `train_wait` 耗时 `46.2s`。
  - `data_preprocess` 耗时 `0.8s`。
  - Actor 训练从 21:49:29 运行到 22:08:36 CST。
  - 训练指标:
    - `perf/update_weights_time`:`0.000027s`
    - `perf/data_preprocess_time`:`0.843s`
    - `perf/train_wait_time`:`45.565s`
    - `perf/actor_train_time`:`1156.597s`
    - `perf/train_time`:`1156.616s`
    - `perf/actor_train_tflops`:`12.635`
    - `perf/actor_train_tok_per_s`:`6179.957`
    - `perf/step_time`:`1202.182s`
    - `perf/wait_time_ratio`:`0.0379`
  - 尽管设了 `--no-save-optim`,checkpoint 仍保存了两次;smoke checkpoint 大小为 `531G`,已删除。
  - W&B 拆解时在 atexit 中抛出 `ConnectionResetError: Connection lost`;Ray job 仍然成功。
- Prometheus:
  - 观测到的最大 `DCGM_FI_DEV_FB_USED`:约 `127212 MiB`
  - 每 GPU 滚动最大值的平均 `DCGM_FI_DEV_FB_USED`:约 `95847 MiB`
- 对比:
  - 相对 P3,P5 慢得多:`6179.957` vs `11414.023` actor train tok/s,`-45.9%`。
  - P5 比 P3 用的内存略多,但远低于 P4 接近 OOM 的峰值;这次变慢并非由 CUDA OOM 边界导致。
- 解读:
  - 把 `max_tokens_per_gpu` 提到 `36864` 对该 workload 是明确的负面结果。它很可能以某种降低训练吞吐的方式改变了动态 batching/packing 或调度。
  - 当前最佳配置保持 `max_tokens_per_gpu=32768`。
  - 当前最佳仍为 P3:CP6/DP1、无 CPU offload、`full/block/4`、`max_tokens_per_gpu=32768`。

### P6:CP6 block4 搭配 VPP=2

- 新 scale 文件:`cluster/scale/tp8pp7cp6ep8_vpp2.env`
- Workload:`sft_kaynzhang_077_134k_smoke_no_offload_block4`
- 规模:`tp8pp7cp6ep8_vpp2`
- 提交时间:2026-06-04 22:25 CST
- Job id:`raysubmit_TU5w17JRP4fraGsP`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffBlock4-H200-20260604-222505`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/wscc6xeb`
- 状态:于 2026-06-04 22:50 CST 成功
- 目标:在不改变 TP/PP/CP/EP、batch size、token 上限或 CPU offload 的前提下,测试 interleaved pipeline 调度是否能比 P3 提升 token 吞吐。
- 相对 P3 的改动:
  - `virtual_pipeline_model_parallel_size=2`,由 14 阶段的 layout 推导得到。
  - `pipeline_model_parallel_layout=Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*4L`
  - 物理层归属与 P3 等价:7 个物理 PP rank 上为 `6/6/6/6/6/6/7` 层。
  - `recompute_granularity=full`、`recompute_method=block`、`recompute_num_layers=4` 不变。
  - `max_tokens_per_gpu=32768` 不变。
- 观测:
  - Megatron 日志中已启用 VPP:
    - `Number of virtual stages per pipeline stage: 2`
    - `virtual_pipeline_model_parallel_size=2`
  - Checkpoint 加载约于 22:30:27 CST 完成。
  - Rollout 于 22:31:22 CST 采集了 128 个样本。
  - Rollout 指标:
    - response length 均值 `4719.26`
    - response length 最大值 `29050`
    - rollout 时间 `48.10s`
    - rollout tokens/GPU/s `167.46`
    - rollout 有效 tokens/GPU/s `37.38`
  - `train_wait` 耗时 `52.4s`。
  - `data_preprocess` 耗时 `1.0s`。
  - Actor 训练从 22:31:24 运行到 22:49:41 CST。
  - 训练指标:
    - `perf/update_weights_time`:`0.000037s`
    - `perf/data_preprocess_time`:`0.842s`
    - `perf/train_wait_time`:`51.835s`
    - `perf/actor_train_time`:`1106.421s`
    - `perf/train_time`:`1106.472s`
    - `perf/actor_train_tflops`:`13.208`
    - `perf/actor_train_tok_per_s`:`6460.217`
    - `perf/step_time`:`1158.307s`
    - `perf/wait_time_ratio`:`0.0448`
  - 尽管设了 `--no-save-optim`,checkpoint 仍保存了两次;smoke checkpoint 大小为 `531G`,因文件归 `root` 所有,故从 Ray head 容器内部删除。
- Prometheus:
  - 观测到的最大 `DCGM_FI_DEV_FB_USED`:约 `71458 MiB`
  - 每 GPU 滚动最大值的平均 `DCGM_FI_DEV_FB_USED`:约 `66325 MiB`
  - 主训练窗口内的平均 GPU util 达到约 `99.6%`。
- 对比:
  - 相对 P3,P6 慢得多:`6460.217` vs `11414.023` actor train tok/s,`-43.4%`。
  - 相对 P3,actor 训练时间从 `626.223s` 增至 `1106.421s`,`+76.7%`。
  - 相对 P3,step 时间从 `671.864s` 增至 `1158.307s`,`+72.4%`。
  - VPP 确实降低了内存:峰值最大 GPU 内存从约 `116852 MiB` 降至 `71458 MiB`,但这对当前 token 吞吐目标没有用。
- 解读:
  - VPP=2 对当前从 P3 派生的配置是明确的负面结果。
  - 最可能的原因是 recompute 语义,而非通信饱和。在 P3 中,`recompute_num_layers=4` 作用于 6/7 层的物理 PP stage 内。在 P6 中,VPP 把每个物理 stage 拆成 3/4 层的虚拟 chunk,同时仍保持 `recompute_num_layers=4`,因此每个虚拟 chunk 接近被完全 recompute。这同时解释了两个观察:内存大幅降低、吞吐大幅降低。
  - VPP 还使 pipeline chunk 数量翻倍,增加了额外的 pipeline P2P 调度和 kernel 启动开销。在 `micro_batch_size=1`、`global_batch_size=128`、DP=1 下,microbatch 足够让 pipeline 保持繁忙,但额外开销不足以抵消多出来的 recompute 工作。
  - 层 layout 在物理上是平衡的,但虚拟 stage 边界与 P3 的 recompute 边界不完全相同。Stage 0 包含 embedding、Stage 13 包含 loss,因此对该 V4/MoE workload 而言,interleaved chunk 并非完全对称。
  - 当前最佳仍为 P3:CP6/DP1、VPP 关闭、无 CPU offload、`full/block/4`、`max_tokens_per_gpu=32768`。

### 当前规模下落地的默认配置

- 日期:2026-06-04 CST
- 当前默认启动现为:
  - `fleet=h200_k8s_42node`
  - `scale=tp8pp7cp6ep8`
  - `workload=sft_kaynzhang_077_134k_3epoch`
- 当未提供 `--fleet`、`--scale` 或 `--workload` 时,`run.sh` 默认使用此组合。
- 正式的 3epoch workload 现采用从 P3 派生的计算设置:
  - CPU offload 禁用
  - DeepEP 启用
  - router dtype `fp32`
  - attention 实现 `tilelang`
  - `HW_SEQ_LENGTH=134136`
  - `HW_MAX_TOKENS_PER_GPU=32768`
  - `HW_RECOMPUTE_GRANULARITY=full`
  - `HW_RECOMPUTE_METHOD=block`
  - `HW_RECOMPUTE_NUM_LAYERS=4`
- 在后续运行能在不降低稳定性余量的前提下超过 `actor_train_tok_per_s=11414.023` 之前,保持以 P3 为默认。

### 正式 3epoch 运行状态

- 提交时间:2026-06-04 23:08 CST
- Job id:`raysubmit_egpwCDb8TiFuPqYk`
- 输出目录:`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-3ep-H200-20260604-230840`
- W&B run URL:`https://wandb.ai/kaynzhang-none/v4-flash-post/runs/3xvn1ho2`
- 配置:
  - `fleet=h200_k8s_42node`
  - `scale=tp8pp7cp6ep8`
  - `workload=sft_kaynzhang_077_134k_3epoch`
  - `TP=8 PP=7 CP=6 EP=8 ETP=1`,VPP 关闭
  - 无 CPU offload、DeepEP 启用、router fp32
  - `full/block/4`、`max_tokens_per_gpu=32768`、`seq_length=134136`
- 首次 rollout:
  - 采集了 128 个样本
  - rollout 时间 `43.494s`
  - rollout tokens/GPU/s `185.177`
  - 有效 tokens/GPU/s `41.335`
- 首个训练步:
  - `perf/train_wait_time`:`46.853s`
  - `perf/actor_train_time`:`629.796s`
  - `perf/train_time`:`629.800s`
  - `perf/actor_train_tflops`:`23.204`
  - `perf/actor_train_tok_per_s`:`11349.262`
  - `perf/step_time`:`676.653s`
  - `perf/wait_time_ratio`:`0.0692`
- 首个训练步期间的 Prometheus:
  - 15 分钟最大 `DCGM_FI_DEV_FB_USED`:约 `116994 MiB`
  - 主计算窗口内平均 GPU util 达到约 `99.6%`
- 解读:
  - 正式运行在首步之后健康,且与 P3 smoke 结果高度一致。
  - 首步 token 吞吐相对 P3 smoke 为 `-0.6%`(`11349.262` vs `11414.023`),在预期的 run 间噪声范围内。

## 已定位根因(2026-06-05):NCCL 跑在 TCP 上,而非 EFA

R1-P6 期间约 2% MFU / 19-23 TFLOPS 的平台并不是 recompute/拓扑
问题。问题出在网络:miles 镜像
(`radixark/miles:sft-only-v4deps-20260603`) **未携带 aws-ofi-nccl plugin**,因此
NCCL 在所有跨节点 collective(CP ring-attention、EP all-to-all、PP P2P)上
静默回退到 **ENA 上的 TCP socket**。16 个 EFA NIC(约 400 GB/s)处于闲置。

在 live worker 上的证据:
- GPU `100% util` 但功耗仅约 `~120W`(H200 TDP 700W)= SM 在 comm 上空转,而非计算。
- 训练进程只映射了 `libnccl.so.2`;**没有 `libnccl-net.so` / `libfabric.so`**。
- ENA `enp7xs0` 承载约 11 GB/s;EFA `rdma_*_bytes` 计数器为 0。

2 节点 nccl 带宽(16 ranks),TCP vs EFA:
| collective (512MB) | TCP busBW | EFA busBW | speedup |
|---|---|---|---|
| all_reduce | 5.6 GB/s | 410 GB/s | 73x |
| all_to_all | 1.3 GB/s | 82.8 GB/s | 64x |

### 修复(无需重建镜像)
将 AWS efa-installer 的预编译 `libfabric 2.4` + `aws-ofi-nccl 1.19` + 匹配的
rdma-core(libefa EFA_1.4) + libhwloc15 暂存到共享 fsx 上的
`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/efa/root`,并用 `fi_info -p efa` 验证。
通过 fleet env(`V4_EFA_ENABLE=1`、`V4_EFA_ROOT`)接入 `run.sh`,注入
到 Ray runtime_env:`LD_LIBRARY_PATH`(efa libs 优先)、`NCCL_NET_PLUGIN`、
`FI_PROVIDER=efa`、`FI_EFA_USE_DEVICE_RDMA=1`、`FI_EFA_FORK_SAFE=1`、
`NCCL_PROTO=simple`、`NCCL_SOCKET_IFNAME=enp`(hostNetwork pod 的每节点 NIC
名为 enp74s0/enp75s0;前缀匹配可路由的 host NIC 用于 OOB bootstrap)。

### 结果(smoke,full/block/4,max_tokens_per_gpu=32768,CP6/PP7/EP8)
| metric | TCP (P3/formal) | EFA | gain |
|---|---|---|---|
| actor_train_tflops | 23.3 | **71.0** | 3.05x |
| MFU (vs 989) | 2.4% | **7.2%** | 3x |
| actor_train_time | 626s | **206s** | 3.04x |
| tok/s | 11414 | **34724** | 3.04x |

单 GPU H200 现已超过 H20 基线(约 53 TFLOPS)。正式的 3-epoch 运行已于
2026-06-05 开启 EFA 重新启动。

### 剩余的 MFU 余量(既然网络已不再是瓶颈)
EFA 把 MFU 从 2.4% 提升到 7.2%;新的上限是计算/wait,而非 comm。下一批杠杆:
1. `train_wait_time` 跳到约 97s(wait_ratio 0.32)——排查 rollout/data-load
   序列化,以及 `/ray_local` 占用 95% 的拖后腿节点 `10.3.22.244`。
2. 单 GPU 功耗峰值约 360W(而非约 700W):tilelang sparse-MLA attention + full/block/4
   recompute 是内存受限的。重新审视 recompute 缩减(既然 EFA 释放了 comm
   预算,早先的 OOM 权衡可能发生变化)以及长上下文拖后腿者(step-13)。
3. 重新测试拓扑(CP6 vs 更低 CP + DP)——早先"CP3 更差"的结论是在
   TCP 瓶颈下测得的,现在很可能反转。
4. FP8 MoE GEMM(需要 kernel 工作,见 hw/h200.env)。

### 持久性说明
EFA libs 存在 fsx 上,通过 env 注入;如果镜像被重建或 fsx 路径
改变,这会失效。长期方案:把 aws-ofi-nccl 烤进训练镜像。

## MFU 优化攻坚(EFA 之后,2026-06-05)

EFA 之后的基线:71 TFLOPS、7.2% MFU、actor_train_time 206s、wait_ratio 0.32。
目标:MFU >= 40%。所有探测均为单步 smoke(每次完整 ckpt 加载约 11min)。

| # | config | TFLOPS | MFU | train_time | note |
|---|--------|-------:|----:|-----------:|------|
| E0 | PP7CP6 full/block4 (EFA baseline) | 71.0 | 7.2% | 206s | reference |
| P1 | E0 + FP8 (TE blockwise MoE GEMM) | 71.2 | 7.2% | 205s | **无增益** —— 在 134K 下 MoE GEMM 并非瓶颈;attention+recompute 占主导。loss 9.84 健康。 |

来自 P1 的发现:FP8-on-MoE-GEMM 在这里毫无作用,因为在 134K 上下文下
expert GEMM 不在关键路径上(max_seq_len_mean 约 86K)。杠杆在于 recompute
缩减和每 rank 的 attention 工作(CP),最终落在 bf16 tilelang
sparse-MLA attention kernel 本身。转向更高 CP + 更轻 recompute 的探测。

### 攻坚继续 —— 内存上限与 recompute 发现(2026-06-05)

| # | config | result |
|---|--------|--------|
| P2 | CP7(PP6) + selective(moe_act,layernorm,mla_up_proj), tok20480 | **OOM 138GB** —— selective 内存太重 |
| P3 | CP6 + full/block/**2** + tok22528 | MoE router recompute 中的 **AssertionError**(`topk_routing_with_score_function: input_ids is not None and not requires_grad`)—— block/N<3 触发了一个 Megatron recompute bug(block3/4 可用) |

结构性发现:
- **激活内存 ≈ 43·tokens/(PP·CP) 在 PP·CP=42 下近似恒定**,因此 CP/PP 重新平衡并不会创造余量(P2 OOM 已证实)。更高 CP 只会降低每 rank 的 max_tokens 下限。
- **通过 block/N 进行的 recompute 缩减已经耗尽**:block3≈block4(P4 为 -1.4%),而 block<3 触发 router-recompute 断言(需要真正的代码修复)。selective/none 需要比 block 更多的内存。
- **MFU 公式把 attention 当作 dense O(seqlen²)** 计算(flops_utils.py:35-46),尽管 kernel 是 sparse 的(约 0.5%)。因此分子被幻影 attention 主导(在 L=86K 时约为 MoE 项的 28 倍)—— 该 workload 是 overhead 受限的,而非 FLOP 受限的;真实 FLOP 下限约 1s,因此只要移除 overhead(recompute+内存受限的 kernel+bubble),高 MFU 在物理上是可达成的。
- Pipeline bubble 在 tok32768 时约 16%(38 microbatch),在 tok22528 时约 11%(53 µb)。
- 已应用 ≤128K 数据过滤(albaliang_077_le128k.jsonl,49667 个样本);seq_length 从 134136→131208(对 CP∈{2,3,6,7,14} 可被 2·CP 整除)。

下一步:P4 = selective + **CPU-offload optimizer**(释放约 25GB 以清除 3GB 的 OOM 余量;selective 只 recompute moe_act/layernorm/mla_up_proj,**不** recompute router,因此避免了 block-recompute 断言)。

### 攻坚结论 —— 在 EFA 基线处配置空间已耗尽(2026-06-05)

| # | config | TFLOPS | MFU | note |
|---|--------|-------:|----:|------|
| E0 | PP7/CP6 block4 tok32768 (EFA) | 71.0 | 7.2% | config optimum |
| P1 | E0 + FP8 MoE | 71.2 | 7.2% | no gain (not GEMM-bound) |
| P2 | CP7 selective | OOM | — | mem |
| P3 | block2 | crash | — | router recompute bug (hash layers) |
| P4 | selective+offload | crash | — | CheckpointWithoutOutput backward bug |
| P5 | block4 **tok22528** | 14.1 | 1.4% | **差 4.5 倍** —— 更多 microbatch 把每 µb 的固定开销倍增了 |

结论:
- **max_tokens=32768 是一个尖锐的最优点**(tok22528 = -80%,tok36864 = -45%,见旧的 P5)。每 microbatch 的固定 overhead(V4 indexer/compressor + EP all-to-all)占主导;更少、更大的 microbatch 占优,直到内存/packing 崩掉为止。
- **Recompute 锁定在 block4**:block<3 触发 hash-router input_ids 断言(需要 Megatron-core 补丁把 input_ids 穿进 block-recompute 分支);selective moe_act/layernorm 触发 CheckpointWithoutOutput backward bug(只有 core_attn 是安全的,但它会保留 MoE 激活 → OOM)。即使有一个完美的 recompute 修复,也只能移除约 18%(recompute 约占整步的 18%)。
- **激活内存在 PP·CP=42 的各种切分下近似恒定**,因此拓扑重新平衡不会带来任何余量。
- 该步现在是计算受限的(计算期间 377W,而 EFA 之前为 118W),但 sparse-MLA/indexer/MoE kernel 是内存受限的(约 54% 的 TDP)。MFU 分子把 attention 当作 dense O(n²) 计算,而 kernel 是 sparse 的 → 达到 40% 在物理上可行,但需要 KERNEL 工作,而非 config。

**配置最优点 = E0(EFA + PP7/CP6/block4/tok32768/le128k)。** 通往 40% MFU 的路径是工程性的,按优先级排序:
1. 修复 Bug 1(把 input_ids 穿进 block-recompute)→ 启用 block/3-2 → 收益小。
2. 降低每 microbatch 的固定开销:优化 V4 indexer/compressor kernel(那个近似二次方的 indexer score+topk 在每个 C4 layer、每个 microbatch 都会运行)—— 很可能是最大的实时耗时点。
3. 为 H200 调优 tilelang sparse-MLA kernel(block size、bwd num_stages>0 的流水化;HW_TILELANG_* 旋钮目前是死的)。
4. (之后,按用户要求)FP8 attention。

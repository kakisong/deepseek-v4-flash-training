# V4-Flash SFT Profiler Capture Plan

> 自包含执行手册。新对话直接读这一份就能跑。
> 上下文承接:2026-05-11 讨论 — 训练 TFLOPS 和 MFU 都偏低,需要一份 profiler trace 来定位时间花在哪。

---

## 1. 目的与假设

### 现象

- 最近一次 64-GPU validation `stageBval-20260510-162322`(tilelang default,20 步)wall ~14 min 含 final ckpt save(211 s)。扣掉 save ≈ **29 s/step**。
- tilelang kernel 单测预测 ~6 s/iter,**~23 s 的差额来源未知**。
- 报表上的 **TFLOPS 和 MFU 都偏低**:两者同时低 ⇒ GPU 在空转,不是单纯"H20 算力弱"。

### 待证 / 待证伪的瓶颈假设(按怀疑度)

| # | 假设 | 预期占 wall |
|---|---|---|
| H1 | 跨节点 NCCL 通信(`NCCL_IB_DISABLE=1` 走 Ethernet)+ MoE EP=8 all-to-all | 30-50% |
| H2 | PP=8 pipeline bubble(micro-batch=1) | 15-25% |
| H3 | `recompute=full uniform num=1` 反向重算 | 15-30% |
| H4 | optimizer cpu-offload D2H/H2D 没完全 overlap | 5-15% |
| H5 | tilelang block size 没在 H20 上 tune(78 SM vs H100 132) | 5-10% |

**目的**:用 torch.profiler 抓 2 步 steady-state trace,判断 H1-H5 各占多少 wall,据此决定下一步动作(修 IB?降 PP?关 recompute?)。

---

## 2. 前置检查(跑之前)

```bash
# 集群健康
ssh root@10.0.8.17 "ray status" | head -10
# 期望:8 节点 ALIVE,64 GPU

# 没有遗留 ray job
ray job list
# 期望:无 RUNNING job

# anti-pollution 时间窗(12 min idle eviction)
# bring_up_cluster.sh 到 profile 提交别拖超过 10 min
```

如果集群没起,先:
```bash
bash examples/deepseek_v4_sft/cluster/bring_up_cluster.sh
```

---

## 3. 改动清单(3 个临时 patch,profile 完撤掉)

### 3.1 Patch `miles/utils/profile_utils.py:60` — 限 rank + 关重型选项

**为什么**:Miles 默认所有 64 rank 都写 trace(`profile_utils.py:71` 用 `torch.distributed.get_rank()` 区分文件名),64 文件 × ~1 GB = 60+ GB,加载 viewer 会炸。`record_shapes=True` + `profile_memory=True` 会让 step 慢 1.5-2× 扭曲 perf 数据。

把 `_create_torch_profiler` 整个函数替换为:

```python
def _create_torch_profiler(args, name):
    rank = torch.distributed.get_rank()
    # 只让 3 个代表 rank 出 trace
    #   rank  0: TP=0 PP=0(头,看 embed + 早期 layer)
    #   rank 32: PP=4 中间(看 1F1B 稳态 bubble)
    #   rank 56: PP=7 尾(看 loss + 反向起点,以前 NaN 出现处)
    SELECTED_RANKS = {0, 32, 56}
    if rank not in SELECTED_RANKS:
        # no-op profiler — schedule 永远在 wait,从不触发 trace
        return torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=10**9, warmup=0, active=0)
        )
    return torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=max(args.profile_step_start - 1, 0),
            warmup=1 if args.profile_step_start > 0 else 0,
            active=args.profile_step_end - args.profile_step_start,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            args.tensorboard_dir,
            worker_name=f"{name}_rank_{rank}",
            use_gzip=True,
        ),
        record_shapes=False,   # 减小 trace + 减少 overhead
        with_stack=True,       # 保留 — 看 Python 调用栈定位代码段
        profile_memory=False,  # 减小 trace + 减少 overhead
        with_flops=False,
    )
```

### 3.2 在 `examples/deepseek_v4_sft/cluster/run.sh` 的 `MISC_ARGS` 里注入 profile flag

文件位置 `cluster/run.sh:170-190`,在 `MISC_ARGS=(` 数组末尾、`)` 之前加四行:

```bash
  --use-pytorch-profiler
  --profile-step-start 5
  --profile-step-end 7
  --profile-target train_overall
```

**为什么 start=5 end=7**:跳过 step 0 的 JIT 编译(~120 s),从 steady-state 抓 2 个完整 step。

### 3.3 在 `RUNTIME_ENV` 里加 `TENSORBOARD_DIR`

run.sh:194 那段 RUNTIME_ENV 的 `env_vars` 里加一行:

```json
"TENSORBOARD_DIR": "$V4_OUT/profiler_traces/${RUN_ID}",
```

(`RUN_ID` 在 run.sh 上文已定义为 `stageBval-...`)

确保所有 rank 写到同一 CFS 目录,事后好捞。

---

## 4. 跑

```bash
# 一次性临时改 validation preset:跑 8 步而非 20 步,save-interval 抬高避免 ckpt save 进 trace 窗口
# (改 cluster/presets/validation.env)
export PRESET_NUM_ROLLOUT=8
export PRESET_SAVE_INTERVAL=1000

# 启动
bash examples/deepseek_v4_sft/cluster/run.sh validation
```

**预期 wall**:6-9 min(8 步 + JIT)。
**预期产出**:`$V4_OUT/profiler_traces/stageBval-<ts>/train_overall_rank_{0,32,56}.<pid>.pt.trace.json.gz`,3 个文件,各 200 MB - 1.5 GB。

跑完先确认:

```bash
ls -lh $V4_OUT/profiler_traces/stageBval-*/
# 期望:3 个 .json.gz 文件
```

---

## 5. 看 trace

### 工具

[https://ui.perfetto.dev](https://ui.perfetto.dev)(浏览器直接打开,支持 `.json.gz`)。把 rank 0 的 trace 文件下载到本地,拖进去即可。

如果在本地不方便,可以 `scp` 到任何能开 Chrome 的机器。**不要**起 http server 服务此目录。

### 看什么(读图清单)

对单个 steady-state step(profile 窗口中间那个),从下往上看:

1. **底部 `cuda HW` 这一行(GPU 真实执行流)**
   - 实心 = kernel 在跑,空白 = SM 空转
   - 算空白率 → 如果 >50% 空白,**GPU 在饿肚子**,瓶颈是上层调度/通信
   - 如果 <20% 空白,**GPU 喂得很满**,低 MFU 是真的 kernel 效率不够

2. **找 NCCL kernel**
   - 关键字:`ncclKernel_AllReduce`、`ncclKernel_AlltoAll`、`ncclKernel_SendRecv`、`ncclKernel_ReduceScatter`、`ncclKernel_AllGather`
   - 加总它们在一个 step 内的 wall 占比
   - **判定 H1**:NCCL > 25% wall → IB-disabled 是主税

3. **PP bubble:对比 rank 0 vs rank 56**
   - PP=0(rank 0)在 step 开头先做 forward,前 P-1 个 micro-batch 期间没有反向工作
   - PP=7(rank 56)反过来,最后才开始
   - **判定 H2**:某个 rank 的 cuda HW 空白集中段 > 15% wall → bubble 显著

4. **反向 vs 前向时间**
   - 标记:`autograd::engine::evaluate_function`(反向入口)vs `aten::*` 的 forward kernel
   - 比值 ≈ 1 → 没 recompute,正常
   - 比值 ≈ 2 → full recompute 双倍代价(**判定 H3**)

5. **D2H / H2D Memcpy 条带(copy 流)**
   - 标记:`Memcpy DtoH`、`Memcpy HtoD`
   - 如果尾部有连续大块 D2H → grad offload 到 CPU 的尖刺(**判定 H4**)
   - 期望:跟 compute 平行 overlap,看不到大的串行段

6. **Python 标注(with_stack=True)**
   - 鼠标悬停 kernel 上看 Python 调用栈
   - 注意 `tilelang_sparse_mla_*`、`apply_rotary_emb`、`_token_dispatcher` 这些 V4 关键 op 的 wall 占比
   - **判定 H5**:tilelang attention kernel 占 wall < 15% → attention 已经不是瓶颈,不要花时间 tune tilelang block size

---

## 6. 决策表

记 NCCL%、Bubble%、Bwd/Fwd 比、Tilelang%、Memcpy% 五个数,对照:

| 观察 | 结论 | 下一步动作 |
|---|---|---|
| NCCL > 30% | H1 坐实,IB-disabled 是主税 | **最高优先级**:协调集群 root 修 host IB 驱动,重开 NCCL_IB |
| NCCL 15-30% + PP idle > 15% | H1 + H2 同等贡献 | PP=8 → PP=4(5.H 已知可跑),同时排 IB 修复队 |
| Bwd/Fwd ≈ 2 + Bubble 不大 | H3 主导 | 关 full recompute 或切 selective(需先确认 PP=4 显存够) |
| Memcpy DtoH > 10% | H4 显现 | 评估改用 distributed optimizer / 取消 cpu-offload |
| Tilelang attention > 30% wall | H5 是真问题(罕见) | 在 H20 上 sweep `block_I / num_stages / threads` |
| 全都低,GPU 仍 50%+ 空白 | 不在上述假设里 | 看 launch overhead / Python GIL,可能要 nsys 二次定位 |
| GPU 空白 < 20% 但 TFLOPS 数字仍低 | **MFU 口径问题**,不是真瓶颈 | 检查 MFU 分母:active params 用对没?sparse attention 的 flops 折算? |

---

## 7. (可选)nsys 二次定位

如果 perfetto trace 显示 NCCL 时间高但**说不清是哪个 collective、用了多少带宽**,跑一次 nsys:

```bash
# 修改 run.sh 生成的 launch_in_container.sh,把 python3 换成 nsys 包一层
# 仅在 rank 0 节点上做,其他节点正常跑
nsys profile -t cuda,nvtx,cudnn,cublas,nccl \
  -o "$V4_OUT/profiler_traces/${RUN_ID}/nsys_rank0" \
  --capture-range cudaProfilerApi --capture-range-end stop \
  -f true \
  python3 train.py ...
```

并在训练代码里围绕 step 5-7 加:
```python
import torch
torch.cuda.cudart().cudaProfilerStart()  # step 5 开头
# ... 2 个 step ...
torch.cuda.cudart().cudaProfilerStop()   # step 7 结尾
```

输出 `.nsys-rep`,在本地装的 Nsight Systems GUI 打开。NCCL 时间线给每个 collective 的 byte 数 + 实测带宽,直接量化"被关掉的 IB 还了多少税"。

---

## 8. 跑完清理(必做)

trace 抓完后:

1. **撤 §3.1 patch**:`profile_utils.py` 还原(`git checkout miles/utils/profile_utils.py`)
2. **撤 §3.2 注入**:run.sh 移除 4 个 profile flag
3. **撤 §3.3**:RUNTIME_ENV 移除 TENSORBOARD_DIR(可选,无副作用但保持干净)
4. **撤 validation.env 临时改动**:`PRESET_NUM_ROLLOUT=20`,`PRESET_SAVE_INTERVAL=100`
5. **保留 trace 数据**:`$V4_OUT/profiler_traces/` 不要清,后续对比用
6. **把分析结果写入 STATUS.md 的 5.I 节**(目前 5.H 是 CP smoke,profiler 验证作为 5.I)

---

## 9. 接力清单(交给新对话)

新对话开场建议这样起:

> 我现在要按 `examples/deepseek_v4_sft/PROFILER_PLAN.md` 跑一次 profiler 验证。当前训练已结束,集群空闲。请按文档 §3 做 3 处 patch,§4 提交 job,跑完按 §5 读图,§6 给我决策建议。

或者更具体:

> profile 已经跑完,trace 在 `$V4_OUT/profiler_traces/stageBval-<ts>/`。请按 PROFILER_PLAN.md §5 + §6 帮我读图、出结论。

---

## 10. 风险清单

- **anti-pollution**:bring_up → submit profile job 之间别拖超 10 min,否则 worker 容器被回收(参 STATUS §3.4 + `project_v4sft_cluster_anti_pollution` memory)
- **trace 文件量**:3 rank × ~1 GB,确保 `$V4_OUT` 所在 CFS 有 5 GB+ 余量
- **profile 自身扰动**:即使关了 `record_shapes/profile_memory`,profile 仍会让 step 慢 ~10%。**不要拿 profile 跑的 step time 当 MFU 基准**,只用来看时间分布比例
- **NCCL hang 风险**:profile 加 `with_stack=True` 在某些 NCCL kernel 上可能影响 stream sync。若 8 步内出现 NCCL timeout,先把 with_stack=False 重跑

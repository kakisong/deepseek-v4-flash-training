# deepseek-v4-flash-training

DeepSeek V4-Flash SFT 训练任务的启动脚本与数据/ckpt 准备工具。模型代码和 miles 框架本身在 [kakisong/miles](https://github.com/kakisong/miles)（radixark/miles 的 fork，含 V4 plugin）。

## 目录

- `cluster/` — 容器集群拉起 + 训练任务提交脚本（fleet/scale/workload 三维分层 env）
- `tools/` — 数据预处理（jsonl → V4 dataset、ckpt 转换）
- `prepare_data.py` `prepare_megatron_ckpt.sh` — 一次性数据/权重准备
- `run_stage_*.sh` — 端到端 stage runner
- `WRITEUP.html` `PROFILER_PLAN.md` — 训练优化分析文档

## 部署

CFS 上需要两份代码：
```
$V4_WORK/miles/                       # kakisong/miles fork
$V4_WORK/deepseek-v4-flash-training/  # 本仓
```

`cluster/bring_up_cluster.sh` 在 master 上跑时会自动 git clone miles fork 到 CFS（如未存在）。

## 用法

```bash
cd cluster/
bash bring_up_cluster.sh
bash run.sh \
  --fleet h20_16node \
  --scale tp8pp8ep8_layout \
  --workload sft_prod
```

详见 `cluster/run.sh --help`。

## 开发期 override

```bash
export V4_MILES_REPO=/local/path/to/miles      # 用本地 editable miles
export V4_TRAINING_REPO=/local/path/to/this    # 用本地 editable 此仓
bash cluster/run.sh ...
```

# deepseek-v4-flash-training

DeepSeek V4-Flash SFT 训练任务的启动脚本与数据/ckpt 准备工具。模型代码和 miles 框架本身在 [kakisong/miles](https://github.com/kakisong/miles)（radixark/miles 的 fork，含 V4 plugin）。

## 目录

- `run.sh` — 训练任务提交主入口（control/fleet/scale/workload 组合）；前提是 Ray 集群和监控已准备好
- `cluster/` — Ray/容器/监控/Caddy 控制面工具，以及 control/fleet/scale/workload 配置
- `kuberay/` — KubeRay 迁移草案：RayCluster 模板、K8s 前置检查、迁移说明
- `smoke/` — Stage A/B、FP8 smoke 等端到端探测脚本
- `verification/` — loss mask、chat template、QA-fit、DeepSeek-V4 correctness 验证工具
- `manifests/` — 每次训练输入包的索引、版本和 checksum
- `tokenizer/encoding_dsv4.py` — DeepSeek-V4 Python chat template；`preflight` 会同步到 HF 模型目录
- `tools/` — 数据预处理、ckpt 转换、非验证类辅助工具
- `prepare_data.py` `prepare_megatron_ckpt.sh` — 一次性数据/权重准备
- `docs/` — 训练优化分析、评估计划、吞吐报告等文档

## 组件边界

训练应收敛到只依赖三类东西：miles 框架、训练镜像、每次训练输入包。其它内容只能作为控制面配置或输出目录，不能作为隐式代码依赖。

| 组件 | 放什么 | 不放什么 |
| --- | --- | --- |
| miles | 训练框架本体、V4 plugin、loss mask 调用逻辑、Megatron backend 适配 | 集群 IP、某次数据路径、临时 CFS patch |
| 训练镜像 | PyTorch/CUDA/TE/Ray/SGLang、FHT、V4 可用 Megatron-LM、运行期 Python/CUDA 依赖 | 每次训练数据、checkpoint 输出、集群现场配置 |
| 本仓 | cluster 启动脚本、fleet/scale/workload 配置、manifest、`tokenizer/encoding_dsv4.py`、数据/ckpt 准备工具 | 大模型权重、训练输出 checkpoint |
| 每次训练输入包 | SFT 数据、BF16 HF checkpoint、Megatron `torch_dist` 初始化 checkpoint、manifest | 框架代码、镜像里应有的二进制依赖 |
| control/env | 固定 Ray head/job server、Dashboard、监控、Caddy 入口 | 本次训练要占用几台机器 |
| fleet/env | 本次 job 的节点池、节点数/GPU 数、镜像、mount | `ray job submit` 的入口地址、代码版本选择、模型 tokenizer artifact |

`encoding_dsv4.py` 在本仓受控，路径固定为 `tokenizer/encoding_dsv4.py`。运行 `run.sh` 时，`cluster/lib/preflight.sh` 会把它同步到 `$V4_BF16_DIR/encoding/encoding_dsv4.py`，因为 miles 的 `deepseek_v4` loss mask 会从 HF checkpoint 的 `encoding/` 目录加载这个文件。

## 部署

CFS 上需要两份代码：
```
$V4_WORK/miles/                       # kakisong/miles fork
$V4_WORK/deepseek-v4-flash-training/  # 本仓
```

`cluster/bring_up_cluster.sh` 在 master 上跑时会自动 git clone miles fork 到 CFS（如未存在）。

生产镜像应内置 V4 可用 Megatron-LM 与 CUDA 扩展依赖；如果运行时还通过 `PYTHONPATH` 指向 `$V4_WORK/Megatron-LM` 或 `$V4_WORK/TileKernels`，说明该镜像还没有完成收敛，需要在 manifest 中标记为迁移中的外部依赖。

## 控制面准备

`run.sh` 只负责提交训练任务，不启动 Ray、不启动监控、不隐式重建容器。常规流程是先把控制面准备好，再提交训练：

```bash
# 1. 准备 Ray head/job server。该步骤只保证固定入口容器和 Ray dashboard 可用。
V4_CONTROL=current V4_FLEET=h20_16node bash cluster/prepare_ray_head.sh

# 2. 准备 Prometheus/Grafana。
V4_CONTROL=current V4_FLEET=h20_16node bash cluster/bring_up_monitoring.sh

# 3. 准备 Caddy 统一入口。
V4_CONTROL=current V4_FLEET=h20_16node bash cluster/bring_up_caddy.sh

# 4. 按需启动/复用 worker 容器并加入 Ray。
V4_CONTROL=current V4_FLEET=h20_16node bash cluster/ensure_ray_workers.sh

# 5. 提交前检查 Ray 容量和监控。Caddy 只是外部 Web 入口，不是训练前置条件。
V4_CONTROL=current V4_FLEET=h20_16node bash cluster/check_infra.sh

# 如需检查外部 Web 路由，再显式加 --with-caddy。
V4_CONTROL=current V4_FLEET=h20_16node bash cluster/check_infra.sh --with-caddy
```

训练结束后，如果希望保留 Ray head/dashboard 但释放 worker 容器：

```bash
V4_CONTROL=current V4_FLEET=h20_16node bash cluster/stop_ray_workers.sh
```

`cluster/bring_up_cluster.sh` 仍保留为全量重建工具；它会删除并重建所有节点的 `miles-v4-sft` 容器，适合清理环境，不适合作为每次训练提交的常规前置步骤。

## 本地临时盘

每台机器的主机侧 `/data0` 是本地 NVMe 数据盘，不是共享训练数据目录。训练容器内不会继续叫 `/data0`，而是挂载为 `/ray_local`：

```bash
/data0:/ray_local
```

Ray 的日志、临时文件和 object spill 放在容器内 `/ray_local/ray`。训练数据、模型和输出仍走共享目录 `/data_train/kaynzhang/v4-sft`。

## 控制面与节点池

`V4_CONTROL=current` 固定 `ray job submit` 的入口，也就是当前 Ray head/dashboard。`run.sh` 永远只通过这个入口提交任务，不用 `--fleet` 表达提交地址。

`V4_FLEET` 只表示本次 job 需要的节点池和容量：`V4_NUM_NODES`、`V4_NUM_GPUS_PER_NODE`、`V4_WORKER_IPS` 会决定 `ensure_ray_workers.sh` 加入哪些 worker，以及 `run.sh` 向 Miles 传入的 `--actor-num-nodes`/`--actor-num-gpus-per-node`。

当前 Miles 的 placement group 是按 GPU/CPU 资源申请，并不按 IP pin actor。因此，如果同一个 Ray 集群里有比 `V4_FLEET` 更多的 alive GPU 节点，Ray 可以把任务调度到任意可用节点。要严格控制本次训练使用哪几个 IP，当前做法是只让目标节点加入这个 Ray 集群；如果要在一个大 Ray 集群内精确指定 IP，需要后续在 Miles/Ray actor scheduling 上增加 node affinity 或 custom resource 约束。

## 训练提交

```bash
# 当前 H200 42-node 默认配置，P3-derived:
# fleet=h200_k8s_42node, scale=tp8pp7cp6ep8, workload=sft_kaynzhang_077_134k_3epoch
bash run.sh

# 等价显式写法
bash run.sh \
  --fleet h200_k8s_42node \
  --scale tp8pp7cp6ep8 \
  --workload sft_kaynzhang_077_134k_3epoch

# 4K SFT prod (winner: 5.23s/step, WRITEUP §4.18)
bash run.sh \
  --control current \
  --fleet h20_16node \
  --scale tp8pp16ep8_layout \
  --workload sft_prod

# 8 节点 fallback (cluster 缩水)
bash run.sh \
  --control current \
  --fleet h20_8node \
  --scale tp8pp8ep8_layout \
  --workload sft_prod_8node

# 32K agent SFT
bash run.sh --control current --fleet h20_16node --scale tp4pp16cp2ep8 --workload sft_albaliang

# 64K context
bash run.sh --control current --fleet h20_16node --scale tp2pp16cp4ep8 --workload sft_64k_pp16
```

详见 `./run.sh --help`。

## 开发期 override

```bash
export V4_MILES_REPO=/local/path/to/miles      # 用本地 editable miles
export V4_TRAINING_REPO=/local/path/to/this    # 用本地 editable 此仓
bash run.sh ...
```

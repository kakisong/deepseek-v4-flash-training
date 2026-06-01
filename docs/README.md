# deepseek-v4-flash-training

DeepSeek V4-Flash SFT 训练任务的启动脚本与数据/ckpt 准备工具。模型代码和 miles 框架本身在 [kakisong/miles](https://github.com/kakisong/miles)（radixark/miles 的 fork，含 V4 plugin）。

## 目录

- `run.sh` — 训练任务提交主入口（fleet/scale/workload 三维组合）
- `cluster/` — 容器集群拉起、tear down、Caddy、fleet/scale/workload 配置
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
| fleet/env | 节点 IP、端口、mount、Redis/proxy/Grafana 等现场配置 | 代码版本选择、模型 tokenizer artifact |

`encoding_dsv4.py` 在本仓受控，路径固定为 `tokenizer/encoding_dsv4.py`。运行 `run.sh` 时，`cluster/lib/preflight.sh` 会把它同步到 `$V4_BF16_DIR/encoding/encoding_dsv4.py`，因为 miles 的 `deepseek_v4` loss mask 会从 HF checkpoint 的 `encoding/` 目录加载这个文件。

## 部署

CFS 上需要两份代码：
```
$V4_WORK/miles/                       # kakisong/miles fork
$V4_WORK/deepseek-v4-flash-training/  # 本仓
```

`cluster/bring_up_cluster.sh` 在 master 上跑时会自动 git clone miles fork 到 CFS（如未存在）。

生产镜像应内置 V4 可用 Megatron-LM 与 CUDA 扩展依赖；如果运行时还通过 `PYTHONPATH` 指向 `$V4_WORK/Megatron-LM` 或 `$V4_WORK/TileKernels`，说明该镜像还没有完成收敛，需要在 manifest 中标记为迁移中的外部依赖。

## 用法

```bash
cd cluster/
bash bring_up_cluster.sh
cd ..
# 4K SFT prod (winner: 5.23s/step, WRITEUP §4.18)
bash run.sh \
  --fleet h20_16node \
  --scale tp8pp16ep8_layout \
  --workload sft_prod

# 8 节点 fallback (cluster 缩水)
bash run.sh \
  --fleet h20_8node \
  --scale tp8pp8ep8_layout \
  --workload sft_prod_8node

# 32K agent SFT
bash run.sh --fleet h20_16node --scale tp4pp16cp2ep8 --workload sft_albaliang

# 64K context
bash run.sh --fleet h20_16node --scale tp2pp16cp4ep8 --workload sft_64k_pp16
```

详见 `./run.sh --help`。

## 开发期 override

```bash
export V4_MILES_REPO=/local/path/to/miles      # 用本地 editable miles
export V4_TRAINING_REPO=/local/path/to/this    # 用本地 editable 此仓
bash run.sh ...
```

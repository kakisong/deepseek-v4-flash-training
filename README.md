# deepseek-v4-flash-training

DeepSeek V4-Flash SFT 训练的启动脚本与数据/ckpt 准备工具。模型代码与 miles 框架本身在
[kakisong/miles](https://github.com/kakisong/miles)（radixark/miles 的 fork，含 V4 plugin）。

组件边界 / 控制面准备 / 节点池语义等**完整参考**见 **[docs/readme.md](docs/readme.md)**；本文件只回答
一件事——**换了新数据，怎么发起一次训练**。

## 配置分层

```
bash run.sh --fleet <节点池> --scale <并行度> --workload <数据/超参>
```

- `--fleet` — 用哪些机器（节点池、节点数、镜像、mount）。如 `h200_k8s_40node`。
- `--scale` — 并行度 TP/PP/CP/EP。如 `tp8pp4cp5ep8`（DP 由 节点数÷并行度 自动得出）。
- `--workload` — **数据、初始权重、seq、轮数、超参**全在这里。换数据 = 换/改一个 workload。

hw 默认（seq/recompute）来自 `cluster/hw/h200.env`，可被 workload / 命令行覆盖；优先级
`fleet → base → hw → scale → workload → 命令行`。

## 发起一次训练（换新数据）

仓库提供两个**基线模板**，复制后只改顶部【必填】几行即可，已写死验证过的 PP/CP/seq/max-tokens/recompute：

| 模板 | 生产配置 | 命令 |
|---|---|---|
| `cluster/workload/sft_base_256k.env` | PP4 CP5 DP2 / 40 节点 / ~20.5% MFU | `bash run.sh --fleet h200_k8s_40node --scale tp8pp4cp5ep8 --workload <你的>` |
| `cluster/workload/sft_base_128k.env` | PP4 CP4 / 32 节点 DP2（或 16 节点 DP1） | `bash run.sh --fleet h200_k8s_32node --scale tp8pp4cp4ep8 --workload <你的>` |

为什么是这套并行度（per-rank 序列拐点 ~32K、MFU 上限 ~20% 由 sparse-MLA 原子内核封顶）见
[docs/h200_bottleneck_analysis.md](docs/h200_bottleneck_analysis.md)。

### 三步

```bash
# 1) 原始数据转 V4 messages 格式（已经是该格式则跳过）
#    --max-tokens 默认 32768 会丢长样本，长上下文务必调大
python3 tools/jsonl_to_v4_dataset.py --in 原始.jsonl --out $V4_DATA/新数据.jsonl --max-tokens 262144

# 2) 复制模板，改顶部【必填】4 行
cp cluster/workload/sft_base_256k.env cluster/workload/sft_新任务.env

# 3) 起训（先确认 Ray 集群在线，见下）
bash run.sh --fleet h200_k8s_40node --scale tp8pp4cp5ep8 --workload sft_新任务
```

模板顶部【必填】块（其余项已是验证过的生产值，一般不动）：

| 字段 | 含义 |
|---|---|
| `V4_SFT_DATA` | 你的 V4 jsonl 数据 → `--prompt-data` |
| `PRESET_RUN_ID_PREFIX` | 输出目录 / wandb group 前缀，**务必每次改**（否则难区分） |
| `PRESET_REF_LOAD_DIR` | 初始权重：底座重训=`$V4_TORCH_DIST`（= `models/DeepSeek-V4-Flash-FP8_torch_dist`）；续训=某 run 的 `.../checkpoints` |
| `PRESET_NUM_EPOCH` | 训练轮数 |

数据约束：`seq_length ≥ 最长样本` 且能被 `2*CP` 整除；单样本需 `≤ max_tokens_per_gpu × CP` 才能单条装下。

提交前可加 `--dry-run`：只生成 `outputs/<RUN_ID>/launch_in_container.sh` 并跑 preflight，不提交。

## 前提：Ray 集群在线

H200 是 **plain Kubernetes**（无 KubeRay operator）。从零拉起集群见
**[kuberay/h200-k8s-42node/](kuberay/h200-k8s-42node/README.md)**：

```bash
kuberay/h200-k8s-42node/bring_up.sh --dry-run   # 先校验 manifest
kuberay/h200-k8s-42node/bring_up.sh             # 起 head→workers，校验 43 节点 / 336 GPU
```

`run.sh`（fleet 为 `h200_k8s_*` 时 `V4_SUBMIT_MODE=k8s`）会 `kubectl exec` 进 head pod 用
`ray job submit` 提交。H20 老集群走 ssh/docker，控制面准备流程见 [docs/readme.md](docs/readme.md)。

## 文档

- [docs/readme.md](docs/readme.md) — 组件边界、控制面准备、节点池语义（完整参考）
- [docs/h200_bottleneck_analysis.md](docs/h200_bottleneck_analysis.md) — MFU / 瓶颈分析，生产配置取舍
- [docs/](docs/) — SFT loss 校验、tokenization、held-out 评估、ckpt→FP8/FP4 等

更多命令行开关见 `./run.sh --help`。

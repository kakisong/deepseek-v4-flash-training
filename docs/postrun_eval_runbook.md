# 运行后评估手册 — V4-Flash SFT (Kaynzhang 077, 3-epoch)

**目的。** 当前的在线 SFT run **没有 held-out 验证 loss**,也**没有下游 eval**(原因见
下文)。本手册给出了一套分阶段、每步一条命令的流程,用于在 **run 结束并释放其 336 个 GPU 之后**验证训练好的模型。
这里的所有内容都是在 run 进行期间以纯 CPU 方式准备的(选项 "D");其中的 GPU 步骤需等待 run 完成。

## 为什么没有在线 eval(已验证)
- `miles/rollout/sft_rollout.py:13` 硬编码了 `assert not evaluation` → miles 的 eval 路径永远不会
  计算 SFT 的 cross-entropy;对于 SFT,它只会做 RL-reward 生成。该 run 也没有设置任何
  `--eval-interval` / `--eval-prompt-data`。
- 全部 **49,667** 条 `albaliang_077_le128k.jsonl` 样本都已被训练(3 个 epoch);没有保留任何 held-out 划分。
  (`le128k ⊂ le134k` 已验证;le134k 中多出来的 332 行是唯一未见过的同域数据。)
- HF checkpoint 目录中**没有 transformers 建模代码**(`config.json` 的 `auto_map=None`、
  `architectures=["DeepseekV4ForCausalLM"]`、`model_type=deepseek_v4` — 上游
  transformers 并不认识)。因此无法直接使用 `AutoModelForCausalLM` 做前向;打分需要通过
  Megatron 或一个 SGLang server 完成。

## Held-out 数据集(已切分,纯 CPU)
`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/data/holdout_eval/`
- **`holdout_albaliang_077_332.jsonl`** — 332 条**同域、真正未见过**的样本(即 le134k∖le128k
  的差集;agent/tool-use、多轮)。**实测**的 deepseek_v4 token 长度(全部 332 条):最小 54.6K、
  p50 108.4K、p90 117.7K、**最大 130.6K → 0/332 超过 128K 窗口**,因此**不会触发任何截断**,
  它们正好落在训练长度分布之内。(导致它们被过滤掉的 metadata `token_length`(131–134K)相比生产环境的 tokenizer
  多计了约 20K — 这正是它们被从 le128k 中丢弃的唯一原因。)总计 1,619,525 个 assistant loss token → 可给出稳定的 CE 估计。
  这是检验 "epoch 2/3 是否对任务过拟合?" 的最佳探针。
- **`holdout_openhermes_512.jsonl`** — 512 条**跨域、真正未见过**的样本(OpenHermes 通用
  对话,与训练数据 0% 重叠,≤4K token)。用作 "SFT 是否导致了对通用指令遵循能力的灾难性遗忘?" 的探针。

## Checkpoints(已在磁盘上验证)
`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-3ep-H200-20260606-172101/checkpoints/`
- 格式:`torch_dist`(`.metadata` + `__<rank>_<n>.distcp`),每个 iter 约 3.7 T(含 optimizer state)。
- 保留策略为 `{latest, every-multiple-of-500}`。**最终**的会是 `iter_0001164`(最后一步)外加
  `iter_0001000`。**在启动任何会修剪它的操作之前,先抓取 `iter_0001164`(或 `iter_0001000`)。**

---

## Step 0 — 等待 run 结束,选定 checkpoint
```bash
CKROOT=/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-3ep-H200-20260606-172101/checkpoints
ls -1dt $CKROOT/iter_*          # confirm iter_0001164 exists; cat $CKROOT/latest_checkpointed_iteration.txt
ITER=$CKROOT/iter_0001164
```

## Step 1 — 将 torch_dist → HF 转换(纯 CPU,不用 GPU)
`convert_torch_dist_to_hf.py` 使用 `no_dist=True`(单进程 CPU 加载 `common.pt` + distcp
分片 → safetensors)。需要一个大内存节点(写入期间模型参数约 568 GB 保存在 CPU state_dict 中)。
```bash
MILES=/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles
HFREF=/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-bf16-unpacked
OUT=/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/.../hf_iter_0001164      # pick a path on fsx

python3 $MILES/tools/convert_torch_dist_to_hf.py \
  --input-dir   $ITER \
  --output-dir  $OUT \
  --origin-hf-dir $HFREF \
  --vocab-size  129280 \
  --force
# copies tokenizer/config from --origin-hf-dir; --vocab-size strips logit padding.
```

## Step 2 — 用 SGLang 部署服务(8×H200,一个 BF16 副本)
`tools/launch_sglang.sh` 默认 `--context-length 32768`,**对 albaliang held-out 来说太短了
(~110–118K)**。请使用带长上下文的显式启动方式:
```bash
python3 -m sglang.launch_server \
  --model-path $OUT --tp 8 --trust-remote-code \
  --attention-backend triton \
  --context-length 131072 \
  --mem-fraction-static 0.85 --disable-cuda-graph \
  --host 0.0.0.0 --port 30000
# DSV4 attention may be unsupported on sglang upstream -> triton backend is the safe path (per launch_sglang.sh).
# If 131072 ctx OOMs at tp8, either raise tp / lower --mem-fraction-static, or eval openhermes only at 8192.
```

## Step 3 (B) — held-out cross-entropy(廉价的健全性检查,约 20 分钟)
```bash
TR=/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/deepseek-v4-flash-training
DATA=/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/data/holdout_eval

# same-domain (long context):
python3 $TR/tools/eval_holdout_ce.py \
  --data $DATA/holdout_albaliang_077_332.jsonl --hf-tokenizer $HFREF \
  --sglang-url http://127.0.0.1:30000 --max-len 131072 --out /tmp/ce_albaliang.json
# cross-domain (short):
python3 $TR/tools/eval_holdout_ce.py \
  --data $DATA/holdout_openhermes_512.jsonl --hf-tokenizer $HFREF \
  --sglang-url http://127.0.0.1:30000 --max-len 8192 --out /tmp/ce_openhermes.json
```
**解读**
- albaliang held-out CE ≈ 在线 `train/loss`(~2.0–2.2) → epoch 2/3 **没有**过拟合;~2.2
  的平台期是真实的数据下限,而非记忆。
- albaliang held-out CE ≫ train/loss → 过拟合;更早的 checkpoint(例如 `iter_0000500`,
  epoch 1 结束)可能泛化得更好。
- openhermes CE 相比 base/更早的 iter 保持稳定 → 没有灾难性遗忘。
- **最佳信号**:在**两个** checkpoint(`iter_0000500`,epoch1 结束;以及 `iter_0001164`,
  最终)上运行 Step 1–3 并对比 — 如果 held-out CE 在两者间持平而 train loss 下降了,那么
  额外的 epoch 对泛化毫无帮助(这与观察到的 train-loss 平台期一致)。

> `eval_holdout_ce.py` 复用了**已验证**的 `MultiTurnLossMaskGenerator`(掩码逐位一致,见
> `tools/verify_sft_pipeline.py`)。SGLang 的 `input_token_logprobs` 管线遵循
> `miles/rollout/sglang_rollout.py` 的约定,但**未做端到端验证**(run 期间没有空闲 GPU):
> 在收到第一个 response 时,先断言 `len(input_token_logprobs)==len(input_ids)`,确认无误后再信任聚合结果。

## Step 3 (A) — 下游 / 生成式 eval(更有意义的那个)
对于 agent/tool-use 的 SFT 来说,任务准确率比 CE 更重要。评测 harness 位于
`miles/examples/eval/`,并与 SGLang 的 OpenAI endpoint 通信:
- **Terminal Bench**(agent/tool-use,代码任务):`examples/eval/terminal_bench/tb_server.py` +
  `tb_client.py`;配置模板 `examples/eval/scripts/eval_tb_example.yaml`。
- **NeMo Skills**(AIME25 / Arena-Hard / HLE):`examples/eval/nemo_skills/skills_server.py`。
- 两者都设置 `api_base=http://127.0.0.1:30000/v1`、`model_name=<your model>`;对 tool-use 类 eval
  在 eval 数据集配置中传入 `tool_key: tools`(`miles/utils/eval_config.py`)。
完整的委托示例见 `examples/eval/scripts/run-eval-tb-qwen.sh`。请独立运行(不要放在
训练循环里),以避免 eval 超时阻塞任何东西。

---
### 出处
于 2026-06-07 在线 run 期间准备(纯 CPU、零 GPU、不干扰训练)。
Run:`stageKaynzhang077-134K-3ep-H200-20260606-172101`,submission `raysubmit_L2dbLAFDyHjRbDPM`。
产物:`data/holdout_eval/*.jsonl`、`tools/eval_holdout_ce.py`、本手册。
相关文档:`docs/sft_loss_verification.md`(loss 正确性)、`docs/h200_bottleneck_analysis.md`。

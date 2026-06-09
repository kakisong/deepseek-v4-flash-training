# SFT Loss:如何计算、如何验证,以及一个有误导性的指标

**TL;DR(2026-06-07 通过精确批次复现 + 代码追踪验证):**
V4-Flash 的 SFT `train/loss` 计算是**正确的**——只对**助手回复 token**做 per-token 交叉熵,
带有正确的 next-token 移位、per-token 归一化以及 CP-aware 聚合。唯一表面上的异常——离线
loss-token 比值为 **0.076**,而 wandb 上的 `train/batch/loss_token_ratio` 为 **0.195**——**并不是 loss bug**:
那个 wandb 指标是一个 **CP-local × cp_size 的近似,对 end-loaded 的 SFT 会高估约 2×**。实际的 loss 并不
使用它。**请信任 `train/loss`,而非 `loss_token_ratio`。**

验证工具:`tools/verify_sft_loss_mask.py`、`tools/verify_sft_pipeline.py`。

---

## 1. loss 是如何计算的(真实路径)

运行配置:`--loss-type sft_loss --calculate-per-token-loss --loss-mask-type deepseek_v4`,
TP8/PP7/**CP6**/EP8,`--qkv-format thd`,vocab 129,280。

1. **Mask = 仅助手 token。** `sft_rollout.generate_rollout` →
   `MultiTurnLossMaskGenerator.get_loss_mask` → `gen_multi_turn_loss_mask_deepseek_v4`
   (`miles/utils/mask_utils.py:135`)。它用 V4 chat template
   (`encoding/encoding_dsv4.py`)渲染对话,带 offset-mapping 进行 tokenize,并**仅**对那些
   char-offset 落在某个 `assistant` 消息区间内的 token 设置 `loss_mask[k]=1`(第 203 行:
   `if role != "assistant": continue`)。系统/用户/工具消息以及特殊/过渡 token
   (`<｜Assistant｜>`、BOS……)保持为 `0`。每轮的 `step_loss_mask=0` 会把该轮清零。
2. **Next-token 移位**(`loss.py:88-93`):`logits_chunk = logits[start-1:end-1]`,
   `tokens_chunk = tokens[-response_length:]` → `logits[t-1]` 预测 `token[t]`(标准的 shift-by-one)。
3. **Per-token 归一化**(`calculate_per_token_loss=True`):reducer 为 `sum_of_token`
   (`cp_utils.py:122`),因此 `loss = -Σ logP` 是对回复 token 求和;Megatron 把累加的 loss
   除以 `num_tokens = Σ loss_mask.sum()`(`loss.py:869`,即**真实**的 mask),覆盖所有 microbatch 以及
   DP/CP rank → 得到真正的 per-token 平均交叉熵(nats)。随机基线 ≈ ln(129280) ≈ 11.8。
4. **CP-aware**(`get_sum_of_sample_mean` 的 CP 分支,`cp_utils.py:86-120`):zigzag-CP 的 loss mask 会
   按 rank 分块并在 CP 上求和;loss 归一化器会正确地聚合 CP-local 计数。

## 2. 它是如何验证的(实证,而非凭信任)

- **真实样本上的 Mask**(`verify_sft_loss_mask.py`,60 个样本):每个样本的系统提示词都被
  mask 掉(`lead_masked=True` 60/60);解码出的 `mask=1` 区间是助手的 reasoning/content/tool_calls;
  `mask=0` 区间是系统提示词 + 用户的 tool_results。每个样本的 loss 比值范围在 0.001–0.82
  (随轨迹中助手文本相对于上下文/工具输出的占比而变化)。
- **精确批次复现**(`verify_sft_pipeline.py`):复刻训练数据 pipeline
  (`apply_chat_template=False` → `_build_messages` 原样返回原始消息;用
  `random.seed(rollout_seed=42 + epoch 0)` 做 shuffle),取与 live rollout 0 **相同的 128 个样本**,运行
  真实的 mask。结果:**总 token 数 6,638,138 = live step-0,逐字节一致(diff +0)**;真实 loss token 数
  **502,530** → 真实比值 **0.0757**。

## 3. 那个有误导性的指标:`train/batch/loss_token_ratio`

wandb 指标报告的是 **0.195**,而非 0.076。根本原因(`megatron_utils/model.py:65-71`):

```python
# tokens/full_loss_masks are CP-local after get_batch(); multiply by CP for a
# rollout-level approximation ...
local_loss_tokens = int(batch["full_loss_masks"].sum().item())
loss_tokens = local_loss_tokens * cp_size          # <-- the approximation
...
loss_token_ratio = loss_tokens / total_tokens      # total_tokens is exact (sum of total_lengths)
```

- `full_loss_masks` 是 mask 的 **CP-local 分块**(`training_utils/data.py:338,357,363`),在
  单个 rank 上求和,然后 **× cp_size(=6)** 以*估计*全局计数。
- 该估计假设 loss token 在 6 个 zigzag-CP 分块上是**均匀**分布的。但对 SFT 而言它们是
  **end-loaded** 的(长 prompt + 工具输出,随后是一段简短的助手回复),所以被测量 rank 的分块持有约
  2.14× 的均匀份额 → `502,530` 变成了 `1,076,592`。
- 特征:该指标会逐 step 抖动(0.16–0.20),并稳定地比真实的 0.076–0.09 高出约 2×。

**为什么这无关紧要:** `full_loss_masks` 及其 `× cp_size` 计数**仅**出现在诊断日志记录器
`_collect_train_batch_debug` 中。实际的 loss + 梯度使用的是 `batch["loss_masks"]`(真实的 mask),
并带有正确的跨 rank 聚合。因此 **loss 值是正确的**;只有这一个被记录的比值被夸大了。

**建议:** 从 **`train/loss`**(平滑下降、无 NaN/spike)以及 epoch 边界来判断训练健康状况——
而不是 `loss_token_ratio`。可选修复:用真正的跨 rank `all_reduce(full_loss_masks.sum())` 替换
`local × cp_size`,使该诊断与真实计数一致。

## 附录 —— 可复现性
- `tools/verify_sft_loss_mask.py [n] [data]` —— 真实样本上的 mask(已训练 vs 被 mask 的区间、比值)。
- `tools/verify_sft_pipeline.py` —— 对 rollout-0 的精确复现,对比 live step-0 的 token 计数。
- 两者:在 GPU pod 中运行(CPU-only,不会干扰训练),`PYTHONPATH=<fsx miles>`,HF ckpt
  `models/DeepSeek-V4-Flash-bf16-unpacked`,数据 `data/albaliang_077_le128k.jsonl`。
- 验证时的 live run:`stageKaynzhang077-134K-3ep-H200-20260606-172101`(submission
  `raysubmit_L2dbLAFDyHjRbDPM`),wandb `v4-flash-post`。

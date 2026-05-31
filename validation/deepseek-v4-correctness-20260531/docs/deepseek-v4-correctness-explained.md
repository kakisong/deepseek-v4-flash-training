# DeepSeek-V4 训练正确性说明

本文面向**不是大模型训练专业背景**、但需要判断这次验证是否可信的人。

它不从算子细节开始，而是先回答三个问题：

1. 为什么不能只看训练能跑、loss 下降？
2. 为什么要把证明拆成很多层？
3. 我们现在的数据到底证明了什么，还没证明什么？

一句话结论：

**Miles 当前 DeepSeek-V4 训练路径没有复现 Megatron-LM PR #4839 中的 HyperConnection 方向错误；核心训练数学、SFT loss、MoE、attention、optimizer、真实 EP=8 MoELayer、loaded 4-layer mini checkpoint 训练链路，都已经在声明的 BF16 训练容差下通过验证。我们没有把 strict parity 失败包装成通过；剩余 strict 差异被记录为边界。**

## 先看结论

当前判断分三层：

| 层级 | 当前结论 | 含义 |
| --- | --- | --- |
| 已证明 | `PASS` | 关键公式、核心算子、SFT loss、训练一步、MoE、optimizer、mini checkpoint gate 都有机器验证证据。 |
| 可接受边界 | BF16 tolerance `PASS` | 真实训练 runtime 与 reference 之间存在 BF16 数值差异，但差异被阈值约束，且来源已定位。 |
| 未严格证明 | strict parity `FAIL` / `FAIL_DIAGNOSTIC` | 不能宣称完整真实 forward 与 reference bitwise 或 strict logprob 完全一致；这些失败保留在文档和 proof ledger 中。 |

换句话说：

- 可以说：**当前 Miles DeepSeek-V4 训练方式在我们声明的 BF16 训练正确性标准下是可信的。**
- 不应该说：**完整真实训练 forward 已经和 reference 做到了 bitwise 或 strict logprob 完全一致。**

## 为什么要这样证明

DeepSeek-V4 训练正确性不是一个“能跑就行”的问题。

原因有三个。

第一，原始风险是**公式方向错误**。Megatron-LM PR #4839 修复的是 HyperConnection / mHC 中 residual mixing 的矩阵方向问题。这个问题不会必然导致训练立刻崩溃，loss 也可能继续下降，但模型实际训练的是错误公式。因此必须用数学 reference 检查公式方向，不能只观察训练曲线。

第二，DeepSeek-V4 使用大量低精度和高性能 kernel，例如 BF16、QAT、sparse attention、TileLang、MoE all-to-all。不同 kernel 的浮点计算顺序不同，结果不一定 bitwise 一样。这里需要判断的是：差异是否在可解释、可重复、可接受的范围内，而不是简单要求所有路径完全相同。

第三，MoE routing 有离散分支。连续数值只要发生很小变化，top-k routing 就可能选择不同 expert。一旦选择不同 expert，后面输出会被放大。这类问题需要先固定离散选择，再判断连续数学是否正确。这就是 routing replay 的意义。

所以证明路线不能只做一个端到端对比。端到端失败时，我们还要知道失败来自哪里；端到端通过时，我们也要知道它有没有掩盖局部错误。

## 我们要证明什么

我们把“训练正确”拆成五个可检查的问题：

| 问题 | 为什么重要 | 验证方式 |
| --- | --- | --- |
| 公式有没有写错？ | 公式方向错会训练错误目标。 | 用显式 PyTorch / fixed formula reference 对比。 |
| 核心算子是否正确？ | attention、QAT、MoE、optimizer 都可能引入精度或梯度错误。 | 单独验证 forward、backward、update。 |
| loss 目标是否正确？ | SFT 训练最终优化的是 loss；loss 错则训练目标错。 | 用 `log_softmax + gather + loss_mask` 显式公式对比。 |
| 真实 checkpoint 是否能闭合？ | 小算子正确不等于真实 checkpoint 串起来正确。 | 用 loaded 4-layer mini checkpoint 和固定 SFT batch 做验证。 |
| 失败项是否被诚实记录？ | 不能把 strict 失败写成通过。 | proof coverage matrix 和 proof ledger 机器校验。 |

这种拆法的核心逻辑是：**先证明尺子是对的，再证明零件是对的，再证明装配后的系统在真实输入上符合预期。**

## Step 1：验证 Megatron PR #4839 风险点

PR #4839 关注的是 HyperConnection residual mixing 的方向。简化后，正确公式是：

```text
H_res.T @ residual
```

错误方向是：

```text
H_res @ residual
```

如果 `H_res` 不是对称矩阵，这两者会明显不同。

验证数据来自：

- `artifacts/deepseek-v4-operator-math-20260531.json`

结果：

| 检查项 | 结果 |
| --- | ---: |
| Miles vs PR #4839 fixed native formula max diff | `2.384185791015625e-07` |
| Miles vs wrong direction max diff | `9.81311321258545` |
| status | `PASS` |

解释：

- `2.38e-7` 是浮点舍入量级，说明 Miles 当前公式和 PR #4839 修复后的方向一致。
- `9.81` 是非常大的差异，说明这个测试能区分正确方向和错误方向。
- 因此，**PR #4839 的 HC 方向错误没有在 Miles 当前 DeepSeek-V4 路径复现。**

## Step 2：验证基础算子

基础算子是训练链路的底座。这里主要看四类风险：

1. QAT 是否和 official 语义一致。
2. Dense / sparse attention 是否一致。
3. TileLang sparse attention 是否一致。
4. 全 mask 行、backward gradient 是否稳定。

验证数据来自：

- `artifacts/deepseek-v4-operator-math-20260531.json`

关键结果：

| 检查项 | 结果 |
| --- | ---: |
| official-compatible KV QAT max_abs | `0.0` |
| official-compatible KV QAT exact_equal | `true` |
| dense vs sparse attention forward max_abs | `0.0078125` |
| dense vs sparse attention forward relative diff | `2.197893374189519e-06` |
| TileLang sparse attention max forward max_abs | `0.004327297210693359` |
| TileLang sparse attention max forward relative diff | `2.175237100887628e-06` |
| fully masked row max_abs | `0.0` |
| gradient finite | `true` |

解释：

- QAT 完全一致，说明前面发现的 QAT 语义问题已经被修复并验证。
- attention 的差异在 BF16 舍入范围内。
- fully masked row 没有产生 NaN。
- 这说明底层 attention/QAT 风险已经有直接证据覆盖。

## Step 3：验证 SFT loss 目标

SFT 训练优化的是 token loss。如果 loss 公式错，模型可以正常训练，但训练目标就是错的。

我们没有用 Miles 自己的 loss helper 当 reference，而是手写最朴素的 PyTorch 公式：

```text
sum(-log_softmax(response_logits)[target_token] * loss_mask)
```

验证数据来自：

- `artifacts/deepseek-v4-sft-loss-reference-20260531.json`
- `artifacts/deepseek-v4-sft-loss-train-reference-20260531.json`

forward/loss 结果：

| 检查项 | 结果 |
| --- | ---: |
| status | `PASS` |
| loss_abs_global_max | `0.0` |
| token_count_abs_global_max | `0.0` |
| logprob_max_abs_global_max | `1.9073486328125e-06` |
| logprob_mean_abs_global_max | `1.2926849990435585e-07` |
| checked logprob tokens | `3216` |

backward/update 结果：

| 检查项 | 结果 |
| --- | ---: |
| status | `PASS` |
| loss_abs_global_max | `0.00048828125` |
| selected_grad_max_abs | `0.009491967037320137` |
| selected_grad_max_rel_gap | `0.006301201940482904` |
| selected_state_max_abs | `9.531504474580288e-10` |
| selected tensors with gradient | `272` |

解释：

- loss 数值完全一致，token 数也完全一致。
- backward/update 的差异在 BF16/fused CE 容差内。
- 因此，**Miles 当前 SFT loss 与显式 PyTorch loss 公式一致。**

## Step 4：验证训练 block 和 MoE

只验证 loss 还不够。DeepSeek-V4 的训练包含 attention、MLP、MoE、shared expert、all-to-all dispatch、optimizer update 等多个部分。

我们做了两类 reference：

1. 1-layer TransformerBlock external reference：用显式 PyTorch 公式复现一个 DeepSeek-V4 block。
2. real EP=8 MoELayer reference：用真实 8 rank expert parallel 布局验证 MoE。

验证数据来自：

- `artifacts/deepseek-v4-external-training-reference-1layer-20260531.json`
- `artifacts/deepseek-v4-external-training-reference-1layer-c4-20260531.json`
- `artifacts/deepseek-v4-external-training-reference-1layer-c128-20260531.json`
- `artifacts/deepseek-v4-external-training-reference-1layer-moe-20260531.json`
- `artifacts/deepseek-v4-external-moe-ep8-reference-20260531.json`

1-layer external reference：

| path | status | loss_abs | output max_abs | input_grad max_abs | grad max_abs | state_after_step max_abs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| non-compressed attention | `PASS` | `0.0` | `0.0` | `0.0` | `5.960464477539063e-08` | `0.0` |
| `compress_ratio=4` indexer path | `PASS` | `4.048051778227091e-07` | `0.03125` | `3.725290298461914e-09` | `1.1920928955078125e-07` | `4.656612873077393e-10` |
| `compress_ratio=128` compressed attention | `PASS` | `0.0` | `0.0` | `2.9802322387695312e-08` | `2.384185791015625e-07` | `4.656612873077393e-10` |
| score-routed MoE + shared expert | `PASS` | `2.6352936401963234e-05` | `0.03125` | `1.9073486328125e-06` | `7.62939453125e-06` | `1.4901161193847656e-08` |

real EP=8 MoELayer：

| 检查项 | 结果 |
| --- | ---: |
| status | `PASS` |
| world size | `8` |
| output_max_abs_global_max | `0.03125` |
| loss_abs_global_max | `7.690023630857468e-05` |
| input_grad_max_abs_global_max | `7.62939453125e-06` |
| expert_grad_max_abs_global_max | `0.0` |
| expert_state_after_step_max_abs_global_max | `0.0` |
| ranks_with_nonzero_local_expert_grad | `8` |

解释：

- 1-layer 验证证明：主要 block 结构的 forward、backward、一步 update 可以被外部公式复现。
- EP=8 验证证明：真实 expert parallel 布局、all-to-all dispatch、local expert gradient 和 update 都被覆盖。
- 因此，**MoE 不是只在简化单机公式里验证，而是在真实 EP=8 并行方式下验证过。**

## Step 5：验证 loaded 4-layer mini checkpoint

前面验证的是零件和一层 block。还需要验证真实 checkpoint 串起来是否成立。

这里使用 loaded 4-layer mini checkpoint 和固定 SFT batch，做 framework-level correctness gate。

验证数据来自：

- `artifacts/deepseek-v4-mini-checkpoint-correctness-gate-20260531.json`
- `artifacts/deepseek-v4-mini-checkpoint-correctness-rerun-sft-attention-output-replay-20260531.json`

mini checkpoint gate 结果：

| 检查项 | 结果 |
| --- | --- |
| status | `PASS` |
| real non-injected forward strict status | `FAIL` |
| real non-injected SFT one-step strict status | `FAIL` |
| strict failure reclassified as pass | `false` |

attention-output replay 的 SFT one-step 结果：

| compare | loss_abs_global_max | selected_grad_max_rel_gap | selected_state_max_abs |
| --- | ---: | ---: | ---: |
| dense vs sparse | `0.0` | `1.6638869012597368e-05` | `4.76837158203125e-07` |
| dense vs tilelang | `0.0` | `2.1286144848309263e-05` | `9.5367431640625e-07` |
| sparse vs tilelang | `0.0` | `1.768724307593672e-05` | `9.5367431640625e-07` |

解释：

- 真实非注入路径的 strict parity 仍失败，所以没有宣称 strict pass。
- 但当把已经定位的 attention forward-value drift replay 掉后，SFT loss 变为 exact，gradient/update 差异落入阈值。
- 这说明训练链路本身可以闭合，剩余 strict 差异来自已定位的 BF16 forward drift。

## Step 6：验证完整 4-layer external forward

为了进一步增强证据，我们又写了 loaded 4-layer full external reference。它不用 Miles/Megatron 的 block forward，而是显式重建：

- embedding
- 4 层 DeepSeek-V4 layer
- dense attention reference
- EP=8 hash-routed / score-routed MoE
- final norm
- output head
- SFT loss

验证数据来自：

- `artifacts/deepseek-v4-mini-external-full-reference-bf16-router-debug-20260531.json`
- `artifacts/deepseek-v4-mini-external-full-reference-bf16-routing-replay-tolerance-20260531.json`
- `artifacts/deepseek-v4-mini-external-full-reference-bf16-routing-replay-train-tolerance-20260531.json`

independent routing debug 结果：

| 检查项 | 结果 |
| --- | ---: |
| status | `FAIL` |
| layer 0/1/2 router map | exact |
| layer 3 router_map max_abs | `1.0` |
| layer 3 router_map mean_abs | `0.0049285888671875` |

解释：

- 前 3 层 hash routing 完全一致。
- 第 4 层 score-routed top-k 出现分支翻转。
- 这是离散 routing 边界问题，不适合直接拿来判断连续数学是否错误。

因此我们做 routing replay：固定 Miles 产生的 expert mask，再用 reference 重新计算被选 expert 的概率。

routing replay forward 结果：

| 检查项 | 结果 | 阈值 |
| --- | ---: | ---: |
| status | `PASS` | - |
| logit max_abs | `0.25` | `0.3` |
| logit mean_abs | `0.01131382118910551` | `0.02` |
| logit p99_abs | `0.05078125` | `0.08` |
| logit relative_l2_gap | `1.1485556081547443e-05` | `2e-05` |
| loss_abs_global_max | `1.25732421875` | `3.0` |
| loss_abs_per_token_global_max | `0.0031276721859452737` | `0.006` |

解释：

- 完整 4-layer forward/loss 在 routing replay BF16 容差下通过。
- loss 是 sum loss，当前 batch 有 402 个监督 token，所以更有解释性的值是 per-token gap：`0.00313/token`。

full external one-step train delta 结果：

| 检查项 | 结果 |
| --- | --- |
| status | `FAIL` |
| failures | `selected_grad_max_abs`, `selected_grad_relative_to_param_l2` |
| loss_abs_per_token_global_max | `0.0026333177860696517` |
| selected_grad_max_abs_global_max | `1.0` |
| selected_grad_relative_to_param_l2_global | `0.01485428690998053` |
| selected_state_max_abs_global_max | `2.2351741790771484e-08` |

解释：

- 这个失败已经真实运行，不是缺少输入。
- 它说明完整 4-layer 图中约 `1e-5` relative L2 的 forward drift 反传后，会在部分小范数参数上放大成 selected-gradient delta。
- 这里的判定信号是 `selected_grad_*`，不是 `selected_state_max_abs_global_max`。state delta 使用诊断性 `lr=1e-7`，数值会天然很小，不能独立证明 gradient 质量。
- 因此这个 artifact 作为 strict boundary 保留，不作为 PASS 证据。

## Step 7：为什么接受 BF16 tolerance

BF16 是低精度格式。不同 kernel、不同计算顺序、不同 fused path，会产生小的数值差异。对 DeepSeek-V4 这类模型，差异还可能经过 MoE routing 和 output head 放大。

我们接受 BF16 tolerance 的前提不是“差不多就行”，而是同时满足四个条件：

1. 差异有限且有阈值。
2. 差异来源能定位。
3. 关键训练目标和 backward/update 有独立 reference。
4. strict failure 不被改写成 pass。

验证数据来自：

- `artifacts/deepseek-v4-end-to-end-bf16-tolerance-20260531.json`
- `artifacts/deepseek-v4-official-forward-bf16-tolerance-20260531.json`

本次 BF16 envelope：

| 检查项 | 阈值 |
| --- | ---: |
| real forward relative_l2_gap | `<= 2e-05` |
| real forward mean_abs | `<= 0.08` |
| real forward p99_abs | `<= 0.37` |
| routing-replay forward relative_l2_gap | `<= 4e-06` |
| routing relative_l2 reduction | `>= 4x` |
| real SFT selected_grad_max_rel_gap | `<= 5e-04` |
| real SFT selected_state_max_abs | `<= 2e-05` |
| SFT attention-output replay selected_grad_max_rel_gap | `<= 3e-05` |
| SFT attention-output replay selected_state_max_abs | `<= 2e-05` |

结果：

| artifact | status |
| --- | --- |
| end-to-end BF16 tolerance | `PASS` |
| official-vs-Miles BF16 forward tolerance | `PASS` |

解释：

- 真实 forward/train drift 没有被忽略，而是被阈值约束。
- official/reference strict parity 仍记录为失败，没有被改成通过。

## Step 8：用 proof ledger 防止证据链自相矛盾

最后，我们还需要防止文档说法和 artifact 不一致。

验证数据来自：

- `artifacts/deepseek-v4-proof-coverage-matrix-20260531.json`
- `artifacts/deepseek-v4-proof-ledger-20260531.json`

结果：

| 检查项 | 结果 |
| --- | --- |
| proof coverage matrix | `PASS` |
| proof ledger | `PASS` |

coverage matrix 明确保留的 open strict gates：

| gate | status |
| --- | --- |
| strict_mini_backend_logprob_parity | `FAIL` |
| strict_mini_checkpoint_train_step_backend_parity | `FAIL` |
| official_reference_mini_checkpoint_forward_parity | `FAIL` |
| production_ep8_moe_path_strict_parity | `PARTIALLY_LOCALIZED` |
| external_reference_mini_checkpoint_one_step_train_parity | `FAIL_DIAGNOSTIC` |

解释：

- proof ledger 证明证据链内部一致。
- coverage matrix 证明文档没有把未关闭 strict gate 藏起来。
- 因此，这份证明是“有边界的训练正确性证明”，不是“所有东西都 strict 完美一致”的证明。

## Step 9：为什么没有用 FP32 直接关闭剩余 FAIL

一个自然问题是：既然剩余差异看起来和 BF16 精度有关，能不能直接把同一套验证切成 FP32，看看 strict gap 是否消失？

这个想法本身是对的，但当前 Miles DeepSeek-V4 runtime 做不到真正的 FP32 生产路径验证。

验证数据来自：

- `artifacts/deepseek-v4-fp32-strict-closure-attempt-20260601.json`
- `docs/deepseek-v4-fp32-strict-closure.md`

尝试结果：

| 检查项 | 结果 |
| --- | --- |
| full external FP32 verifier | `FAILED_BEFORE_CHECKPOINT_LOAD` |
| 失败位置 | model construction |
| 根因 | `DeepSeekV4Attention` 要求部分 attention projection 权重是 `torch.bfloat16` |
| strict logprob parity 是否被 FP32 关闭 | 否 |
| selected-gradient diagnostic 是否被 FP32 关闭 | 否 |

解释：

- 如果存在真实 FP32 DeepSeek-V4 runtime，FP32 确实可以作为很强的诊断：看 dense/sparse/tilelang backend 差异和 full external selected-gradient 差异是否塌缩。
- 但当前 Miles DeepSeek-V4 是 BF16 训练实现：attention、helper path、TileLang backward kernel 都有 BF16 输入假设。
- 强行删断言或替换 kernel 只会测试一个临时 FP32 变体，不再是当前真实训练路径。
- 因此 FP32 不能用来把这两个 strict gate 改成 PASS。正确做法是保留 strict boundary，并继续依赖 BF16 生产路径上的分层证据。

## 最终判断

本次验证支持以下判断：

1. **Miles 当前 DeepSeek-V4 路径没有 Megatron-LM PR #4839 的 HC 方向错误。**
2. **SFT loss 目标正确，并且 backward/update 有显式 reference 证据。**
3. **attention、QAT、MoE、optimizer、训练 block、真实 EP=8 MoELayer 都有独立验证。**
4. **loaded 4-layer mini checkpoint 的 framework-level correctness gate 在 BF16 训练容差下为 `PASS`。**
5. **完整 4-layer full external forward/loss 在 routing replay BF16 容差下为 `PASS`。**
6. **FP32 direct closure 在当前 Miles DeepSeek-V4 runtime 不可执行，不能用来关闭剩余 strict gate。**
7. **strict logprob parity 和 full external one-step train selected-gradient strict parity 仍未关闭，已作为边界记录。**

所以，面向项目决策可以这样表述：

**当前 Miles DeepSeek-V4 训练方式已经通过严格分层的数学和端到端验证，可以作为 BF16 训练正确性证据；但它不是 bitwise/strict parity 证明，后续如果目标变成 strict parity，需要继续处理已记录的 BF16 forward drift、routing branch flip 和 selected-gradient diagnostic failure。**

## 读原始材料时看哪里

| 想确认的问题 | 推荐入口 |
| --- | --- |
| 完整技术细节 | `docs/deepseek-v4-hyperconnection-runtime.md` |
| 机器可读总摘要 | `artifacts/deepseek-v4-proof-summary-20260531.json` |
| 证据链是否一致 | `artifacts/deepseek-v4-proof-ledger-20260531.json` |
| 覆盖哪些证明项 | `artifacts/deepseek-v4-proof-coverage-matrix-20260531.json` |
| mini checkpoint 总 gate | `artifacts/deepseek-v4-mini-checkpoint-correctness-gate-20260531.json` |
| full external forward/loss | `artifacts/deepseek-v4-mini-external-full-reference-bf16-routing-replay-tolerance-20260531.json` |
| full external train diagnostic | `artifacts/deepseek-v4-mini-external-full-reference-bf16-routing-replay-train-tolerance-20260531.json` |
| FP32 strict closure attempt | `artifacts/deepseek-v4-fp32-strict-closure-attempt-20260601.json` |

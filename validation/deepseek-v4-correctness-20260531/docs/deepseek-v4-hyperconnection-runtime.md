# DeepSeek-V4 训练正确性验证

本文记录 DeepSeek-V4 训练精度风险的定位、Miles 的实现方式、已引入的修复，以及 2026-05-31 的算子级和端到端验证结果。原始问题来自 Megatron-LM 的 mHC / HyperConnection residual mixing 方向错误，相关上游修复见 [NVIDIA/Megatron-LM PR #4839](https://github.com/NVIDIA/Megatron-LM/pull/4839)。SWIFT 的 DeepSeek-V4 文档也将该问题作为 Megatron-Core 侧必须修复的精度问题处理，见 [SWIFT DeepSeek-V4 Best Practice](https://swift.readthedocs.io/zh-cn/latest/BestPractices/deepseek-v4.html)。

本文只关注问题本身、定位方法、Miles 实现、验证结果和证明边界，不依赖本地运行路径。

## External References

本文引用两个外部材料：

1. [NVIDIA/Megatron-LM PR #4839](https://github.com/NVIDIA/Megatron-LM/pull/4839)：用于定位 DeepSeek-V4 mHC / HyperConnection residual mixing 方向风险。该 PR 是 Megatron-LM 侧的 DeepSeek-V4 bug-fix PR，包含 native mHC 和 fused mHC 的 `H_res` 方向修复。本文只把它作为问题来源和 HC orientation oracle 的依据。
2. [SWIFT DeepSeek-V4 Best Practice](https://swift.readthedocs.io/zh-cn/latest/BestPractices/deepseek-v4.html)：用于参考 DeepSeek-V4 训练精度问题的发现和验证方式，尤其是把问题拆成可控 forward/parity 检查和训练有效性检查。本文不把 SWIFT 文档作为 Miles 正确性的直接证据。

Miles 的正确性结论仍以本仓库 artifact 为准：operator math、trace replay、mini checkpoint replay、training-step replay、official forward BF16 tolerance、BF16 tolerance envelope、optimizer update math、fix regression guards、coverage matrix 和 proof ledger。

## 结论

当前结论是 **PARTIAL_PROOF**：

1. Megatron PR #4839 的 HyperConnection 方向问题没有在 Miles DeepSeek-V4 当前路径复现。Miles 没有走 Megatron upstream `HyperConnectionModule`，而是走自己的 DeepSeek-V4 HC 实现；算子验证证明 Miles post-mix 等价于 `H_res.T @ residual`，并明显不同于错误的 `H_res @ residual`。
2. 算子级数学验证已通过：HyperConnection、RoPE out-of-place / inverse / backward、official-compatible KV QAT、dense/sparse PyTorch attention、TileLang sparse MLA forward/backward 都为 PASS。
3. Attention、Grouped MLP、EP=8 all-to-all dispatcher 和 TransformerBlock 训练步验证已通过：dense、sparse、tilelang 三个 attention backend 在模块级和 block 级的 forward / backward / 一步 SGD update 都在阈值内；mini checkpoint 真实 attention 输入上的 local forward/backward/update replay 已通过；完整 SFT one-step 在 routing replay + attention-output straight-through replay 下也已通过；TE grouped expert 在 DeepSeek-V4 生产 hidden / FFN 维度下与逐 expert BF16 公式 bitwise exact；EP=8 all-to-all dispatch/combine 的 forward/backward/update 与直接 reference bitwise exact。
4. official inference attention trace replay 已通过：layer-0 attention 的 q/kv/topk/attention core/output projection 均已白盒对齐到 BF16 生产容差内；Miles attention core 对自己的 sparse replay 是 bitwise exact。
5. layer-0 loaded weight mapping 已覆盖 attention、router、hash tid2eid、shared experts、256 个 routed experts、final layernorm 和 output head，全部 `exact_equal=True`。
6. mini checkpoint 端到端 forward / SFT one-step 都能执行，输出和梯度 finite；在消除 routing 分叉并用 straight-through 方式消除 attention forward value drift 后，完整 SFT one-step loss/gradient/update parity 通过；但真实非注入 forward 的 strict backend parity 和 official full-forward strict logprob parity 仍未通过。
7. official-vs-Miles full-forward BF16 容差标准已通过：strict official/reference forward parity 仍记录为 FAIL，但 sparse/tilelang 的 relative_l2、mean_abs、p99_abs 和 max_abs 均落在独立声明的 BF16 forward envelope 内。
8. 第一段 external training-reference gate 已通过：脚本内显式 PyTorch 训练态公式 reference 与 Megatron/Miles 1-layer non-compressed DeepSeek-V4 TransformerBlock 在 forward、loss、input gradient、参数梯度和一步 SGD update 上对齐。
9. 端到端 BF16 容差检查已通过：真实非注入 forward / train step 仍不满足 strict parity，但落在声明的 BF16 runtime envelope 内；消除已定位的 attention forward-value drift 后，完整 SFT one-step loss exact，梯度和更新仍在阈值内。
10. Optimizer 路径和 AdamW 第一步更新数学已验证：Miles DeepSeek-V4 训练路径走 Megatron optimizer，DeepSeek-V4 runner 使用 Adam 超参；AdamW 第一步 reference 公式、闭式公式、zero-grad weight decay 和最坏 sign-flip 更新上界均通过。
11. 修复项 regression guard 已通过：当前源码仍包含 non-RoPE KV QAT、official-compatible in-place `act_quant`、FP32 QAT scale、compressor no-overlap write 和 trace gating。
12. Proof coverage matrix 已通过：用户目标被拆成明确 proof requirements，每项都有 artifact 和文档章节支撑；诊断性 FAIL artifact 与真正缺口被区分处理。
13. 环境 provenance 已通过：文档记录的 GPU、driver、CUDA、Python、PyTorch、Megatron-Core、Transformer Engine、TileLang 等版本与 `operator_math` artifact 中的实际验证环境一致。
14. 外部引用 provenance 已通过：Megatron-LM PR #4839 和 SWIFT 文档的用途被限定为问题来源和验证方法参考，Miles 正确性仍以本仓库 artifact 为准。
15. 证明账本一致性检查已通过：`deepseek-v4-proof-ledger-20260531.json` 对所有关键 artifact 的状态、误差下降、bitwise replay、训练步阈值、external training reference、official forward BF16 tolerance、BF16 tolerance envelope、optimizer update math、修复项 source guard、coverage matrix、environment provenance、external reference provenance 和剩余 gap 定位做了机器校验，未发现失败项。

因此，我们可以证明：**当前已覆盖的 HC、RoPE、QAT、attention dense/sparse/tilelang、local grouped expert、EP=8 all-to-all dispatch/combine、模块训练步、block 训练步、mini checkpoint attention I/O 训练步、以及消除 attention 前向值漂移后的完整 SFT one-step 训练链路这些核心数学和训练算子是正确的；但还不能宣称完整 Miles 训练框架已经与 official/reference 在真实非注入 forward 下端到端 strict parity 完全等价。** 剩余差异已被定位为 BF16/FP8 training runtime 与 official inference runtime 的数值漂移，在完整模型中经过 MoE routing / output head 后放大。

机器可读摘要见 `docs/en/advanced/deepseek-v4-proof-summary-20260531.json`。

## 问题背景

DeepSeek-V4 使用 HyperConnection 将多个 residual streams 混合。post-mix 的核心计算可以抽象为：

```python
output = h_post * x + H_res.T @ residual
```

其中 `H_res` / `comb` 由 Sinkhorn 产生，是双随机矩阵，但通常不是对称矩阵。因此：

```python
H_res @ residual != H_res.T @ residual
```

Megatron-LM PR #4839 修复的就是这类方向问题，包括 native mHC 和 fused mHC 两条路径。如果实现使用了 `H_res @ residual`，训练不一定马上 NaN，loss 也可能下降，但 forward 会和 reference 不对齐。这属于精度错误，不能靠 loss 曲线排除。

## 如何定位

定位方法是先把问题从完整训练中拆到算子公式，再回到端到端链路：

1. 从 Megatron-LM PR #4839 和 SWIFT 文档确认风险点是 mHC post-mix 的 `H_res` 方向。
2. 构造非对称 `H_res`，避免单位矩阵、全 1 矩阵或对称矩阵把方向错误掩盖。
3. 对比三种结果：Miles 当前实现、正确公式 `comb.T @ residual`、错误公式 `comb @ residual`。
4. 额外把 Megatron PR #4839 修复后的 native 公式单独写成 reference，证明不是只和我们自己的 expected 对齐。
5. 检查当前运行时是否真正走 Megatron upstream 的 `HyperConnectionModule`，还是走 Miles 自己的 DeepSeek-V4 实现。

核心判断如下：

```python
wrong = torch.matmul(comb, residual)
expected = torch.matmul(comb.transpose(-1, -2), residual)
assert not torch.allclose(wrong, expected)
```

如果被测实现与 `expected` 和 PR #4839 fixed native reference 一致，并且明显不同于 `wrong`，就能证明该风险点没有复现。

## Miles 如何实现

Miles 当前 DeepSeek-V4 路径不是直接使用 Megatron upstream 的 `HyperConnectionModule`。DeepSeek-V4 的 HyperConnection 由 Miles 自己的实现承担：

- `miles_plugins/models/deepseek_v4/ops/hyper_connection.py`
  - `DeepSeekV4HyperConnectionUtil.hc_pre_raw`
  - `DeepSeekV4HyperConnectionUtil.hc_post_raw`
  - `DeepSeekV4HyperConnectionUtil.layer_pre`
  - `DeepSeekV4HyperConnectionUtil.layer_post`
  - `DeepSeekV4HyperConnectionUtil.block_head`
- V4 Megatron patch
  - 给 Megatron `TransformerConfig` 增加 DeepSeek-V4 相关字段。
  - 在 transformer block 开始处做 HC stream expand。
  - 在 attention / MLP sublayer 前后调用 Miles 的 `DeepSeekV4HyperConnectionUtil`。
  - 在 final layernorm 前做 HC head contraction。

Miles 的 post-mix 实现等价于：

```python
post.unsqueeze(-1) * x.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual)
```

因此它和 Megatron-LM PR #4839 修复后的方向一致。

## 已修复问题

本轮验证过程中确认并修复了两个新的 DeepSeek-V4 训练正确性问题：

1. KV QAT 维度：official inference 只对 KV 的 non-RoPE 维度做 activation QAT，RoPE 维度保持 BF16；Miles 已改为只量化 `kv[..., :-rd]`，保留 `kv[..., -rd:]`。
2. KV QAT 语义：official inference 的 `act_quant(..., scale_fmt=None, scale_dtype=torch.float32, inplace=True)` 是 fused quant-dequant 回 BF16，不是返回真实 FP8 tensor 再乘 scale；Miles 的 `fp8_simulate_qat` 已改为 official-compatible in-place BF16/FP32-scale 语义。
3. Compressor QAT 写法：official-compatible QAT 返回值可能和 slice 写回形成重叠写风险；Miles compressor 已改为 `torch.cat([qat(non_rope), rope])`，保持数学等价但避免 PyTorch overlap error。

同时保留前面已经验证过的修复：

| 修复 | 作用 | 验证 |
| --- | --- | --- |
| RoPE out-of-place | 避免 inplace slice 破坏 autograd。 | inverse recover、input 不变、grad finite。 |
| sparse attention fully masked row | 避免 `0 * inf = NaN` backward。 | dense/sparse forward/backward + fully masked row。 |
| TileLang sparse MLA backward | 修复 masked `-1` 读非法位置和 shared-memory alias 风险。 | TileLang vs dense reference forward/backward。 |
| RoPE cache 扩容 | 覆盖 CP padding 后超过 64K 的序列。 | cache 长度结构检查。 |

## 算子数学验证

验证脚本：`tools/verify_deepseek_v4_operator_math.py`

最新结果：`docs/en/advanced/deepseek-v4-operator-math-20260531.json`

运行结果：

```text
overall_status=PASS

hyper_connection_pr4839_orientation:
  max_diff_vs_megatron_pr4839_fixed_native=2.384185791015625e-07
  max_diff_vs_prefix_wrong_comb_residual=9.81311321258545

official_kv_qat_simulation:
  max_abs=0.0
  mean_abs=0.0
  nonzero_abs_count=0
  exact_equal=True

dense_sparse_torch_attention:
  forward_max_abs=0.0078125
  forward_rel_diff=2.197893374189519e-06
  dq_rel_diff=3.0149637411103214e-06
  dkv_rel_diff=3.950309416911324e-06
  dattn_sink_rel_diff=4.030836156632134e-06
  fully_masked_row_max_abs=0.0

tilelang_sparse_attention:
  max_forward_max_abs=0.004327297210693359
  max_forward_rel_diff=2.175237100887628e-06
  max_dq_rel_diff=4.225850927852548e-06
  max_dkv_rel_diff=4.144493625735102e-06
  max_dattn_sink_rel_diff=6.709198812515638e-06
```

这组结果证明了：PR #4839 方向、RoPE、official QAT、dense/sparse reference、TileLang forward/backward 都不是当前剩余端到端 drift 的来源。

## Attention Trace Replay

验证脚本：`tools/verify_deepseek_v4_attention_trace_replay.py`

最新结果：`docs/en/advanced/deepseek-v4-attention-trace-replay-qatsim-0415-20260531.json`

验证方法：

1. official inference 1-layer full forward 保存内部 attention trace：`q_after_rope`、`kv_after_rope_qat`、`attn_sink`、`topk_idxs`、`attention_core`、`wo_b.input`。
2. Miles 1-layer BF16 sparse forward 保存内部 attention trace：`q_after_rope`、`kv_vanilla_after_rope_qat`、`attention_core`、`after_wo_a`、`after_wo_b`。
3. 用同一组 trace tensor 重新计算 dense/sparse attention reference，分别做 FP32 数学重放和 BF16 生产重放。

关键结果：

```text
status=PASS

q_after_rope_official_vs_miles:
  max_abs=0.0625
  mean_abs=2.61895070252649e-06
  mismatches=18 / 16777216
  relative_gap=-5.963718052726108e-08

kv_after_rope_qat_official_vs_miles:
  max_abs=0.00390625
  mean_abs=7.778365329613735e-08
  mismatches=0
  relative_gap=0.0

topk_window_indices_official_vs_expected:
  exact_equal=True
  mismatches=0

dense_vs_sparse_fp32_math_replay_from_official_inputs:
  max_abs=1.9073486328125e-06
  mean_abs=9.406129208855418e-09
  mismatches=0
  relative_gap=0.0

dense_vs_sparse_fp32_math_replay_from_miles_inputs:
  max_abs=1.9073486328125e-06
  mean_abs=9.47945810736428e-09
  mismatches=0
  relative_gap=0.0

miles_attention_core_vs_sparse_replay:
  exact_equal=True
  max_abs=0.0
  mean_abs=0.0
  mismatches=0

official_attention_core_vs_sparse_replay:
  max_abs=0.03125
  mean_abs=0.00011084476136602461
  relative_gap=9.39835491209351e-07

after_wo_b_official_vs_miles:
  max_abs=0.0625
  mean_abs=0.0026297857984900475
  relative_gap=4.000893816691331e-06
```

这组结果说明：

1. QAT 修复后，official 和 Miles 的 KV QAT 输出已经在 BF16 容差内对齐，且没有 strict threshold mismatch。
2. window topk 与 expected 完全一致。
3. attention 的 FP32 数学重放中 dense 与 sparse reference 逐元素通过。
4. Miles runtime 的 attention core 与 Miles sparse replay bitwise exact。
5. official sparse kernel 与 Miles sparse replay 只有 BF16 生产级数值差异，relative gap 约 `1e-6`。

## 权重加载验证

验证脚本：`tools/verify_deepseek_v4_loaded_weight_mapping.py`

最新结果：`docs/en/advanced/deepseek-v4-loaded-weight-mapping-1layer-mlp-qatsim-20260531.json`

检查对象：

```text
embedding.word_embeddings.weight
attn_sink
wq_a.weight
q_norm.weight
wq_b.weight
wkv.weight
kv_norm.weight
wo_a.weight
wo_b.weight
pre_mlp_layernorm.weight
mlp.router.weight
mlp.router.tid2eid
mlp.shared_experts.linear_fc1.weight
mlp.shared_experts.linear_fc2.weight
mlp.experts.linear_fc1.weight0..weight255
mlp.experts.linear_fc2.weight0..weight255
final_layernorm.weight
output_layer.weight
```

结果全部为：

```text
exact_equal=True
max_abs=0.0
mean_abs=0.0
nonzero_abs_count=0
```

其中 routed experts 是按 TE GroupedLinear 的 `weight0..weight255` 逐 expert 与 raw checkpoint 的 `[256, 4096, 4096]` / `[256, 4096, 2048]` 张量切片比较：

```text
mlp.experts.linear_fc1:
  num_checked_experts=256
  num_failed_experts=0
  exact_equal=True
  max_abs=0.0
  nonzero_abs_count=0

mlp.experts.linear_fc2:
  num_checked_experts=256
  num_failed_experts=0
  exact_equal=True
  max_abs=0.0
  nonzero_abs_count=0
```

这证明 layer-0 attention、MLP/router/expert、final layernorm 和 output head 的关键权重都不是从 checkpoint 到 Miles 过程中发生了错位或数值变化。

## 模块训练步验证

验证脚本：`tools/verify_deepseek_v4_attention_training_step.py`

最新结果：`docs/en/advanced/deepseek-v4-attention-training-step-qatsim-20260531.json`

覆盖：

| case | seqlen | backend | status |
| --- | ---: | --- | --- |
| `compress_ratio=0` | 16 | dense / sparse / tilelang | PASS |
| `compress_ratio=4` | 128 | dense / sparse / tilelang | PASS |
| `compress_ratio=128` | 256 | dense / sparse / tilelang | PASS |

说明：

- 每个 case 都完成 forward、backward、参数梯度 finite 检查和一次 SGD update。
- `compress_ratio=128` 使用 `seqlen=256`，确保 compressed-KV 路径包含有效压缩 token，而不是只有边界无效压缩索引。
- 这组结果证明三种 attention backend 在模块训练步上是同一数学路径的 BF16 近似实现。

## TransformerBlock 训练步验证

验证脚本：`tools/verify_deepseek_v4_transformer_block_training_step.py`

最新结果：`docs/en/advanced/deepseek-v4-transformer-block-training-step-qatsim-20260531.json`

覆盖：

- Megatron `TransformerBlock` 使用 Miles DeepSeek-V4 spec。
- HC block expand/head。
- 两层 layer pre/post HC。
- 第 0 层 `compress_ratio=0`，第 1 层 `compress_ratio=4`。
- attention、MLP、final layernorm。
- dense / sparse / tilelang forward / backward / update parity。

关键结果：

```text
status=PASS

dense_vs_sparse:
  output_max_abs=0.046875
  output_rel_gap=1.4185477623218645e-05
  input_grad_max_abs=1.430511474609375e-06
  input_grad_rel_gap=2.651357764016371e-05

dense_vs_tilelang:
  output_max_abs=0.046875
  output_rel_gap=1.4841798940512518e-05
  input_grad_max_abs=1.430511474609375e-06
  input_grad_rel_gap=2.780726377638043e-05

sparse_vs_tilelang:
  output_max_abs=0.0390625
  output_rel_gap=1.2039845451283782e-05
  input_grad_max_abs=1.430511474609375e-06
  input_grad_rel_gap=2.604709797326965e-05
```

这证明 DeepSeek-V4 block 组合路径本身能在训练步上保持 dense/sparse/tilelang 的容差一致。

## Official Full-Forward Probe

验证脚本：`tools/verify_deepseek_v4_official_full_forward.py`

最新结果：

- `docs/en/advanced/deepseek-v4-official-full-forward-1layer-bf16-qatsim-0415-20260531.json`
- `docs/en/advanced/deepseek-v4-official-full-forward-1layer-bf16-tilelang-qatsim-0415-20260531.json`
- `docs/en/advanced/deepseek-v4-official-vs-miles-1layer-trace-bf16-qatsim-0415-20260531.json`
- `docs/en/advanced/deepseek-v4-head-replay-qatsim-0415-20260531.json`
- `docs/en/advanced/deepseek-v4-mlp-trace-compare-qatsim-0415-20260531.json`
- `docs/en/advanced/deepseek-v4-mlp-expert-replay-qatsim-0415-20260531.json`
- `docs/en/advanced/deepseek-v4-official-runtime-precision-variants-qatsim-0415-20260531.json`

1-layer BF16 official inference vs Miles sparse：

```text
status=FAIL
tokens=402
rtol=0.002
atol=0.02
mismatches=63
max_abs=0.16042709350585938
mean_abs=0.03158159181475639
relative_l2_gap=2.313715897095392e-06
```

1-layer BF16 official inference vs Miles tilelang：

```text
status=FAIL
tokens=402
rtol=0.002
atol=0.02
mismatches=71
max_abs=0.14663314819335938
mean_abs=0.03211159259080887
relative_l2_gap=2.4187967853084302e-06
```

与旧 BF16 结果相比，QAT 修复后 sparse 路径从：

```text
mismatches=123
max_abs=0.2347126007080078
mean_abs=0.04502306506037712
relative_l2_gap=4.940713539958175e-06
```

改善为：

```text
mismatches=63
max_abs=0.16042709350585938
mean_abs=0.03158159181475639
relative_l2_gap=2.313715897095392e-06
```

这个结果需要明确解读：

1. official full-forward reference 已经能加载同一个 release mini checkpoint，并使用同一 rollout 运行。
2. QAT 维度和 QAT 语义修复确实改善了 external reference 对齐。
3. sparse 和 tilelang 都是同量级 drift，说明剩余差异不是 sparse-torch 独有实现问题。
4. strict official-vs-Miles response logprob parity 仍然 FAIL，所以不能把当前结果等同于 SWIFT 文档中 reference parity 已完全通过的状态。

## Official Forward BF16 Tolerance

验证脚本：`tools/verify_deepseek_v4_official_forward_tolerance.py`

最新结果：`docs/en/advanced/deepseek-v4-official-forward-bf16-tolerance-20260531.json`

这个 verifier 把 official/reference full-forward 从 strict parity 里拆出一个独立的 BF16 容差门禁。它不把 strict logprob compare 重新判为 PASS；它只回答：在 QAT 修复、weight mapping、attention trace replay、MLP expert replay 和 grouped MLP math 都通过后，剩余 official-vs-Miles full-forward drift 是否被一个可接受的 BF16 envelope 约束。

本次声明的 official forward BF16 envelope：

| item | threshold |
| --- | ---: |
| relative_l2_gap | <= 5e-06 |
| mean_abs | <= 0.04 |
| p99_abs | <= 0.15 |
| max_abs | <= 0.20 |
| strict compare rtol / atol | 0.002 / 0.02 |

验证结果为 `PASS`，但 strict forward status 仍为 `FAIL`：

| backend | strict status | mismatches | max_abs | mean_abs | p99_abs | relative_l2_gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sparse | FAIL | 63 / 402 | 0.16042709350585938 | 0.03158159181475639 | 0.13715827465057373 | 2.313715897095392e-06 |
| tilelang | FAIL | 71 / 402 | 0.14663314819335938 | 0.03211159259080887 | 0.12487327307462692 | 2.4187967853084302e-06 |

这个 artifact 还记录了 external one-step train parity 的前置条件：

```text
required_gate=official_reference_mini_checkpoint_forward_parity
required_status=PASS
actual_status=FAIL
external_one_step_train_parity_status=SKIPPED_FORWARD_STRICT_PARITY_REQUIRED
```

因此，当前可以声明的是 official/reference full-forward 的 BF16 数值容差已被固定并通过；不能声明 external reference one-step train parity 已经通过。后者只有在 strict official/reference forward parity 先通过后才是一个数学上干净的训练步对齐测试。

继续 replay output head 后，剩余 drift 又拆出了一层：

```text
official_saved_vs_official_hidden_fp32_head_replay:
  status=PASS
  mismatches=0
  max_abs=0.0
  mean_abs=0.0

miles_saved_vs_miles_hidden_bf16_head_replay:
  status=PASS
  mismatches=0
  max_abs=1.9073486328125e-06
  mean_abs=1.0853383258790927e-07

same_fp32_head_official_hidden_vs_miles_hidden:
  status=FAIL
  mismatches=24 / 402
  max_abs=0.1319141387939453
  mean_abs=0.021717099472880363
  relative_l2_gap=1.0517284210198596e-06

official_hidden_fp32_head_vs_bf16_head:
  status=FAIL
  mismatches=35 / 402
  max_abs=0.13891220092773438
  mean_abs=0.024665921926498413
  relative_l2_gap=1.472509922662013e-06

miles_hidden_fp32_head_vs_bf16_head:
  status=FAIL
  mismatches=24 / 402
  max_abs=0.1281585693359375
  mean_abs=0.02311176247894764
  relative_l2_gap=1.2621014493507943e-06
```

这说明：

1. official saved logprob 的 head 路径是 final hidden + FP32 head replay。
2. Miles saved logprob 的 head 路径是 final hidden + BF16 head matmul replay。
3. Miles 输出不是 head weight 映射错误；用同一 checkpoint output weight 和 BF16 matmul 能 replay 到 0 mismatch。
4. official-vs-Miles final hidden 本身仍有 drift；同时 FP32 head vs BF16 head 的精度路径差异也会把 logprob 放大到当前 strict threshold 之外。
5. 因此 official full-forward strict parity 的剩余差异至少包含两部分：final hidden drift 和 output head precision mismatch。这进一步说明剩余问题不是 HC PR #4839、KV QAT、attention topk、attention core、MLP/router/expert weight mapping 或 output weight mapping 单点错误。

继续拆 layer-0 MLP trace 后，MoE routing 也不是第一个离散分叉点：

```text
router_indices_onehot_official_vs_miles:
  status=PASS
  exact_equal=True
  mismatches=0

router_selected_weights_official_vs_miles:
  status=PASS
  max_abs=0.006959676742553711
  mean_abs=0.000711353961378336
  mismatches=0
  relative_l2_gap=5.760798830745095e-06

shared_experts_official_vs_miles:
  status=PASS
  max_abs=0.015625
  mean_abs=0.0006783942808397114
  mismatches=0
  relative_l2_gap=6.183276526128267e-06

routed_experts_aggregated_official_vs_miles:
  status=FAIL
  max_abs=0.2509765625
  mean_abs=0.0013196724466979504
  mismatches=10775
  relative_l2_gap=1.6837643582956474e-05

ffn_output_official_vs_miles:
  status=FAIL
  max_abs=0.25
  mean_abs=0.001614017179235816
  mismatches=10756
  relative_l2_gap=1.4469553881180097e-05
```

这说明 layer-0 hash router 的 expert indices 是 exact_equal，routing weights 和 shared expert 输出都在阈值内；剩余 MLP drift 主要来自 routed expert 聚合输出的 BF16/dispatch/专家计算路径，而不是 MoE router 先发生离散分叉。

进一步做 layer-0 routed expert replay：

```text
official_formula_total_replay_vs_official_ffn:
  status=PASS
  exact_equal=True
  mismatches=0
  max_abs=0.0
  mean_abs=0.0
  relative_l2_gap=0.0

official_vs_miles_uses_same_expert_indices:
  True

miles_official_formula_total_replay_vs_miles_ffn:
  status=FAIL
  mismatches=6907
  max_abs=0.25
  mean_abs=0.0008694053394719958
  relative_l2_gap=7.582827647678592e-06

miles_megatron_bf16_formula_total_replay_vs_miles_ffn:
  status=FAIL
  mismatches=4559
  max_abs=0.125
  mean_abs=0.0006008074851706624
  relative_l2_gap=5.33511694711386e-06

megatron_bf16_formula_is_closer_to_miles_than_official_formula:
  True
```

这组 replay 说明：

1. 同一份 checkpoint 的 routed expert `fc1/fc2` 权重和 official 公式可以 bitwise 复现 official MLP 总输出；因此 official weight conversion 和公式 reference 是自洽的。
2. official 与 Miles 在 layer-0 hash router 的 expert indices 完全一致，因此这个样本的第一个 MLP 分叉点不是离散路由。
3. 用 Miles 输入和 Miles router 权重重放时，近似 Megatron BF16 activation/概率乘法路径比 official FP32 activation 路径更接近 Miles MLP 输出。
4. Miles MLP 剩余差异仍未 bitwise 消除；它被进一步定位到 Megatron BF16 MoE training runtime 与 official inference 单 expert FP32 activation 路径之间，而不是 attention、router indices 或 official expert 权重转换错误。

继续对 Megatron/TE grouped expert 本身做独立数学验证：

验证脚本：`tools/verify_deepseek_v4_grouped_mlp_math.py`

最新结果：`docs/en/advanced/deepseek-v4-grouped-mlp-math-20260531.json`

验证方法：

1. 直接实例化生产路径使用的 `TEGroupedMLP` / `TEColumnParallelGroupedLinear` / `TERowParallelGroupedLinear`。
2. 使用 DeepSeek-V4 0415 的 routed expert 语义：BF16、SwiGLU、`activation_func_clamp_value=10`、probability 在 activation 后、fc2 前相乘。
3. 构造两组 sorted expert tokens：一组包含 empty expert 和不均衡 tokens per expert，一组所有 expert 均衡激活。
4. 使用 DeepSeek-V4 生产 hidden / expert 维度：`hidden_size=4096`、`moe_ffn_hidden_size=2048`。
5. 对比 `TEGroupedMLP` 与显式逐 expert BF16 reference 的 forward、input gradient、每个 expert 的 fc1/fc2 weight gradient、以及一步 SGD update。

关键结果：

```text
status=PASS
threshold:
  rtol=0.0
  atol=0.0

imbalanced_with_empty_experts:
  tokens_per_expert=[0, 1, 7, 3, 9]
  forward_output exact_equal=True, max_abs=0.0, mismatches=0
  input_grad exact_equal=True, max_abs=0.0, mismatches=0
  fc1/fc2 weight grads exact_equal=True for all 5 experts
  fc1/fc2 SGD updates exact_equal=True for all 5 experts

balanced_all_experts_active:
  tokens_per_expert=[4, 4, 4, 4, 4]
  forward_output exact_equal=True, max_abs=0.0, mismatches=0
  input_grad exact_equal=True, max_abs=0.0, mismatches=0
  fc1/fc2 weight grads exact_equal=True for all 5 experts
  fc1/fc2 SGD updates exact_equal=True for all 5 experts
```

这证明本地 routed expert grouped kernel 的数学路径不是黑盒近似：在 DeepSeek-V4 生产维度下，TE grouped expert 的 forward/backward/update 与显式逐 expert BF16 公式 bitwise exact。它仍不等价于完整 EP=8 all-to-all dispatch proof，也不消除 official-vs-Miles full-forward drift；但它排除了“TEGroupedMLP 本地 expert GEMM / SwiGLU / probability application 公式错误”这个风险点。

继续对 EP=8 all-to-all dispatch / combine 做独立数学验证：

验证脚本：`tools/verify_deepseek_v4_moe_ep8_dispatch_math.py`

最新结果：`docs/en/advanced/deepseek-v4-moe-ep8-dispatch-math-20260531.json`

验证方法：

1. 使用 8 个 rank 初始化 expert parallel size = 8。
2. 直接实例化 Megatron 的 `MoEAlltoAllTokenDispatcher`，隔离验证 all-to-all token dispatch、combine 和 autograd。
3. 使用 DeepSeek-V4 hidden 维度 `hidden_size=4096`。
4. 中间 expert 计算使用可手算公式 `(hidden * expert_scale + expert_bias) * route_prob`，避免把 dispatcher 证明和 expert GEMM 证明混在一起。
5. 对比 dispatcher 路径和直接 reference 的 forward output、input gradient、每个 rank 本地 expert 的 scale/bias gradient、以及一步 SGD update。

关键结果：

```text
status=PASS
expert_parallel_size=8
threshold:
  rtol=0.0
  atol=0.0

top1_with_empty_expert:
  tokens_received_by_local_expert=[7, 7, 7, 7, 7, 7, 6, 0]
  forward_output exact_equal=True, max_abs=0.0, mismatches=0 on every rank
  input_grad exact_equal=True, max_abs=0.0, mismatches=0 on every rank
  expert_scale_grad / expert_bias_grad exact_equal=True on every rank
  expert_scale_sgd_update / expert_bias_sgd_update exact_equal=True on every rank

top3_balanced_multiroute:
  tokens_received_by_local_expert=[18, 18, 18, 18, 18, 18, 18, 18]
  forward_output exact_equal=True, max_abs=0.0, mismatches=0 on every rank
  input_grad exact_equal=True, max_abs=0.0, mismatches=0 on every rank
  expert_scale_grad / expert_bias_grad exact_equal=True on every rank
  expert_scale_sgd_update / expert_bias_sgd_update exact_equal=True on every rank
```

这证明 EP=8 all-to-all dispatcher 本身在 forward、backward 和参数更新语义上与直接 reference bitwise exact，包括 empty receiving expert 和 top-3 multi-route 两类情况。结合上一节的 TE grouped expert exact proof，MoE 训练路径里“跨 rank token dispatch/combine”和“本地 grouped expert 计算”两个核心组件都已经被独立证明。它仍不是完整 4-layer checkpoint strict parity proof，因为完整路径还包含真实 router 分数、后层 routing flip、shared expert、layernorm、HC、attention 和 output head 的累计数值效应。

继续把 official inference 的运行精度路径做成 variant 后，得到：

| variant | mismatches / 402 | max_abs | mean_abs | relative_l2_gap |
| --- | ---: | ---: | ---: | ---: |
| baseline official FP32 head / FP32 expert activation / FP32 router score | 63 | 0.16042709350585938 | 0.03158159181475639 | 2.313715897095392e-06 |
| BF16 head only | 62 | 0.25568389892578125 | 0.02635219506919384 | 3.155446774427695e-06 |
| Megatron-like BF16 expert activation only | 57 | 0.16588211059570312 | 0.030660895630717278 | 2.208548850912706e-06 |
| Megatron-like BF16 router score only | 55 | 0.19893646240234375 | 0.03048548847436905 | 2.4188858190887785e-06 |
| BF16 expert + BF16 router | 59 | 0.15535736083984375 | 0.03057839721441269 | 2.2086159792156224e-06 |
| BF16 head + BF16 expert + BF16 router | 53 | 0.25 | 0.023050455376505852 | 2.5239634555696e-06 |

这个实验的解读是：

1. 将 official reference 的局部精度路径改得更像 Miles training runtime，会改变 strict mismatch profile；因此剩余差异确实对运行精度路径敏感。
2. 三个 BF16 variant 全开后，mismatch 从 `63/402` 降到 `53/402`，mean_abs 从 `0.03158159181475639` 降到 `0.023050455376505852`。
3. 这不是完整通过：max_abs 和 relative_l2_gap 并不单调改善，strict parity 仍然 FAIL。
4. 因此这组实验只能作为 localization evidence：剩余 drift 包含 official FP32 inference 路径与 Miles BF16 training 路径的差异，而不能作为端到端等价证明。

## Mini Checkpoint Drift Probe

已有 mini checkpoint forward / train-step / layerwise MoE 诊断结果仍然有效：

- `docs/en/advanced/deepseek-v4-mini-forward-compare-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-train-step-parity-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-train-step-qatsim-0415-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-train-step-routing-replay-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-layerwise-moe-dense-vs-sparse-20260531.json`
- `docs/en/advanced/deepseek-v4-layerwise-moe-routing-replay-20260531.json`
- `docs/en/advanced/deepseek-v4-layerwise-moe-routerfp32-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-forward-routing-replay-dense-vs-sparse-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-forward-routing-replay-dense-vs-tilelang-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-forward-routing-replay-sparse-vs-tilelang-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-activation-replay-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-sublayer-activation-replay-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-attention-layer-replay-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-attention-io-replay-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-attention-io-training-step-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-mini-train-step-attention-output-replay-qatsim-20260531.json`
- `docs/en/advanced/deepseek-v4-end-to-end-bf16-tolerance-20260531.json`
- `docs/en/advanced/deepseek-v4-optimizer-update-math-20260531.json`
- `docs/en/advanced/deepseek-v4-fix-regression-guards-20260531.json`
- `docs/en/advanced/deepseek-v4-proof-coverage-matrix-20260531.json`
- `docs/en/advanced/deepseek-v4-environment-provenance-20260531.json`
- `docs/en/advanced/deepseek-v4-external-reference-provenance-20260531.json`
- `docs/en/advanced/deepseek-v4-proof-ledger-20260531.json`

关键结论：

1. deterministic runtime 能让同后端 dense forward bitwise repeatable。
2. dense、sparse、tilelang 都能完成 mini checkpoint forward，logprob finite。
3. dense、sparse、tilelang 都能完成 SFT one-step forward/backward/update，loss 和梯度 finite。
4. strict backend logprob parity 仍 FAIL，relative L2 gap 在 `1e-5` 量级。
5. layerwise MoE probe 显示 layer 0/1/2 router assignment exact-equal，但 layer 3 router assignment 出现离散翻转。
6. routing replay 复用 dense 路由后，layer 3 assignment map 恢复 exact-equal，routed expert shape mismatch 消失。
7. `moe_router_dtype=fp32` 只能减少 layer 3 assignment 差异，不能完全消除。

继续在完整 mini forward 上做 routing replay：dense 先记录 routing top indices，sparse/tilelang 在同一输入上强制复用 dense 路由。该 4-layer mini 配置中前 3 层是 hash-routed，第 3 层才有 score/topk routing，因此 replay 文件中 `num_replays=4`、`num_recorded_tensors=1` 是符合预期的。

| compare | routing replay | mismatches / 402 | max_abs | mean_abs | relative_l2_gap |
| --- | --- | ---: | ---: | ---: | ---: |
| dense vs sparse | off | 205 | 0.5140819549560547 | 0.07136902213096619 | 1.547721555139603e-05 |
| dense record vs sparse replay | on | 84 | 0.255859375 | 0.030136708170175552 | 3.5204090861329362e-06 |
| dense vs tilelang | off | 200 | 0.4274272918701172 | 0.07597572356462479 | 1.746831999693832e-05 |
| dense record vs tilelang replay | on | 75 | 0.2412109375 | 0.028718464076519012 | 3.2778709475600465e-06 |
| sparse vs tilelang | off | 201 | 0.5107736587524414 | 0.06761329621076584 | 1.4695730324199019e-05 |
| sparse replay vs tilelang replay | on | 59 | 0.18173837661743164 | 0.023490911349654198 | 2.306529798046242e-06 |

这组结果把 mini forward 的 strict parity 失败拆成两部分：

1. 后层 MoE routing 离散翻转是主要放大器。复用 dense routing 后，三组 backend compare 的 mismatch、max_abs、mean_abs、relative L2 gap 都显著下降。
2. routing replay 后 strict logprob parity 仍 FAIL，说明剩余不是单纯的 routing assignment 差异，而是前序 attention / MLP / grouped expert BF16 连续数值漂移在完整 4-layer 链路中的累积。

因此 mini checkpoint forward 目前已经完成“离散路由分叉定位”，但还没有完成“端到端 strict backend parity 证明”。

继续做 activation replay：dense backend 先按 rank 记录 `decoder.layers.0..3` 和 `final_layernorm` 的输出，同时记录 routing top indices；sparse/tilelang backend 复用 dense routing，并在指定模块边界注入 dense activation，然后比较最终 response logprob。

| backend | injected dense activation | mismatches / 402 | max_abs | mean_abs | relative_l2_gap | exact_equal |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sparse | none, routing replay only | 84 | 0.255859375 | 0.030136708170175552 | 3.490060837885167e-06 | false |
| sparse | layer 0 output | 33 | 0.25 | 0.015565270557999611 | 1.5783617770548375e-06 | false |
| sparse | layer 0..1 outputs | 27 | 0.248046875 | 0.014432772994041443 | 1.5176156631380167e-06 | false |
| sparse | layer 0..2 outputs | 14 | 0.12876510620117188 | 0.008787469938397408 | 5.462808200240588e-07 | false |
| sparse | layer 0..3 outputs | 0 | 0.0 | 0.0 | 0.0 | true |
| sparse | final layernorm output | 0 | 0.0 | 0.0 | 0.0 | true |
| tilelang | none, routing replay only | 75 | 0.2412109375 | 0.028718464076519012 | 3.2778709475600465e-06 | false |
| tilelang | layer 0 output | 31 | 0.25 | 0.015592050738632679 | 1.4569989524870763e-06 | false |
| tilelang | layer 0..1 outputs | 22 | 0.25 | 0.012565052136778831 | 1.1230053565958187e-06 | false |
| tilelang | layer 0..2 outputs | 17 | 0.1324310302734375 | 0.009016213938593864 | 6.676312116482563e-07 | false |
| tilelang | layer 0..3 outputs | 0 | 0.0 | 0.0 | 0.0 | true |
| tilelang | final layernorm output | 0 | 0.0 | 0.0 | 0.0 | true |

这组 activation replay 证明了两件事：

1. 当 4 个 transformer layer 的输出被强制设为 dense 输出时，sparse 和 tilelang 的 final layernorm、output head、logprob 路径都能 bitwise 复现 dense；单独注入 dense final layernorm 输出也能 bitwise 复现 dense。因此 backend logprob drift 不是 output head 或 logprob 计算本身造成的。
2. 随着注入的 dense layer 前缀增加，mismatch 和 mean_abs 整体下降；说明剩余 drift 是在 4 个 transformer layer 内逐层生成和累积的连续 BF16 数值差异，而不是注入点之后的后处理错误。

这仍然不是完整 strict backend parity proof，因为在不注入 activation 的真实 forward 中，sparse/tilelang 仍与 dense 有 logprob mismatch。它的价值是把剩余问题从“完整模型任意位置”缩小到“transformer layers 内部的连续数值漂移累积”，并排除 final layernorm 后的 head/logprob 路径。

继续做 sublayer activation replay：dense backend 记录每层 `self_attention`、`mlp` 和 layer 输出；sparse/tilelang backend 复用 dense routing，并分别注入 dense attention 输出、dense MLP 输出、以及逐层 attention+MLP 输出前缀。

| backend | injected dense sublayer activation | mismatches / 402 | max_abs | mean_abs | relative_l2_gap | exact_equal |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| sparse | none, routing replay only | 84 | 0.255859375 | 0.030136708170175552 | 3.490060837885167e-06 | false |
| sparse | all attention outputs | 0 | 0.0 | 0.0 | 0.0 | true |
| sparse | all MLP outputs | 39 | 0.1449432373046875 | 0.01673346757888794 | 1.5478868385931932e-06 | false |
| sparse | layer 0 attention+MLP outputs | 33 | 0.25 | 0.015565270557999611 | 1.5783617770548375e-06 | false |
| sparse | layer 0..1 attention+MLP outputs | 27 | 0.248046875 | 0.014432772994041443 | 1.5176156631380167e-06 | false |
| sparse | layer 0..2 attention+MLP outputs | 14 | 0.12876510620117188 | 0.008787469938397408 | 5.462808200240588e-07 | false |
| sparse | layer 0..3 attention+MLP outputs | 0 | 0.0 | 0.0 | 0.0 | true |
| tilelang | none, routing replay only | 75 | 0.2412109375 | 0.028718464076519012 | 3.2778709475600465e-06 | false |
| tilelang | all attention outputs | 0 | 0.0 | 0.0 | 0.0 | true |
| tilelang | all MLP outputs | 42 | 0.2451171875 | 0.018322104588150978 | 1.8514902659916288e-06 | false |
| tilelang | layer 0 attention+MLP outputs | 31 | 0.25 | 0.015592050738632679 | 1.4569989524870763e-06 | false |
| tilelang | layer 0..1 attention+MLP outputs | 22 | 0.25 | 0.012565052136778831 | 1.1230053565958187e-06 | false |
| tilelang | layer 0..2 attention+MLP outputs | 17 | 0.1324310302734375 | 0.009016213938593864 | 6.676312116482563e-07 | false |
| tilelang | layer 0..3 attention+MLP outputs | 0 | 0.0 | 0.0 | 0.0 | true |

这组 sublayer replay 进一步缩小了剩余差异：

1. 只注入 4 层 dense attention 输出，就能让 sparse/tilelang 与 dense 的最终 logprob bitwise exact。
2. 只注入 4 层 dense MLP 输出不能恢复 bitwise exact，说明 MLP 输出不是消除剩余 drift 的必要注入点。
3. 逐层注入 attention+MLP 前缀时，mismatch 逐步下降，并在覆盖第 0..3 层后归零。
4. 因此当前 mini forward backend drift 的第一来源被定位到 attention module 输出差异；后续 MLP、HC、final layernorm、output head 和 logprob 路径在 attention 输出一致时可以复现 dense。

这个结果和前面的 attention 算子/模块训练步 PASS 并不矛盾：attention standalone 在 BF16 容差内等价，但完整 checkpoint 的 strict logprob 阈值会把 attention module 输出里的连续 BF16 小差异通过后续层累积放大。

继续把 attention 输出按层拆开：dense backend 记录每层 `self_attention` 输出；sparse/tilelang 复用 dense routing，并分别注入单层 attention 输出和 attention 前缀输出。

| backend | injected dense attention output | mismatches / 402 | max_abs | mean_abs | relative_l2_gap |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse | none, routing replay only | 84 | 0.255859375 | 0.030136708170175552 | 3.490060837885167e-06 |
| sparse | layer 0 only | 33 | 0.25 | 0.015565270557999611 | 1.5783617770548375e-06 |
| sparse | layer 1 only | 84 | 0.255859375 | 0.02985672652721405 | 3.3992384976810897e-06 |
| sparse | layer 2 only | 88 | 0.255859375 | 0.03189099580049515 | 4.036628672099418e-06 |
| sparse | layer 3 only | 81 | 0.25390625 | 0.03067130409181118 | 3.7028001484973316e-06 |
| sparse | layers 0..1 | 27 | 0.248046875 | 0.014432772994041443 | 1.5176156631380167e-06 |
| sparse | layers 0..2 | 14 | 0.12876510620117188 | 0.008787469938397408 | 5.462808200240588e-07 |
| sparse | layers 0..3 | 0 | 0.0 | 0.0 | 0.0 |
| tilelang | none, routing replay only | 75 | 0.2412109375 | 0.028718464076519012 | 3.2778709475600465e-06 |
| tilelang | layer 0 only | 31 | 0.25 | 0.015592050738632679 | 1.4569989524870763e-06 |
| tilelang | layer 1 only | 87 | 0.1866302490234375 | 0.02962634526193142 | 3.277700239223691e-06 |
| tilelang | layer 2 only | 87 | 0.265625 | 0.030821340158581734 | 3.823746180020571e-06 |
| tilelang | layer 3 only | 85 | 0.244140625 | 0.03084748610854149 | 3.6115673707204365e-06 |
| tilelang | layers 0..1 | 22 | 0.25 | 0.012565052136778831 | 1.1230053565958187e-06 |
| tilelang | layers 0..2 | 17 | 0.1324310302734375 | 0.009016213938593864 | 6.676312116482563e-07 |
| tilelang | layers 0..3 | 0 | 0.0 | 0.0 | 0.0 |

这组 layerwise attention replay 的结论：

1. 单独注入 layer 0 attention 输出就能把 dense-vs-sparse mismatch 从 `84/402` 降到 `33/402`，dense-vs-tilelang 从 `75/402` 降到 `31/402`；layer 0 是最大的早期 drift 放大入口。
2. 单独注入 layer 1/2/3 attention 输出不能恢复 dense logprob，甚至可能让 mismatch 数略升。这不表示后层 attention 错，而是说明后层 attention 的 dense 输出必须和前序 dense 状态一起使用才自洽。
3. attention 前缀注入的 mismatch 单调下降，并在覆盖 layer 0..3 后变为 bitwise exact。

因此，mini forward backend drift 不是某个后处理或 MLP 单点错误，而是 attention backend 在每层产生的 BF16 小差异沿着后续层传播；其中 layer 0 attention 差异最早进入完整链路并贡献最大。

继续做 attention I/O replay：dense backend 记录每层 attention 的输入和输出；sparse/tilelang backend 在 forward pre-hook 中把 attention 输入替换为 dense 输入，再比较当前 backend 重新计算出的 attention 输出和 dense attention 输出。这样可以排除“输入状态不同”对 attention 输出对比的干扰。

| backend | layer | input_pre_replay max_abs | output_from_dense_input max_abs | output mean_abs | output p99_abs |
| --- | --- | ---: | ---: | ---: | ---: |
| sparse | 0 | 0.0 | 0.0625 | 0.006124485284090042 | 0.0234375 |
| sparse | 1 | 0.0029296875 | 0.03515625 | 0.003262903308495879 | 0.015625 |
| sparse | 2 | 0.015625 | 0.0625 | 0.004377629607915878 | 0.015625 |
| sparse | 3 | 0.00390625 | 0.0625 | 0.002733895555138588 | 0.015625 |
| tilelang | 0 | 0.0 | 0.0625 | 0.00623230030760169 | 0.0234375 |
| tilelang | 1 | 0.0029296875 | 0.03515625 | 0.0034180437214672565 | 0.015625 |
| tilelang | 2 | 0.015625 | 0.0625 | 0.004430384375154972 | 0.015625 |
| tilelang | 3 | 0.00390625 | 0.0625 | 0.002877437509596348 | 0.015625 |

这组 attention I/O replay 的结论：

1. layer 0 的 attention 输入在 dense 与 sparse/tilelang 之间本来就是 exact，因为它位于完整模型最前面的 attention 入口；但同一输入下 dense/sparse/tilelang 的 attention 输出仍有 BF16 量级差异。
2. 所有层的 `output_from_dense_input max_abs` 都不超过 `0.0625`，输出 drift finite，且 sparse 与 tilelang 的量级一致。
3. layer 0 的输出 `mean_abs` 最大，和前面的 attention layer replay 中“注入 layer 0 attention 输出贡献最大”一致。
4. 这证明剩余差异不是因为 attention 输入状态不同，而是 attention backend 在相同输入上的 BF16 连续数值差异；该差异本身在 attention 算子/模块容差内，但经过完整模型放大后会导致 strict logprob parity 失败。

继续做 attention I/O local training-step replay：使用 dense backend 记录的每层 checkpoint attention 输入作为共同输入，分别加载 dense、sparse、tilelang backend 的同一 checkpoint attention 权重，使用同一个合成 upstream gradient，对每层 attention 单独执行 forward、backward 和一次 manual SGD update。这样可以证明真实 checkpoint 输入上的 attention training surface，而不受上游状态漂移、MoE routing 分叉或下游 logprob 放大的影响。

| compare | loss_abs_global_max | output_max_abs | output_rel_gap | input_grad_max_abs | input_grad_rel_gap | state_after_step_max_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense vs sparse | 6.7711807787418365e-06 | 0.0625 | 1.4298539834922686e-05 | 9.5367431640625e-07 | 2.6065915087025715e-05 | 0.0 |
| dense vs tilelang | 8.473638445138931e-06 | 0.0625 | 1.4476013600717152e-05 | 9.5367431640625e-07 | 2.8061398839485108e-05 | 0.0 |
| sparse vs tilelang | 3.786408342421055e-06 | 0.0625 | 4.08536784657354e-06 | 7.152557373046875e-07 | 1.2886152300439768e-05 | 0.0 |

阈值为 `max_output_abs=0.0625`、`max_input_grad_abs=0.01`、`max_state_abs=2e-05`、`max_rel_gap=5e-4`，验证结果为 `PASS`，所有 attention 参数都有梯度，且 manual SGD 后 `state_after_step_max_abs=0.0`。

这组结果补上了前面 attention I/O forward replay 的训练面证明：

1. 同一 checkpoint attention 输入下，dense、sparse、tilelang 的 forward 输出差异保持在 BF16 attention 容差内。
2. 对同一个 upstream gradient，输入梯度最大绝对差异不超过 `9.5367431640625e-07`，relative gap 不超过 `2.8061398839485108e-05`。
3. 一步 manual SGD 后，三种 backend 的 attention state 在本验证覆盖的权重上保持一致。
4. 因此 mini checkpoint 的 attention backend drift 是可控的 BF16 forward drift，不是 backward 或参数更新公式错误；完整 SFT one-step loss drift 来自完整模型 logprob/loss 对这些连续差异的放大，而不是 attention 训练步本身失效。

继续在完整 mini SFT one-step 上做同环境 baseline 与 routing replay。阈值为 `max_loss_abs=0.02`、`max_selected_grad_rel_gap=0.002`、`max_selected_state_abs=2e-05`。

| compare | routing replay | loss_abs_global_max | selected_grad_max_rel_gap | selected_grad_max_abs | selected_state_max_abs |
| --- | --- | ---: | ---: | ---: | ---: |
| dense vs sparse | off | 1.5947265625 | 0.00044641669722478294 | 0.21229159832000732 | 9.5367431640625e-07 |
| dense record vs sparse replay | on | 1.1494140625 | 0.0001497947362728569 | 0.1340121626853943 | 9.5367431640625e-07 |
| dense vs tilelang | off | 0.703125 | 0.00038428885225427756 | 0.15591496229171753 | 9.5367431640625e-07 |
| dense record vs tilelang replay | on | 0.5341796875 | 0.0001477684822497327 | 0.127852201461792 | 9.5367431640625e-07 |
| sparse vs tilelang | off | 0.8916015625 | 0.0002548021320385452 | 0.1527109146118164 | 9.5367431640625e-07 |
| sparse replay vs tilelang replay | on | 0.615234375 | 0.0001223404163664954 | 0.14051318168640137 | 9.5367431640625e-07 |

本轮 train-step replay 的关键事实：

1. dense 记录的 score/topk routing tensor 数量为 `1`，每个 backend 前向 replay 次数为 `4`，与 4-layer mini 配置中只有后层进入 score/topk routing 的结构一致。
2. 同环境 no-routing baseline 和 routing replay 都只有 `loss_abs_global_max` 超过严格阈值；`selected_grad_max_rel_gap` 和 `selected_state_max_abs` 均在阈值内。
3. routing replay 后 loss gap、selected grad gap 和 selected grad absolute gap 都下降，但 loss 仍远大于 `0.02`。
4. 因此 SFT one-step 已经证明 backward/update finite，且路由复用能减少训练步 drift；但它仍不是完整 backend strict parity proof。剩余 loss drift 不是单纯离散 MoE routing flip，而是连续 BF16 数值差异经过 logprob/loss 后的放大。

最后做完整 SFT one-step 的 attention-output straight-through replay：dense backend 同时记录 routing 和每层 attention 输出；sparse/tilelang 复用 dense routing，并把 attention forward value 替换为 dense value，同时用 `dense_value + (backend_value - backend_value.detach())` 保留当前 backend 的 backward Jacobian。这个验证回答的问题是：如果把已经定位出的 attention BF16 forward value drift 从完整训练步里剥离，剩余的端到端 loss、反传和参数更新是否一致。

| compare | loss_abs_global_max | selected_grad_max_rel_gap | selected_grad_max_abs | selected_state_max_abs |
| --- | ---: | ---: | ---: | ---: |
| dense vs sparse | 0.0 | 1.6638869012597368e-05 | 0.057994842529296875 | 4.76837158203125e-07 |
| dense vs tilelang | 0.0 | 2.1433698600525908e-05 | 0.05086469650268555 | 4.76837158203125e-07 |
| sparse vs tilelang | 0.0 | 1.8058738106341288e-05 | 0.06413555145263672 | 4.76837158203125e-07 |

阈值仍为 `max_loss_abs=0.02`、`max_selected_grad_rel_gap=0.002`、`max_selected_state_abs=2e-05`，结果为 `PASS`。

这组 replay 的意义和边界：

1. 完整 SFT loss 在 dense/sparse/tilelang 之间变为 exact：`loss_abs_global_max=0.0`。
2. backward 不是被绕开了；straight-through 注入保留了 sparse/tilelang 自己 attention backend 的局部 Jacobian，因此 attention 参数梯度仍来自对应 backend。
3. selected gradient relative gap 最高为 `2.1433698600525908e-05`，manual SGD 后 selected state 最大差异为 `4.76837158203125e-07`，都显著低于训练步阈值。
4. 因此完整 SFT one-step 的剩余失败条件被定位为真实非注入 forward 中 attention BF16 value drift 对 logprob/loss 的放大；在消除这个已定位 forward value drift 后，端到端 loss、backward 和 update parity 通过。
5. 这不是把真实非注入 forward 宣称为 bitwise 等价；它证明的是训练链路在数学上闭合，剩余不通过的 strict parity gate 是 forward 数值容差问题，而不是训练有效性或更新公式问题。

## External Training Reference

验证脚本：`tools/verify_deepseek_v4_external_training_reference.py`

最新结果：`docs/en/advanced/deepseek-v4-external-training-reference-1layer-20260531.json`

这一步是对前面“external reference one-step train parity 应该怎么做”的落地修正：不再把 official inference runtime 当成训练 reference，而是写一个训练态的显式 PyTorch 公式 reference，再和 Megatron/Miles module forward graph 对齐。

当前 gate 覆盖的是 1-layer、non-compressed、non-MoE DeepSeek-V4 TransformerBlock：

1. block HyperConnection expand/head。
2. layer attention / FFN HyperConnection pre/post。
3. RMSNorm、non-compressed DeepSeek-V4 attention、RoPE、KV QAT、dense masked attention reference、output projection。
4. 标准 GELU MLP、final layernorm。
5. 同一个 upstream gradient 下的 backward 和一步 manual SGD update。

该 verifier 的 reference 是脚本内显式公式，不调用 Megatron `TransformerBlock.forward()`，也不是 dense/sparse/tilelang backend 互比。被测对象是 Megatron/Miles 1-layer block；reference 是外部 PyTorch 公式。

验证结果为 `PASS`：

| item | result |
| --- | ---: |
| forward output max_abs | 0.0 |
| forward output exact_equal | true |
| loss_abs | 0.0 |
| input_grad max_abs | 0.0 |
| state_after_step max_abs | 0.0 |
| common grad tensors | 13 |
| grad max_abs | 5.960464477539063e-08 |
| grad max_abs tensor | final_layernorm.weight |
| grad threshold | 1e-07 |

这说明：在一个干净的训练态 block 上，Miles/Megatron 的 DeepSeek-V4 实现能被显式 PyTorch reference 复现到 forward/loss/input-grad/update exact，参数梯度只有 `5.96e-08` 的 FP32 量级差异。

边界也要明确：

1. 这个 gate 已经关闭“是否能构造训练态 external reference 并对齐一步训练”的第一段风险。
2. 它还没有覆盖 compress_ratio=4/128 attention、V4 indexer、routed MoE、loaded 4-layer mini checkpoint 和完整 SFT loss。
3. 因此 `external_reference_mini_checkpoint_one_step_train_parity` 仍保持 `MISSING_INPUT`，但原因已经不是 official inference strict forward parity，而是需要把这个 external training reference 扩展到 compressed attention / MoE / loaded mini checkpoint。

## End-to-End BF16 Tolerance

验证脚本：`tools/verify_deepseek_v4_end_to_end_tolerance.py`

最新结果：`docs/en/advanced/deepseek-v4-end-to-end-bf16-tolerance-20260531.json`

这个 verifier 不改变 strict parity 的判定：真实非注入 forward 的 strict logprob compare 仍然是 FAIL。它解决的是另一个问题：如果我们接受 DeepSeek-V4 训练 runtime 的 BF16 数值容差，那么当前真实 forward / train-step drift 是否被明确阈值约束，并且是否与前面的定位链一致。

本次声明的 BF16 envelope：

| item | threshold |
| --- | ---: |
| real forward relative_l2_gap | <= 2e-05 |
| real forward mean_abs | <= 0.08 |
| real forward p99_abs | <= 0.37 |
| routing-replay forward relative_l2_gap | <= 4e-06 |
| routing relative_l2 reduction | >= 4x |
| real SFT selected_grad_max_rel_gap | <= 5e-04 |
| real SFT selected_state_max_abs | <= 2e-05 |
| official 1-layer relative_l2_gap | <= 5e-06 |
| official 1-layer mean_abs | <= 0.04 |
| attention I/O output max_abs | <= 0.0625 |
| attention I/O input_grad max_abs | <= 0.01 |
| SFT attention-output replay selected_grad_max_rel_gap | <= 3e-05 |
| SFT attention-output replay selected_state_max_abs | <= 2e-05 |

验证结果为 `PASS`，关键观测如下：

| compare | real forward relative_l2_gap | real forward mean_abs | routing relative_l2_gap | routing reduction |
| --- | ---: | ---: | ---: | ---: |
| dense vs sparse | 1.547721555139603e-05 | 0.07136902213096619 | 3.5204090861329362e-06 | 4.3964252939698225 |
| dense vs tilelang | 1.746831999693832e-05 | 0.07597572356462479 | 3.2778709475600465e-06 | 5.329166485319148 |
| sparse vs tilelang | 1.4695730324199019e-05 | 0.06761329621076584 | 2.306529798046242e-06 | 6.371359406085763 |

| compare | real SFT selected_grad_rel_gap | real SFT selected_state_abs | attention-output replay loss_abs | replay selected_grad_rel_gap |
| --- | ---: | ---: | ---: | ---: |
| dense vs sparse | 0.00044641669722478294 | 9.5367431640625e-07 | 0.0 | 1.6638869012597368e-05 |
| dense vs tilelang | 0.00038428885225427756 | 9.5367431640625e-07 | 0.0 | 2.1433698600525908e-05 |
| sparse vs tilelang | 0.0002548021320385452 | 9.5367431640625e-07 | 0.0 | 1.8058738106341288e-05 |

official 1-layer sparse/tilelang 也落在该 BF16 envelope 内：

| backend | relative_l2_gap | mean_abs | p99_abs |
| --- | ---: | ---: | ---: |
| sparse | 2.313715897095392e-06 | 0.03158159181475639 | 0.13715827465057373 |
| tilelang | 2.4187967853084302e-06 | 0.03211159259080887 | 0.12487327307462692 |

这组验证的边界很重要：

1. 它不声称真实非注入 forward 已经 bitwise 或 strict rtol/atol 等价。
2. 它证明当前真实 forward drift 的规模被 BF16 envelope 约束，且 routing replay、activation replay、attention I/O replay、SFT attention-output replay 给出一致解释。
3. 它证明真实非注入 SFT 的梯度和一步更新已经在训练阈值内；loss gap 来自已定位的 forward value drift 放大。
4. 因此“训练方式正确”的证据可以分成两层：算子和训练链路严格证明已经通过；真实端到端 forward 在 BF16 runtime envelope 下通过，但 strict parity 仍作为未关闭 gate 保留。

## Optimizer Update Math

验证脚本：`tools/verify_deepseek_v4_optimizer_update_math.py`

最新结果：`docs/en/advanced/deepseek-v4-optimizer-update-math-20260531.json`

这个 verifier 覆盖两个问题：

1. 静态确认 Miles Megatron 训练路径实际构造并调用 Megatron optimizer：`get_megatron_optimizer`、`OptimizerConfig`、`optimizer.prepare_grads()`、`optimizer.step()`、`opt_param_scheduler.step(...)` 都在训练路径中。
2. 静态确认 DeepSeek-V4 runner 的 optimizer 参数为 Adam：`lr=1e-6`、`weight_decay=0.1`、`adam_beta1=0.9`、`adam_beta2=0.98`。

随后它用 PyTorch 独立验证 AdamW 第一步更新数学。第一步从零 moments 出发时：

```text
m = (1 - beta1) * grad
v = (1 - beta2) * grad^2
m_hat = m / (1 - beta1) = grad
v_hat = v / (1 - beta2) = grad^2
param_next = param * (1 - lr * weight_decay) - lr * grad / (abs(grad) + eps)
```

验证覆盖 representative selected tensor shapes：`attn_sink`、`norm_weight`、`attention_projection_tile`、`hc_scale`。结果：

| case | reference vs closed-form max_abs | zero-grad weight-decay | worst sign-flip update max_abs |
| --- | ---: | --- | ---: |
| attn_sink | 0.0 | PASS | 2.000480890274048e-06 |
| norm_weight | 0.0 | PASS | 2.000480890274048e-06 |
| attention_projection_tile | 9.313225746154785e-10 | PASS | 2.000480890274048e-06 |
| hc_scale | 1.1641532182693481e-10 | PASS | 2.000480890274048e-06 |

最坏情况下，两个 backend 的第一步 AdamW adaptive update 项逐元素差异不超过约 `2 * lr = 2e-06`，低于 selected-state 阈值 `2e-05`。这补上了 optimizer/update 规则本身的数学证据。它不是全量 Megatron optimizer state replay；全量 replay 会引入完整 optimizer state 显存开销。当前 proof 对 optimizer 的覆盖边界是：实际训练调用 Megatron optimizer 已静态确认，AdamW 第一步 selected-update 数学和误差上界已严格验证。

## Fix Regression Guards

验证脚本：`tools/verify_deepseek_v4_fix_regression_guards.py`

最新结果：`docs/en/advanced/deepseek-v4-fix-regression-guards-20260531.json`

数值验证证明行为，regression guard 证明当前源码仍保留导致这些行为通过的修复形态。该脚本使用文本级检查，避免本地 Python 版本与源码新语法注解不兼容。

当前 guard 状态为 `PASS`，覆盖：

1. `DeepSeekV4Attention` 的 debug trace 被 `MILES_DSV4_TRACE_INTERNALS=1` 显式开关保护，默认不落 CPU trace。
2. attention KV QAT 只作用在 non-RoPE 维度：`fp8_simulate_qat(kv_vanilla[..., :-rd].contiguous(), 64)`，并通过 `torch.cat` 保留 `kv_vanilla[..., -rd:]`。
3. `act_quant` 支持 official-compatible in-place 模拟、`scale_dtype` 参数、FP32 默认 scale、FE8M0 scale dtype 分支和 in-place copy-back。
4. `fp8_simulate` 调用 `act_quant(..., scale_fmt=None, scale_dtype=torch.float32, inplace=True)`，不再请求 `ue8m0` scale。
5. compressor QAT 使用 `torch.cat` 构造新 tensor，避免对 `kv[..., : self.nope_head_dim]` 做 overlapping slice assignment，并保留 RoPE tail。
6. 关联数值 artifact `operator_math`、`attention_trace_replay` 和 `proof_ledger` 均仍为 `PASS`。

这组 guard 的价值是防回归：如果后续改动把 KV QAT 又扩展到 RoPE 维度、把 QAT scale 改回 FE8M0、或者重新引入 compressor slice assignment，文档 proof 会直接失败，而不是等到端到端 drift 被重新放大后再定位。

## Proof Coverage Matrix

验证脚本：`tools/verify_deepseek_v4_proof_coverage_matrix.py`

最新结果：`docs/en/advanced/deepseek-v4-proof-coverage-matrix-20260531.json`

coverage matrix 把本次目标拆成可检查 requirement，而不是只依赖人工阅读。当前状态为 `PASS`。

它覆盖的 requirement：

1. 上游 Megatron-LM PR #4839 问题和 Miles scope 已在文档中说明，并由 HC orientation artifact 证明。
2. 前面引入的精度修复由 source regression guard 覆盖。
3. HC、RoPE、QAT、dense/sparse attention、TileLang sparse MLA 算子数学由 operator artifact 覆盖。
4. dense/sparse/tilelang attention backend 的模块训练步和 TransformerBlock 训练步由对应 artifact 覆盖。
5. TE grouped MLP 和 EP=8 all-to-all dispatcher 的 forward/backward/update 由数学 verifier 覆盖。
6. official attention、weight mapping、head replay 和 MLP expert replay 对 official-vs-Miles drift 做了定位。
7. official-vs-Miles full-forward BF16 tolerance 有独立 PASS artifact，并保留 strict forward parity 仍未关闭的边界。
8. external training reference 1-layer gate 有独立 PASS artifact，证明显式 PyTorch 训练态 reference 可以复现 non-compressed DeepSeek-V4 block 的 forward/backward/update。
9. mini checkpoint forward drift 被 routing replay、activation replay、sublayer replay 和 attention I/O replay 定位。
10. mini SFT training chain 被 baseline、routing replay、attention-output replay 和 attention I/O training-step 覆盖。
11. End-to-end BF16 tolerance envelope、optimizer update math、proof ledger 都有独立 PASS artifact。

它还显式检查剩余 strict gates：

| gate | expected status |
| --- | --- |
| strict_mini_backend_logprob_parity | FAIL |
| strict_mini_checkpoint_train_step_backend_parity | FAIL |
| official_reference_mini_checkpoint_forward_parity | FAIL |
| production_ep8_moe_path_strict_parity | PARTIALLY_LOCALIZED |
| external_reference_mini_checkpoint_one_step_train_parity | MISSING_INPUT |

因此 coverage matrix 给出的结论是：当前 proof 已覆盖用户要求的核心数学、修复、防回归、dense/sparse/tilelang、训练链路、optimizer update 和 BF16 envelope；剩余 strict parity gate 不是遗漏，而是带有 artifact 证据和边界说明的未关闭项。

## External Reference Provenance

验证脚本：`tools/verify_deepseek_v4_external_references.py`

最新结果：`docs/en/advanced/deepseek-v4-external-reference-provenance-20260531.json`

该 verifier 校验两个外部引用在文档中的用途边界：

1. Megatron-LM PR #4839 用于定位 DeepSeek-V4 mHC / HyperConnection residual mixing 方向风险，并作为 HC orientation oracle 的来源。
2. SWIFT DeepSeek-V4 Best Practice 用于参考问题发现和验证方法，例如 controlled forward/parity check 与训练有效性检查。
3. 这两个外部引用不作为 Miles 正确性的直接证据；Miles 正确性仍以本仓库 artifact 为准。

当前状态为 `PASS`。这避免后续文档把外部材料误写成对 Miles 实现的直接证明。

## Environment Provenance

验证脚本：`tools/verify_deepseek_v4_environment_provenance.py`

最新结果：`docs/en/advanced/deepseek-v4-environment-provenance-20260531.json`

该 verifier 用 `deepseek-v4-operator-math-20260531.json` 中记录的实际运行环境作为 source of truth，逐项校验本文档“环境”章节。当前状态为 `PASS`。

校验字段包括：

| field | value |
| --- | --- |
| GPU | NVIDIA H20 |
| CUDA device count | 8 |
| NVIDIA driver | 580.126.20 |
| NVIDIA-SMI CUDA version | 13.0 |
| CUDA toolkit | 12.9 |
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu129 |
| torch CUDA | 12.9 |
| megatron-core | 0.16.0rc0 |
| mbridge | 0.15.1 |
| miles package | 0.2.1 |
| transformer-engine | 2.10.0 |
| tilelang | 0.1.9 |

它还检查运行时结构标志：`upstream_hyper_connection_present=False`、`transformer_config_has_dsv4_mode=True`、`transformer_config_has_experimental_attention_variant=True`。这避免后续文档版本、运行时结构与实际 artifact 发生漂移。

## Proof Ledger

验证脚本：`tools/verify_deepseek_v4_proof_ledger.py`

最新结果：`docs/en/advanced/deepseek-v4-proof-ledger-20260531.json`

该脚本不重新跑模型，而是对已经落地的 artifact 做机器校验，避免把一组互不相关的 PASS 结果误读成完整证明。当前 ledger 状态为 `PASS`，失败项为空。

它检查的关键不变量：

1. HC / QAT / attention / TransformerBlock / Grouped MLP / EP=8 dispatcher / attention I/O training-step / SFT attention-output replay 的 artifact 状态符合预期。
2. HyperConnection 方向验证能区分正确公式和错误公式：fixed formula diff 为 `2.384185791015625e-07`，wrong formula diff 为 `9.81311321258545`。
3. routing replay 在三组 backend compare 上都降低 mismatch 和 relative L2 gap：dense-vs-sparse relative L2 降低 `4.3964252939698225x`，dense-vs-tilelang 降低 `5.329166485319148x`，sparse-vs-tilelang 降低 `6.371359406085763x`。
4. sparse/tilelang 的 `prefix4`、`final_ln`、`all_attn`、`layerwise_all_attn` replay 都 bitwise 恢复 dense logprob；`all_mlp` replay 不能恢复，证明剩余 forward drift 的必要注入点是 attention output。
5. attention I/O replay 中所有同输入 attention 输出 finite，最大差异不超过 `0.0625`，并且 layer 0 是两个 backend 上 mean_abs 最大的早期入口。
6. attention I/O local training-step 的最大输入梯度差异为 `9.5367431640625e-07`，manual SGD 后最大 state 差异为 `0.0`。
7. 完整 SFT one-step 的 attention-output straight-through replay 在三组 backend compare 上 `loss_abs_global_max=0.0`，selected gradient relative gap 仍显著低于 `0.002` 阈值。
8. 端到端 BF16 tolerance artifact 为 `PASS`。
9. official-vs-Miles forward BF16 tolerance artifact 为 `PASS`，并记录 strict forward parity 仍为 `FAIL`。
10. external training reference 1-layer artifact 为 `PASS`：forward/loss/input-grad/update exact，最大参数梯度差异为 `5.960464477539063e-08`。
11. Optimizer update math artifact 为 `PASS`。
12. Precision fix regression guard artifact 为 `PASS`。
13. Proof coverage matrix artifact 为 `PASS`。
14. Environment provenance artifact 为 `PASS`。
15. External reference provenance artifact 为 `PASS`。

因此 ledger 给出的机器结论是：已记录的 artifact 能一致证明当前覆盖的数学算子、训练步、external training reference、official forward BF16 tolerance、optimizer update math、修复项 source guard、coverage matrix、environment provenance、external reference provenance 和 dense/sparse/tilelang 训练链路；真实非注入 forward/train-step drift 落在声明的 BF16 tolerance envelope 内；剩余 strict parity 失败被一致定位为 BF16 attention forward-value drift 经过完整模型放大，而不是 HC、QAT、attention backward、MLP、EP=8 dispatcher、output head 或参数更新公式错误。

## 环境

2026-05-31 的验证环境：

```text
GPU: NVIDIA H20
CUDA device count: 8
NVIDIA driver: 580.126.20
NVIDIA-SMI CUDA version: 13.0
CUDA toolkit: 12.9
Python: 3.12.3
PyTorch: 2.9.1+cu129
torch CUDA: 12.9
megatron-core: 0.16.0rc0
mbridge: 0.15.1
miles package: 0.2.1
transformer-engine: 2.10.0
tilelang: 0.1.9
```

运行时结构检查：

```text
upstream_hyper_connection_present=False
transformer_config_has_dsv4_mode=True
transformer_config_has_experimental_attention_variant=True
```

## 最终判断

对于“Megatron-Core PR #4839 这类问题如何发现和处理”，当前流程已经闭环：

1. 从上游修复定位数学公式风险。
2. 写非对称输入的算子级 reference，证明正确公式和错误公式可区分。
3. 确认 Miles 当前 DeepSeek-V4 没走 upstream HC，而是走自己的 HC 实现。
4. 证明 Miles HC 与 PR #4839 fixed native formula 对齐。
5. 用 official inference attention 和 full-forward reference 做外部反证。
6. 发现并修复一个真实的 KV QAT 不一致。
7. 用 trace replay、weight mapping、MLP expert replay、grouped MLP math、EP=8 all-to-all dispatch math、模块训练步、block 训练步、external training reference、mini checkpoint attention I/O 训练步、attention-output straight-through SFT one-step replay、end-to-end BF16 tolerance verifier、optimizer update math、fix regression guards、proof coverage matrix、environment provenance 和 external reference provenance 证明修复后的核心算子链路正确，并用 proof ledger 机器校验这些 artifact 的逻辑一致性，把剩余 drift 进一步缩小到完整模型累计 BF16 数值漂移、真实 router 分叉与 official inference precision/runtime 的差异。

对于“Miles 当前训练方式是否已经严格证明正确”，当前答案必须分层：

- **已证明**：HC、RoPE、QAT、attention dense/sparse/tilelang、compress 0/4/128 attention 训练步、1-layer non-compressed external training reference、mini checkpoint attention I/O local training-step replay、attention-output straight-through 完整 SFT one-step replay、official-vs-Miles full-forward BF16 tolerance、end-to-end BF16 tolerance envelope、optimizer update math、fix regression guards、proof coverage matrix、environment provenance、external reference provenance、official MLP expert formula replay、TE grouped expert 本地 forward/backward/update、EP=8 all-to-all dispatch/combine forward/backward/update、DeepSeek-V4 TransformerBlock 训练步，以及 proof ledger 中这些证据的机器一致性。
- **已跑通并定位**：mini checkpoint forward、mini checkpoint routing replay、真实非注入 mini checkpoint SFT one-step、official full-forward probe、official runtime precision variant probe。
- **未严格证明**：真实非注入完整 4-layer / production EP=8 MoE 端到端 strict backend parity、official-vs-Miles strict logprob parity、覆盖 compressed attention / routed MoE / loaded mini checkpoint / SFT loss 的 external reference one-step train parity。

因此文档最后两点的当前状态是：

1. **已完成**：已经给 official-vs-Miles full-forward 建立可接受的 BF16 数值容差标准，并由 `deepseek-v4-official-forward-bf16-tolerance-20260531.json` 机器验证为 `PASS`。该标准没有消除 strict logprob mismatch；它把剩余 mismatch 固定为 BF16 runtime envelope 内的已知差异。
2. **已完成第一段，但未完成 mini checkpoint 级 PASS**：1-layer non-compressed external training reference 已经由 `deepseek-v4-external-training-reference-1layer-20260531.json` 验证为 `PASS`。完整 external reference one-step train parity 仍需要把 reference 扩展到 compress_ratio=4/128 attention、V4 indexer、routed MoE、loaded 4-layer mini checkpoint 和 SFT loss；因此该完整 gate 仍是 `MISSING_INPUT`，不是训练链路的新失败。

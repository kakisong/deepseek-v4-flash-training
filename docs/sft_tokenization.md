# V4-Flash SFT 训练数据是如何被 tokenize 的

**范围。** 本文档描述 Kaynzhang-077 SFT 运行所采用的精确 tokenize + loss-mask 路径
(`--loss-mask-type deepseek_v4`、`--qkv-format thd`)。以下全部内容均读自代码,并
**在真实样本上验证过**(在 worker pod 上运行,仅 CPU)。该 mask 与训练逐比特一致
(`tools/verify_sft_pipeline.py`:总 token 数能逐字节复现线上 step-0)。

**TL;DR。** 每一段对话由一个 Python chat template 模块
(`encoding_dsv4.py`,*而非* jinja)渲染成一个字符串,带 offset-mapping **一次性** tokenize,并且
**仅对落在某个 `assistant` message 渲染区间内的 token** 设置 `loss_mask=1`(reasoning + content +
tool_calls + EOS)。其余一切 —— system、user、`tool_result`,以及所有 role/transition 特殊
token —— 都被 mask 为 0。随后仅在 `mask==1` 的 token 上计算 next-token CE。

---

## 1. Tokenizer
- **Class** `PreTrainedTokenizerFast`(Rust BPE),**vocab 129,280**,`model_max_length` 1,048,576。
- **Special tokens**(DeepSeek 全角竖线 `｜`):
  - BOS `<｜begin▁of▁sentence｜>`,EOS = PAD `<｜end▁of▁sentence｜>`
  - role `<｜User｜>`、`<｜Assistant｜>`;thinking `<think>` / `</think>`
  - tool-call 标记 `<｜DSML｜tool_calls>…<｜DSML｜invoke name="…">…`
- `tokenizer_config.json` 中的 `chat_template` 是**空**的 —— V4 把模板以代码形式发布
  (HF 目录内的 `encoding/encoding_dsv4.py`),因此 `apply_chat_template` **不是** jinja 路径。

## 2. 渲染(messages → 一个字符串)
`miles/utils/mask_utils.py: gen_multi_turn_loss_mask_deepseek_v4`(第 135–225 行):
1. 从 `tokenizer.name_or_path/encoding/` 懒加载 `encoding_dsv4.py`(`:156-169`)。
2. 预处理:`merge_tool_messages` + `sort_tool_results_by_call_order`(`:178-179`)。
3. 从 `bos_token` 开始,然后用 `render_message(i, …)` **逐条逐段渲染每个 message**,
   记录每个 message 的字符区间 `(start, end, role)`(`:186-194`)。
   - assistant 模板 = `{reasoning}{content}{tool_calls}` + EOS(`encoding_dsv4.py:45`)。
   - tool docs 通过 `TOOLS_TEMPLATE` 注入;`tool_result` 渲染在一个 `<｜User｜>` 轮次之下。
4. **Thinking mode**(本次运行:`MILES_DSV4_THINKING_MODE=chat`、`MILES_DSV4_DROP_THINKING=0`):
   reasoning 被**保留并参与训练**;携带 `tools` 的样本强制 `drop_thinking=False`。

## 3. Tokenize + loss mask(核心规则)
```python
enc_out = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)  # mask_utils.py:197
```
- `add_special_tokens=False` —— specials 已经在渲染后的文本里;tokenizer 不能再额外添加。
- `return_offsets_mapping=True` —— 给每个 token 提供其字符区间 `(s, e)`。
- **设置 mask=1**(`:202-210`):当且仅当一个 token 的 `(s, e)` 完全落在某个
  `role=="assistant"` 区间 `[start, end)` 内时,该 token 才参与训练。带 `(0,0)` 的 token(无 offset 的 specials)会被跳过 → 保持为 0。
- **逐轮覆盖**(`:214-223`):带 `step_loss_mask=0` 的 assistant 轮次会被重新归零。

结果:`mask=1` = assistant reasoning + content + tool_calls + EOS;`mask=0` = system / user /
`tool_result` + 所有 role/transition token(`<｜Assistant｜>`、`<think>`、`</think>`、…)。

下游的 loss(`loss.py:88-93`)施加标准的位移 `logits[t-1] → token[t]`,因此 CE 落在
每个被 mask 的 assistant token 上(基于其前缀)。逐 token 归一化除以真实的
`Σ loss_mask`(`loss.py:869`)。参见 `docs/sft_loss_verification.md`。

## 4. 数据流(jsonl → model)
```
albaliang_077_le128k.jsonl  (per line: {messages, tools, token_length})
  → Dataset stores raw messages verbatim (apply_chat_template=False)
  → each rollout: random.seed(42 + epoch) shuffle, take 128 samples
  → per sample: get_loss_mask() → (token_ids, loss_mask)
  → dynamic packing to token_cap 196,608 / microbatch (packing_efficiency ~0.94–0.96)
  → --qkv-format thd (variable-length packed, not padded rectangles) → Megatron, CP6 zigzag split
```

## 5. 验证过的真实样例(仅对 assistant 做 masking)
`M` = 参与训练(计入 loss),`.` = 被 mask。由真实的 `(token_ids, loss_mask)` 解码得到。

**(a) 简单对话** —— openhermes,ratio 0.60:
```
[. 12t] <｜begin▁of▁sentence｜><｜User｜>Do you know any jokes about librarians?<｜Assistant｜></think>
[M 18t] Why do librarians like the wind? It says, "Shhh!" all day!<｜end▁of▁sentence｜>
```

**(b) 长 prompt、短 answer** —— classification,ratio 0.13(解释了为何有大量低 ratio 样本):
```
[. 87t] <｜begin▁of▁sentence｜>Analyze if this message indicates a new conversation topic...
[M 13t] {"isNewTopic": false, "title": null}<｜end▁of▁sentence｜>
```

**(c) 多轮 tool use** —— 真实的 agent-SFT 形态,ratio 0.37:
```
[. 425t] <｜begin▁of▁sentence｜>You are a file search specialist...        ← system prompt (masked)
[M 139t] \n\n我先搜索 controller 目录...<｜DSML｜tool_calls>...               ← assistant turn 1 (reasoning+tool_call, trained)
[.  15t] <｜User｜><tool_result>Found 0 files</tool_result><｜Assistant｜></think>  ← tool result (masked, = user input)
[M 119t] \n\n<｜DSML｜tool_calls><｜DSML｜invoke name="search_file">...           ← assistant turn 2 (trained)
```
要点:只有 assistant 生成的内容会参与训练;`tool_result` 是上下文(渲染为一个 `<｜User｜>`
轮次)且被 mask,所有 role/transition token 也被 mask。

## 6. 一个测量陷阱
每行 jsonl 携带一个 `token_length` 字段,它由一个**不同的计数器**计算得到,相比本生产 tokenizer
**多算了约 20K token**。(正是这个 metadata 导致 332 个样本因 ">128K" 被从 le128k 中过滤掉,
而它们在这里实际只 tokenize 到 55K–131K —— 全部都在窗口内。)**对于"一个样本有多长"这个问题,
请相信上面 `tokenizer(text, …)` 的路径,而不是 metadata 字段。** 参见
`docs/postrun_eval_runbook.md` / memory `sft-holdout-eval-prep`。

---
### 来源
读自代码并于 2026-06-07 在真实样本上验证(worker pod,仅 CPU,无 GPU,不干扰线上运行)。
关键代码:`miles/utils/mask_utils.py:135-225`、`encoding_dsv4.py`、
`miles/backends/training_utils/loss.py:88-93,869`。配置:run.sh `--loss-mask-type deepseek_v4`、
`MILES_DSV4_THINKING_MODE=chat`。相关:`docs/sft_loss_verification.md`。

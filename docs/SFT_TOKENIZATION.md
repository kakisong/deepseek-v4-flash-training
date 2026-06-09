# How V4-Flash SFT Training Data Is Tokenized

**Scope.** The exact tokenization + loss-mask path for the Kaynzhang-077 SFT run
(`--loss-mask-type deepseek_v4`, `--qkv-format thd`). Everything below was read from code and
**verified on real samples** (run on a worker pod, CPU-only). The mask is bit-identical to training
(`tools/verify_sft_pipeline.py`: total tokens reproduce live step-0 to the byte).

**TL;DR.** Each conversation is rendered to one string by a Python chat-template module
(`encoding_dsv4.py`, *not* jinja), tokenized **once** with offset-mapping, and `loss_mask=1` is set
**only on tokens that fall inside an `assistant` message's render span** (reasoning + content +
tool_calls + EOS). Everything else — system, user, `tool_result`, and all role/transition special
tokens — is masked to 0. Next-token CE is then computed over `mask==1` tokens only.

---

## 1. Tokenizer
- **Class** `PreTrainedTokenizerFast` (Rust BPE), **vocab 129,280**, `model_max_length` 1,048,576.
- **Special tokens** (DeepSeek fullwidth bar `｜`):
  - BOS `<｜begin▁of▁sentence｜>`, EOS = PAD `<｜end▁of▁sentence｜>`
  - roles `<｜User｜>`, `<｜Assistant｜>`; thinking `<think>` / `</think>`
  - tool-call markup `<｜DSML｜tool_calls>…<｜DSML｜invoke name="…">…`
- `tokenizer_config.json`'s `chat_template` is **empty** — V4 ships the template as code
  (`encoding/encoding_dsv4.py` inside the HF dir), so `apply_chat_template` is **not** the jinja path.

## 2. Rendering (messages → one string)
`miles/utils/mask_utils.py: gen_multi_turn_loss_mask_deepseek_v4` (lines 135–225):
1. Lazy-load `encoding_dsv4.py` from `tokenizer.name_or_path/encoding/` (`:156-169`).
2. Pre-process: `merge_tool_messages` + `sort_tool_results_by_call_order` (`:178-179`).
3. Start from `bos_token`, then **render each message piece-by-piece** with `render_message(i, …)`,
   recording every message's char span `(start, end, role)` (`:186-194`).
   - assistant template = `{reasoning}{content}{tool_calls}` + EOS (`encoding_dsv4.py:45`).
   - tool docs injected via `TOOLS_TEMPLATE`; `tool_result` is rendered under a `<｜User｜>` turn.
4. **Thinking mode** (this run: `MILES_DSV4_THINKING_MODE=chat`, `MILES_DSV4_DROP_THINKING=0`):
   reasoning is **kept and trained**; samples carrying `tools` force `drop_thinking=False`.

## 3. Tokenize + loss mask (the core rule)
```python
enc_out = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)  # mask_utils.py:197
```
- `add_special_tokens=False` — specials are already in the rendered text; the tokenizer must not add more.
- `return_offsets_mapping=True` — gives each token its char span `(s, e)`.
- **Set mask=1** (`:202-210`): a token is trained iff its `(s, e)` lies fully inside some
  `role=="assistant"` span `[start, end)`. Tokens with `(0,0)` (offset-less specials) are skipped → stay 0.
- **Per-turn override** (`:214-223`): an assistant turn with `step_loss_mask=0` is zeroed back out.

Result: `mask=1` = assistant reasoning + content + tool_calls + EOS; `mask=0` = system / user /
`tool_result` + all role/transition tokens (`<｜Assistant｜>`, `<think>`, `</think>`, …).

The downstream loss (`loss.py:88-93`) applies the standard shift `logits[t-1] → token[t]`, so CE lands
on each masked assistant token given its prefix. Per-token normalization divides by the real
`Σ loss_mask` (`loss.py:869`). See `docs/SFT_LOSS_VERIFICATION.md`.

## 4. Data flow (jsonl → model)
```
albaliang_077_le128k.jsonl  (per line: {messages, tools, token_length})
  → Dataset stores raw messages verbatim (apply_chat_template=False)
  → each rollout: random.seed(42 + epoch) shuffle, take 128 samples
  → per sample: get_loss_mask() → (token_ids, loss_mask)
  → dynamic packing to token_cap 196,608 / microbatch (packing_efficiency ~0.94–0.96)
  → --qkv-format thd (variable-length packed, not padded rectangles) → Megatron, CP6 zigzag split
```

## 5. Verified real examples (assistant-only masking)
`M` = trained (in loss), `.` = masked. Decoded from actual `(token_ids, loss_mask)`.

**(a) Simple chat** — openhermes, ratio 0.60:
```
[. 12t] <｜begin▁of▁sentence｜><｜User｜>Do you know any jokes about librarians?<｜Assistant｜></think>
[M 18t] Why do librarians like the wind? It says, "Shhh!" all day!<｜end▁of▁sentence｜>
```

**(b) Long prompt, short answer** — classification, ratio 0.13 (explains the many low-ratio samples):
```
[. 87t] <｜begin▁of▁sentence｜>Analyze if this message indicates a new conversation topic...
[M 13t] {"isNewTopic": false, "title": null}<｜end▁of▁sentence｜>
```

**(c) Multi-turn tool use** — the real agent-SFT shape, ratio 0.37:
```
[. 425t] <｜begin▁of▁sentence｜>You are a file search specialist...        ← system prompt (masked)
[M 139t] \n\n我先搜索 controller 目录...<｜DSML｜tool_calls>...               ← assistant turn 1 (reasoning+tool_call, trained)
[.  15t] <｜User｜><tool_result>Found 0 files</tool_result><｜Assistant｜></think>  ← tool result (masked, = user input)
[M 119t] \n\n<｜DSML｜tool_calls><｜DSML｜invoke name="search_file">...           ← assistant turn 2 (trained)
```
Takeaway: only assistant-generated content trains; `tool_result` is context (rendered as a `<｜User｜>`
turn) and is masked, as are all role/transition tokens.

## 6. One measurement gotcha
Each jsonl line carries a `token_length` field computed by a **different counter** that **over-counts
~20K tokens** vs this production tokenizer. (That metadata is why 332 samples were filtered out of
le128k as ">128K", though they actually tokenize to 55K–131K here — all within window.) **For "how
long is a sample," trust the `tokenizer(text, …)` path above, not the metadata field.** See
`docs/POSTRUN_EVAL_RUNBOOK.md` / memory `sft-holdout-eval-prep`.

---
### Provenance
Read from code + verified on real samples 2026-06-07 (worker pod, CPU-only, no GPU, no disturbance to
the live run). Key code: `miles/utils/mask_utils.py:135-225`, `encoding_dsv4.py`,
`miles/backends/training_utils/loss.py:88-93,869`. Config: run.sh `--loss-mask-type deepseek_v4`,
`MILES_DSV4_THINKING_MODE=chat`. Related: `docs/SFT_LOSS_VERIFICATION.md`.

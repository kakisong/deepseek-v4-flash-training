# SFT Loss: How It's Computed, How It Was Verified, and One Misleading Metric

**TL;DR (verified 2026-06-07 by exact-batch reproduction + code trace):**
The V4-Flash SFT `train/loss` is computed **correctly** — per-token cross-entropy over
**assistant response tokens only**, with the correct next-token shift, per-token normalization,
and CP-aware aggregation. The one apparent anomaly — an offline loss-token ratio of **0.076** vs a
wandb `train/batch/loss_token_ratio` of **0.195** — is **not a loss bug**: that wandb metric is a
**CP-local × cp_size approximation that over-counts ~2× for end-loaded SFT**. The actual loss does
not use it. **Trust `train/loss`, not `loss_token_ratio`.**

Verification tools: `tools/verify_sft_loss_mask.py`, `tools/verify_sft_pipeline.py`.

---

## 1. How the loss is computed (the real path)

Run config: `--loss-type sft_loss --calculate-per-token-loss --loss-mask-type deepseek_v4`,
TP8/PP7/**CP6**/EP8, `--qkv-format thd`, vocab 129,280.

1. **Mask = assistant tokens only.** `sft_rollout.generate_rollout` →
   `MultiTurnLossMaskGenerator.get_loss_mask` → `gen_multi_turn_loss_mask_deepseek_v4`
   (`miles/utils/mask_utils.py:135`). It renders the conversation with the V4 chat template
   (`encoding/encoding_dsv4.py`), tokenizes with offset-mapping, and sets `loss_mask[k]=1` **only**
   for tokens whose char-offset falls inside an `assistant` message span (line 203:
   `if role != "assistant": continue`). System/user/tool messages and special/transition tokens
   (`<｜Assistant｜>`, BOS, …) stay `0`. Per-turn `step_loss_mask=0` zeroes that turn.
2. **Next-token shift** (`loss.py:88-93`): `logits_chunk = logits[start-1:end-1]`,
   `tokens_chunk = tokens[-response_length:]` → `logits[t-1]` predicts `token[t]` (standard shift-by-one).
3. **Per-token normalization** (`calculate_per_token_loss=True`): the reducer is `sum_of_token`
   (`cp_utils.py:122`), so `loss = -Σ logP` over response tokens; Megatron divides the accumulated loss
   by `num_tokens = Σ loss_mask.sum()` (`loss.py:869`, the **real** masks) across all microbatches and
   DP/CP ranks → a true per-token mean cross-entropy (nats). Random baseline ≈ ln(129280) ≈ 11.8.
4. **CP-aware** (`get_sum_of_sample_mean` CP branch, `cp_utils.py:86-120`): the zigzag-CP loss masks are
   chunked per rank and summed across CP; the loss normalizer aggregates CP-local counts correctly.

## 2. How it was verified (empirical, not by trust)

- **Mask on real samples** (`verify_sft_loss_mask.py`, 60 samples): every sample has the system prompt
  masked (`lead_masked=True` 60/60); decoded `mask=1` spans are assistant reasoning/content/tool_calls;
  `mask=0` spans are the system prompt + user tool_results. Per-sample loss ratio ranges 0.001–0.82
  (varies with how much of the trajectory is assistant text vs context/tool output).
- **Exact-batch reproduction** (`verify_sft_pipeline.py`): replicate the training data pipeline
  (`apply_chat_template=False` → `_build_messages` returns raw messages unchanged; shuffle with
  `random.seed(rollout_seed=42 + epoch 0)`), take the **same 128 samples as live rollout 0**, run the
  real mask. Result: **total tokens 6,638,138 = live step-0 to the byte (diff +0)**; true loss tokens
  **502,530** → true ratio **0.0757**.

## 3. The one misleading metric: `train/batch/loss_token_ratio`

The wandb metric reports **0.195**, not 0.076. Root cause (`megatron_utils/model.py:65-71`):

```python
# tokens/full_loss_masks are CP-local after get_batch(); multiply by CP for a
# rollout-level approximation ...
local_loss_tokens = int(batch["full_loss_masks"].sum().item())
loss_tokens = local_loss_tokens * cp_size          # <-- the approximation
...
loss_token_ratio = loss_tokens / total_tokens      # total_tokens is exact (sum of total_lengths)
```

- `full_loss_masks` is the **CP-local chunk** of the mask (`training_utils/data.py:338,357,363`), summed on
  one rank, then **× cp_size (=6)** to *estimate* the global count.
- That estimate assumes loss tokens are **uniformly** distributed across the 6 zigzag-CP chunks. For SFT
  they are **end-loaded** (long prompt + tool outputs, then a short assistant response), so the measured
  rank's chunk holds ~2.14× its uniform share → `502,530` becomes `1,076,592`.
- Signature: the metric jitters step-to-step (0.16–0.20) and sits a steady ~2× above the true 0.076–0.09.

**Why it doesn't matter:** `full_loss_masks` and its `× cp_size` count appear **only** in the diagnostic
logger `_collect_train_batch_debug`. The actual loss + gradient use `batch["loss_masks"]` (the real masks)
with correct cross-rank aggregation. So the **loss value is right**; only this one logged ratio is inflated.

**Guidance:** judge training health from **`train/loss`** (smooth decrease, no NaN/spike) and the epoch
boundaries — not `loss_token_ratio`. Optional fix: replace `local × cp_size` with a real cross-rank
`all_reduce(full_loss_masks.sum())` so the diagnostic matches the true count.

## Appendix — reproducibility
- `tools/verify_sft_loss_mask.py [n] [data]` — mask on real samples (trained vs masked spans, ratios).
- `tools/verify_sft_pipeline.py` — exact rollout-0 reproduction vs the live step-0 token counts.
- Both: run in a GPU pod (CPU-only, won't disturb training), `PYTHONPATH=<fsx miles>`, HF ckpt
  `models/DeepSeek-V4-Flash-bf16-unpacked`, data `data/albaliang_077_le128k.jsonl`.
- Live run when verified: `stageKaynzhang077-134K-3ep-H200-20260606-172101` (submission
  `raysubmit_L2dbLAFDyHjRbDPM`), wandb `v4-flash-post`.

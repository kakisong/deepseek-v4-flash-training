# Post-run Eval Runbook — V4-Flash SFT (Kaynzhang 077, 3-epoch)

**Purpose.** The live SFT run has **no held-out validation loss** and **no downstream eval** (see
why below). This runbook is the staged, one-command-each procedure to validate the trained model
**after the run finishes and frees its 336 GPUs**. Everything here was prepared CPU-only during the
run (option "D"); the GPU steps wait for run completion.

## Why there is no in-run eval (verified)
- `miles/rollout/sft_rollout.py:13` hard-codes `assert not evaluation` → miles' eval path never
  computes an SFT cross-entropy; for SFT it would only do RL-reward generation. The run also sets no
  `--eval-interval` / `--eval-prompt-data`.
- All **49,667** `albaliang_077_le128k.jsonl` samples were trained (3 epochs); no held-out split was
  reserved. (`le128k ⊂ le134k` verified; the 332 extra lines in le134k are the only unseen same-domain data.)
- The HF checkpoint dir has **no transformers modeling code** (`config.json` `auto_map=None`,
  `architectures=["DeepseekV4ForCausalLM"]`, `model_type=deepseek_v4` — unknown to upstream
  transformers). So a plain `AutoModelForCausalLM` forward is **not** available; scoring goes through
  Megatron or an SGLang server.

## Held-out sets (already carved, CPU-only)
`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/data/holdout_eval/`
- **`holdout_albaliang_077_332.jsonl`** — 332 **same-domain, truly-unseen** samples (the le134k∖le128k
  difference; agent/tool-use, multi-turn). **Measured** deepseek_v4 token lengths (all 332): min 54.6K,
  p50 108.4K, p90 117.7K, **max 130.6K → 0/332 exceed the 128K window**, so **no truncation triggers**
  and they sit squarely in the trained-length distribution. (The metadata `token_length` of 131–134K
  that got them filtered over-counts by ~20K vs the production tokenizer — that's the only reason they
  were dropped from le128k.) 1,619,525 assistant loss tokens total → a stable CE estimate. Best
  "did epoch 2/3 overfit the task?" probe.
- **`holdout_openhermes_512.jsonl`** — 512 **cross-domain, truly-unseen** samples (OpenHermes general
  chat, 0% overlap with training, ≤4K tokens). "Did SFT cause catastrophic forgetting of general
  instruction-following?" probe.

## Checkpoints (verified on disk)
`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-3ep-H200-20260606-172101/checkpoints/`
- Format: `torch_dist` (`.metadata` + `__<rank>_<n>.distcp`), ~3.7 T per iter (incl. optimizer state).
- Retention keeps `{latest, every-multiple-of-500}`. **Final** will be `iter_0001164` (last step) plus
  `iter_0001000`. **Grab `iter_0001164` (or `iter_0001000`) before launching anything that prunes it.**

---

## Step 0 — wait for run end, pick the checkpoint
```bash
CKROOT=/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-3ep-H200-20260606-172101/checkpoints
ls -1dt $CKROOT/iter_*          # confirm iter_0001164 exists; cat $CKROOT/latest_checkpointed_iteration.txt
ITER=$CKROOT/iter_0001164
```

## Step 1 — convert torch_dist → HF (CPU-only, NO GPU)
`convert_torch_dist_to_hf.py` uses `no_dist=True` (single-process CPU load of `common.pt` + distcp
shards → safetensors). Needs a big-RAM node (model params ≈ 568 GB held in CPU state_dict during write).
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

## Step 2 — serve with SGLang (8×H200, one BF16 replica)
`tools/launch_sglang.sh` defaults to `--context-length 32768`, **too short for the albaliang held-out
(~110–118K)**. Use the explicit launch with a long context:
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

## Step 3 (B) — held-out cross-entropy (cheap sanity, ~20 min)
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
**Interpretation**
- albaliang held-out CE ≈ live `train/loss` (~2.0–2.2) → epochs 2/3 did **not** overfit; the ~2.2
  plateau is a genuine data floor, not memorization.
- albaliang held-out CE ≫ train/loss → overfitting; an earlier checkpoint (e.g. `iter_0000500`,
  end of epoch 1) may generalize better.
- openhermes CE stable vs the base/earlier iters → no catastrophic forgetting.
- **Best signal**: run Step 1–3 on **two** checkpoints (`iter_0000500` end-of-ep1 and `iter_0001164`
  final) and compare — if held-out CE is flat across them while train loss fell, the extra epochs
  bought generalization nothing (consistent with the observed train-loss plateau).

> `eval_holdout_ce.py` reuses the **proven** `MultiTurnLossMaskGenerator` (bit-identical masks, per
> `tools/verify_sft_pipeline.py`). The SGLang `input_token_logprobs` plumbing follows
> `miles/rollout/sglang_rollout.py` conventions but is **unvalidated end-to-end** (no spare GPU during
> the run): on the first response, assert `len(input_token_logprobs)==len(input_ids)` before trusting
> aggregates.

## Step 3 (A) — downstream / generative eval (the meaningful one)
For an agent/tool-use SFT, task accuracy matters more than CE. Harnesses live in
`miles/examples/eval/` and talk to the SGLang OpenAI endpoint:
- **Terminal Bench** (agent/tool-use, code tasks): `examples/eval/terminal_bench/tb_server.py` +
  `tb_client.py`; config template `examples/eval/scripts/eval_tb_example.yaml`.
- **NeMo Skills** (AIME25 / Arena-Hard / HLE): `examples/eval/nemo_skills/skills_server.py`.
- Both set `api_base=http://127.0.0.1:30000/v1`, `model_name=<your model>`; pass `tool_key: tools`
  in the eval dataset config for tool-use evals (`miles/utils/eval_config.py`).
See `examples/eval/scripts/run-eval-tb-qwen.sh` for a full delegated example. Run standalone (not
in a training loop) to avoid the eval timeout blocking anything.

---
### Provenance
Prepared 2026-06-07 during the live run (CPU-only, zero GPU, no disturbance to training).
Run: `stageKaynzhang077-134K-3ep-H200-20260606-172101`, submission `raysubmit_L2dbLAFDyHjRbDPM`.
Artifacts: `data/holdout_eval/*.jsonl`, `tools/eval_holdout_ce.py`, this runbook.
Related: `docs/SFT_LOSS_VERIFICATION.md` (loss correctness), `docs/H200_BOTTLENECK_ANALYSIS.md`.

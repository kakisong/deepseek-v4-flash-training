"""Held-out cross-entropy (SFT validation loss) for V4-Flash, scored via an SGLang server.

WHY this exists
---------------
The running SFT job has NO held-out validation loss:
  - miles' built-in eval path asserts `not evaluation` for SFT (sft_rollout.py), so it
    only does RL-reward generation, never a held-out CE.
  - all 49,667 le128k samples were trained (3 epochs), so there is no in-run held-out split.
  - the HF checkpoint has NO transformers modeling code (config auto_map=None,
    arch=DeepseekV4ForCausalLM unknown to upstream transformers), so a plain
    `AutoModelForCausalLM` forward is NOT available.

So we score held-out CE by **teacher-forcing through an SGLang server** (the same serving
path used for downstream eval), reusing the PROVEN loss-mask generator so the masked tokens
are bit-identical to training.

WHAT it computes
----------------
For each held-out sample:
  1. tokenize + assistant-only loss-mask via MultiTurnLossMaskGenerator (deepseek_v4) — the
     exact generator verified bit-identical to training (tools/verify_sft_pipeline.py).
  2. left-truncate to --max-len tokens (keep the tail = the assistant response; the held-out
     albaliang samples are 131-134K, only 0-3K over the 128K window, so this drops a few K of
     the OLDEST context and keeps the full response).
  3. POST input_ids to SGLang /generate with return_logprob + logprob_start_len=0,
     read meta_info["input_token_logprobs"][i] = logP(token_i | token_<i).
  4. CE = -mean over positions where loss_mask[i]==1  (same next-token alignment as training:
     input_token_logprobs[i] scores token_i given its prefix == training's logits[t-1]->token[t]).

Aggregate: token-weighted mean CE (nats) per file, plus per-sample distribution.

USAGE (post-run, after GPUs free; ~8xH200 for one BF16 replica)
---------------------------------------------------------------
  # 1. convert a checkpoint to HF (CPU-only, no GPU; see postrun_eval_runbook.md)
  # 2. serve it (NOTE: bump context-length to >=131072 for the albaliang held-out set):
  #      python3 -m sglang.launch_server --model-path <hf_dir> --tp 8 --trust-remote-code \
  #        --attention-backend triton --context-length 131072 --port 30000 --host 0.0.0.0
  # 3. score:
  python3 tools/eval_holdout_ce.py \
    --data /mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/data/holdout_eval/holdout_albaliang_077_332.jsonl \
    --hf-tokenizer /mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-bf16-unpacked \
    --sglang-url http://127.0.0.1:30000 --max-len 131072 --out /tmp/ce_albaliang.json

  python3 tools/eval_holdout_ce.py \
    --data .../holdout_eval/holdout_openhermes_512.jsonl \
    --hf-tokenizer .../DeepSeek-V4-Flash-bf16-unpacked \
    --sglang-url http://127.0.0.1:30000 --max-len 8192 --out /tmp/ce_openhermes.json

INTERPRETATION
--------------
  - albaliang held-out CE  ~ the live train/loss (~2.0-2.2)  -> epochs 2/3 did NOT overfit the
    task distribution (the plateau is a genuine data floor, not memorization).
  - albaliang held-out CE  >> train/loss                     -> overfitting; earlier ckpt better.
  - openhermes (cross-domain) CE stable across iters         -> no catastrophic forgetting of
    general instruction-following.

NOTE: the SGLang input_token_logprobs plumbing is validated against miles/rollout/sglang_rollout.py
payload conventions but has NOT been run end-to-end (no spare GPU during the live run). Validate the
first response's len(input_token_logprobs)==len(input_ids) before trusting aggregates.
"""
import argparse
import json
import math
import sys
import time

import requests
from transformers import AutoTokenizer

from miles.utils.mask_utils import MultiTurnLossMaskGenerator


def load_samples(path):
    out = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get("messages")
            tools = d.get("tools") or d.get("metadata", {}).get("tools")
            if isinstance(tools, str):
                tools = json.loads(tools)
            out.append((msgs, tools))
    return out


def score_one(url, input_ids):
    """Return list of input_token_logprobs aligned to input_ids (logP(tok_i | tok_<i))."""
    payload = {
        "input_ids": input_ids,
        "return_logprob": True,
        "logprob_start_len": 0,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
    }
    r = requests.post(f"{url}/generate", json=payload, timeout=3600)
    r.raise_for_status()
    meta = r.json()["meta_info"]
    # input_token_logprobs: list of [logprob, token_id, text|None]; first token has null logprob
    itl = meta["input_token_logprobs"]
    return [(x[0] if x[0] is not None else None) for x in itl]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hf-tokenizer", required=True)
    ap.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    ap.add_argument("--max-len", type=int, default=131072, help="left-truncate to keep last N tokens")
    ap.add_argument("--max-samples", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.hf_tokenizer, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type="deepseek_v4")
    samples = load_samples(args.data)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    tot_nll = 0.0
    tot_loss_tok = 0
    per_sample = []
    t0 = time.time()
    for k, (msgs, tools) in enumerate(samples):
        token_ids, loss_mask = gen.get_loss_mask(msgs, tools=tools)
        # left-truncate to keep the tail (the assistant response)
        if len(token_ids) > args.max_len:
            cut = len(token_ids) - args.max_len
            token_ids = token_ids[cut:]
            loss_mask = loss_mask[cut:]
        logps = score_one(args.sglang_url, token_ids)
        if len(logps) != len(token_ids):
            print(f"  WARN s{k}: logprob len {len(logps)} != tokens {len(token_ids)} — skipping", file=sys.stderr)
            continue
        nll = 0.0
        n = 0
        for i, m in enumerate(loss_mask):
            if m and logps[i] is not None:
                nll += -logps[i]
                n += 1
        if n:
            tot_nll += nll
            tot_loss_tok += n
            per_sample.append(nll / n)
        if k < 3 or k % 50 == 0:
            ce = (tot_nll / tot_loss_tok) if tot_loss_tok else float("nan")
            print(f"  s{k}: loss_tok={n} running_CE={ce:.4f} ({k+1}/{len(samples)}, {time.time()-t0:.0f}s)")

    ce = (tot_nll / tot_loss_tok) if tot_loss_tok else float("nan")
    ppl = math.exp(ce) if tot_loss_tok else float("nan")
    per_sample.sort()
    res = {
        "data": args.data,
        "n_samples": len(per_sample),
        "max_len": args.max_len,
        "held_out_CE_nats": ce,
        "held_out_ppl": ppl,
        "total_loss_tokens": tot_loss_tok,
        "per_sample_CE": {
            "min": per_sample[0] if per_sample else None,
            "p50": per_sample[len(per_sample) // 2] if per_sample else None,
            "max": per_sample[-1] if per_sample else None,
        },
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n=== HELD-OUT CE === {args.data}")
    print(f"  token-weighted CE = {ce:.4f} nats  (ppl {ppl:.2f})  over {tot_loss_tok} loss tokens, {len(per_sample)} samples")
    print(f"  -> compare to live train/loss (~2.0-2.2). saved {args.out}")


if __name__ == "__main__":
    main()

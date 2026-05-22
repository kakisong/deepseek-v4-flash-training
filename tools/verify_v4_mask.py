#!/usr/bin/env python3
"""Verify V4 loss mask on a converted V4-format JSONL.

For the first --num records, load tokenizer + MultiTurnLossMaskGenerator(type='deepseek_v4'),
run get_loss_mask, and print:

  - token count vs source-claimed token_length
  - per-assistant-turn: step_loss_mask, token span size, decoded loss=1 span preview
  - sanity check: are loss=1 token regions actually inside `assistant` rendered text?
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hf-checkpoint", default="/data_fast_v3/kaynzhang/v4-sft/models/DeepSeek-V4-Flash-bf16-unpacked")
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    # Ensure miles is importable
    miles_root = "/data_fast_v3/kaynzhang/v4-sft/miles"
    sys.path.insert(0, miles_root)

    from miles.utils.mask_utils import MultiTurnLossMaskGenerator
    from miles.utils.processing_utils import load_tokenizer

    print(f"loading tokenizer from {args.hf_checkpoint} ...", flush=True)
    tok = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    mg = MultiTurnLossMaskGenerator(tok, tokenizer_type="deepseek_v4")
    print("ok\n", flush=True)

    n = 0
    with open(args.data) as f:
        for line in f:
            if not line.strip():
                continue
            if n >= args.num:
                break
            n += 1
            rec = json.loads(line)
            messages = rec["messages"]
            tools = rec.get("tools") or None
            tlen_src = rec.get("token_length")

            token_ids, loss_mask = mg.get_loss_mask(messages, tools=tools)
            n_tok = len(token_ids)
            n_loss = sum(loss_mask)
            print(f"=== record {n}  source token_length={tlen_src}  rendered={n_tok}  loss-on tokens={n_loss}  loss ratio={n_loss/max(1,n_tok):.2%}")

            n_asst = sum(1 for m in messages if m.get("role") == "assistant")
            n_asst_loss_on = sum(1 for m in messages if m.get("role") == "assistant" and m.get("step_loss_mask", 1) == 1)
            print(f"   assistant turns: {n_asst} (step_loss_mask=1: {n_asst_loss_on})")

            # Decode contiguous loss=1 spans; show first 3 previews.
            spans = []
            cur = []
            for i, b in enumerate(loss_mask):
                if b == 1:
                    cur.append(token_ids[i])
                elif cur:
                    spans.append(cur)
                    cur = []
            if cur:
                spans.append(cur)
            print(f"   contiguous loss=1 spans: {len(spans)}; previewing first 3:")
            for i, span in enumerate(spans[:3]):
                txt = tok.decode(span)
                preview = txt[:200].replace("\n", "\\n")
                print(f"     span[{i}] len={len(span)}  {preview}{'...' if len(txt)>200 else ''}")
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

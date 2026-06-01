#!/usr/bin/env python3
"""QA-fit verification: sample N rows from train set, ask model to predict each
assistant turn given the prefix, compare against ground truth.

Purpose: validate training framework integrity (gradient flow, loss mask, ckpt
save/load). On train-set samples, a well-fit model should reproduce assistant
turns nearly verbatim. Large divergence → framework bug.

Usage (from inside miles-v4-sft container, after launching sglang):
    python3 qa_fit_check.py \\
        --data /data_train/kaynzhang/v4-sft/data/albaliang_057_le64k.jsonl \\
        --endpoint http://localhost:30000 \\
        --num-samples 10 \\
        --turn-index 1   # 1st assistant turn (0-indexed in assistant turns)

Outputs per-sample: prefix length, ground-truth A length, generated A length,
exact-match, char-level Levenshtein ratio, BLEU-like prefix match length.
"""
import argparse
import json
import os
import random
import sys
from difflib import SequenceMatcher

import requests


def load_samples(path, n, seed=0, max_token_length=32000):
    rng = random.Random(seed)
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("token_length", 0) > max_token_length:
                continue
            rows.append(d)
            if len(rows) >= 5000:
                break
    rng.shuffle(rows)
    return rows[:n]


def split_at_assistant_turn(messages, turn_index):
    """Return (prefix_messages, ground_truth_assistant_content) for the
    `turn_index`-th assistant turn (0-indexed). Returns None if not enough
    assistant turns."""
    seen = 0
    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            if seen == turn_index:
                return messages[:i], m.get("content") or ""
            seen += 1
    return None


def call_sglang_chat(endpoint, messages, tools=None, max_tokens=2048):
    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    if tools:
        payload["tools"] = tools
    r = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"].get("content") or ""


def lev_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--endpoint", default="http://localhost:30000")
    ap.add_argument("--num-samples", type=int, default=10)
    ap.add_argument("--turn-index", type=int, default=0,
                    help="Which assistant turn to predict (0=first)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-token-length", type=int, default=16000,
                    help="Skip rows longer than this many tokens (avoids ctx overflow)")
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    args = ap.parse_args()

    samples = load_samples(args.data, args.num_samples, args.seed, args.max_token_length)
    print(f"Loaded {len(samples)} samples (seed={args.seed}, max_token_length={args.max_token_length})")
    print(f"Endpoint: {args.endpoint}")
    print(f"Predicting assistant turn #{args.turn_index}\n")

    em_count = 0
    ratios = []
    for idx, sample in enumerate(samples):
        msgs = sample["messages"]
        tools_str = sample.get("tools")
        tools = None
        if tools_str:
            try:
                tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
            except Exception:
                tools = None

        split = split_at_assistant_turn(msgs, args.turn_index)
        if split is None:
            print(f"[{idx}] skip (only {sum(1 for m in msgs if m['role']=='assistant')} assistant turns)")
            continue
        prefix, gt = split

        # Strip non-content fields from prefix that vendor APIs may not accept
        cleaned = []
        for m in prefix:
            cm = {"role": m["role"], "content": m.get("content") or ""}
            if m["role"] == "assistant" and m.get("tool_calls"):
                cm["tool_calls"] = m["tool_calls"]
            if m["role"] == "tool":
                cm["tool_call_id"] = m.get("tool_call_id", "")
            cleaned.append(cm)

        try:
            gen = call_sglang_chat(args.endpoint, cleaned, tools=tools,
                                   max_tokens=args.max_output_tokens)
        except Exception as e:
            print(f"[{idx}] request failed: {e}")
            continue

        em = (gen.strip() == gt.strip())
        ratio = lev_ratio(gen, gt)
        cp = common_prefix_len(gen, gt)
        em_count += int(em)
        ratios.append(ratio)
        print(f"[{idx}] prefix_msgs={len(prefix)} gt_chars={len(gt)} gen_chars={len(gen)} "
              f"EM={em} sim={ratio:.3f} common_prefix={cp}")
        if not em:
            # show first divergence
            cut = min(cp + 80, len(gen), len(gt))
            print(f"     gt : ...{gt[max(0,cp-20):cut]!r}")
            print(f"     gen: ...{gen[max(0,cp-20):cut]!r}")

    n = len(ratios)
    if n:
        print(f"\n=== summary ({n} samples) ===")
        print(f"exact-match: {em_count}/{n} ({em_count/n*100:.1f}%)")
        print(f"mean similarity: {sum(ratios)/n:.3f}")
        print(f"median similarity: {sorted(ratios)[n//2]:.3f}")


if __name__ == "__main__":
    main()

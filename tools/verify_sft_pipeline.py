"""Faithfully reproduce the TRAINING data pipeline for rollout 0 and compare token counts
to the LIVE step-0 metric. This isolates whether the offline-vs-live loss_token_ratio gap is
a mask bug or just a sampling-order artifact.

Training path (apply_chat_template=False): Dataset stores sample.prompt = raw messages (no
transform); RolloutDataSource shuffles with random.seed(rollout_seed + epoch_id=0) then pulls
samples[0:rollout_batch_size]; sft_rollout.generate_rollout runs get_loss_mask on each.

LIVE step-0 metric (rollout 0): tokens=6,638,138  loss_tokens=1,076,592  ratio=0.16218
So if this script (same shuffle, same 128 samples) reproduces those counts -> mask is correct and
the earlier 0.07 was just my non-shuffled sampling. If loss_tokens is ~half -> real mask difference.

Run on a pod: python3 tools/verify_sft_pipeline.py
"""
import json
import random

from transformers import AutoTokenizer

from miles.utils.mask_utils import MultiTurnLossMaskGenerator

HF = "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-bf16-unpacked"
DATA = "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/data/albaliang_077_le128k.jsonl"
ROLLOUT_SEED = 42
ROLLOUT_BATCH = 128
# live step-0 ground truth
LIVE_TOKENS, LIVE_LOSS, LIVE_RATIO = 6_638_138, 1_076_592, 0.16218


def main():
    tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type="deepseek_v4")

    # 1. load all samples exactly as Dataset does (apply_chat_template=False -> raw messages)
    origin = []
    with open(DATA) as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("messages")  # _build_messages returns this verbatim
            tools = data.get("tools")
            if isinstance(tools, str):
                tools = json.loads(tools)
            origin.append((prompt, tools))
    n = len(origin)

    # 2. replicate RolloutDataSource shuffle: random.seed(seed + epoch_id=0); shuffle(permutation)
    perm = list(range(n))
    random.seed(ROLLOUT_SEED + 0)
    random.shuffle(perm)
    batch_idx = perm[:ROLLOUT_BATCH]  # rollout 0 = first 128 shuffled

    # 3. run the real mask on exactly those 128 samples
    tot_tok = tot_loss = 0
    for k, i in enumerate(batch_idx):
        messages, tools = origin[i]
        token_ids, loss_mask = gen.get_loss_mask(messages, tools=tools)
        tot_tok += len(token_ids)
        tot_loss += sum(loss_mask)
        if k < 3:
            print(f"  s{k} (orig#{i}): tok={len(token_ids)} loss={sum(loss_mask)}")

    ratio = tot_loss / tot_tok
    print(f"\n=== REPRODUCED rollout 0 (shuffle seed={ROLLOUT_SEED}, first {ROLLOUT_BATCH}) ===")
    print(f"  tokens     = {tot_tok:>10,}   (live {LIVE_TOKENS:>10,})  diff {tot_tok-LIVE_TOKENS:+,}")
    print(f"  loss_tokens= {tot_loss:>10,}   (live {LIVE_LOSS:>10,})  diff {tot_loss-LIVE_LOSS:+,}")
    print(f"  ratio      = {ratio:.5f}      (live {LIVE_RATIO:.5f})")
    match = abs(tot_tok - LIVE_TOKENS) / LIVE_TOKENS < 0.02 and abs(tot_loss - LIVE_LOSS) / LIVE_LOSS < 0.02
    print(f"  --> {'MATCH: mask computation is correct, earlier 0.07 was non-shuffled sampling' if match else 'MISMATCH: real difference -> investigate render/mask'}")


if __name__ == "__main__":
    main()

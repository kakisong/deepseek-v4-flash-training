"""忠实复现 rollout 0 的训练数据流水线,并把 token 计数与线上 step-0 指标对比。
以此区分离线与线上 loss_token_ratio 的差距究竟是 mask bug,
还是仅仅是采样顺序造成的假象。

训练路径(apply_chat_template=False):Dataset 存 sample.prompt = 原始 messages(不做
变换);RolloutDataSource 用 random.seed(rollout_seed + epoch_id=0) 洗牌后取
samples[0:rollout_batch_size];sft_rollout.generate_rollout 对每条样本跑 get_loss_mask。

线上 step-0 指标(rollout 0):tokens=6,638,138  loss_tokens=1,076,592  ratio=0.16218
因此若本脚本(同样的洗牌、同样的 128 条样本)能复现这些计数 -> mask 正确,
之前的 0.07 只是我未洗牌采样所致。若 loss_tokens 约为一半 -> mask 存在真实差异。

在 pod 上运行: python3 tools/verify_sft_pipeline.py
"""
import json
import random

from transformers import AutoTokenizer

from miles.utils.mask_utils import MultiTurnLossMaskGenerator

HF = "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-bf16-unpacked"
DATA = "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/data/albaliang_077_le128k.jsonl"
ROLLOUT_SEED = 42
ROLLOUT_BATCH = 128
# 线上 step-0 真值
LIVE_TOKENS, LIVE_LOSS, LIVE_RATIO = 6_638_138, 1_076_592, 0.16218


def main():
    tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type="deepseek_v4")

    # 1. 完全按 Dataset 的方式加载所有样本(apply_chat_template=False -> 原始 messages)
    origin = []
    with open(DATA) as f:
        for line in f:
            data = json.loads(line)
            prompt = data.get("messages")  # _build_messages 原样返回这个值
            tools = data.get("tools")
            if isinstance(tools, str):
                tools = json.loads(tools)
            origin.append((prompt, tools))
    n = len(origin)

    # 2. 复刻 RolloutDataSource 的洗牌:random.seed(seed + epoch_id=0); shuffle(permutation)
    perm = list(range(n))
    random.seed(ROLLOUT_SEED + 0)
    random.shuffle(perm)
    batch_idx = perm[:ROLLOUT_BATCH]  # rollout 0 = 洗牌后的前 128 条

    # 3. 对正好这 128 条样本跑真实的 mask
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

"""V4-Flash 的留出集交叉熵(SFT 验证 loss),通过 SGLang 服务打分。

为什么需要这个脚本
---------------
正在运行的 SFT 任务没有任何留出集验证 loss:
  - miles 内置的 eval 路径对 SFT 断言 `not evaluation`(sft_rollout.py),因此
    它只做 RL-reward 生成,从不计算留出集 CE。
  - 全部 49,667 条 le128k 样本都参与了训练(3 个 epoch),因此训练过程中没有留出切分。
  - HF checkpoint 没有 transformers 建模代码(config auto_map=None,
    arch=DeepseekV4ForCausalLM 对上游 transformers 是未知架构),因此无法直接用
    `AutoModelForCausalLM` 做前向。

所以我们改为**通过 SGLang 服务做 teacher-forcing** 来计算留出集 CE(与下游评测使用
同一条 serving 路径),并复用已验证的 loss-mask 生成器,使被 mask 的 token
与训练时逐 bit 一致。

它计算什么
----------------
对每条留出样本:
  1. 用 MultiTurnLossMaskGenerator(deepseek_v4)做 tokenize + 仅 assistant 的 loss-mask —
     该生成器已被验证与训练逐 bit 一致(tools/verify_sft_pipeline.py)。
  2. 左截断到 --max-len 个 token(保留尾部 = assistant 回复;留出的
     albaliang 样本长 131-134K,只超出 128K 窗口 0-3K,所以只会丢掉几 K 最旧的
     上下文,完整保留回复)。
  3. 把 input_ids POST 到 SGLang /generate,带上 return_logprob + logprob_start_len=0,
     读取 meta_info["input_token_logprobs"][i] = logP(token_i | token_<i)。
  4. CE = 对 loss_mask[i]==1 的位置取 -mean(与训练相同的 next-token 对齐方式:
     input_token_logprobs[i] 在给定前缀下为 token_i 打分 == 训练里的 logits[t-1]->token[t])。

汇总:每个文件按 token 加权的平均 CE(nats),外加逐样本分布。

用法(训练跑完、GPU 空闲后;一个 BF16 副本约需 8xH200)
---------------------------------------------------------------
  # 1. 把 checkpoint 转成 HF(纯 CPU,无需 GPU;见 postrun_eval_runbook.md)
  # 2. 启动服务(注意:针对 albaliang 留出集需把 context-length 提到 >=131072):
  #      python3 -m sglang.launch_server --model-path <hf_dir> --tp 8 --trust-remote-code \
  #        --attention-backend triton --context-length 131072 --port 30000 --host 0.0.0.0
  # 3. 打分:
  python3 tools/eval_holdout_ce.py \
    --data /mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/data/holdout_eval/holdout_albaliang_077_332.jsonl \
    --hf-tokenizer /mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-bf16-unpacked \
    --sglang-url http://127.0.0.1:30000 --max-len 131072 --out /tmp/ce_albaliang.json

  python3 tools/eval_holdout_ce.py \
    --data .../holdout_eval/holdout_openhermes_512.jsonl \
    --hf-tokenizer .../DeepSeek-V4-Flash-bf16-unpacked \
    --sglang-url http://127.0.0.1:30000 --max-len 8192 --out /tmp/ce_openhermes.json

结果解读
--------------
  - albaliang 留出 CE  ~ 在线 train/loss(约 2.0-2.2)  -> epoch 2/3 没有过拟合
    任务分布(平台期是真实的数据下限,不是记忆效应)。
  - albaliang 留出 CE  >> train/loss                     -> 过拟合;更早的 ckpt 更好。
  - openhermes(跨域)CE 在各 iter 间保持稳定           -> 通用指令跟随能力没有
    灾难性遗忘。

注意:SGLang 的 input_token_logprobs 链路已对照 miles/rollout/sglang_rollout.py 的
payload 约定做过校验,但尚未端到端跑通(在线训练期间没有空闲 GPU)。在信任聚合结果前,
请先验证第一条响应满足 len(input_token_logprobs)==len(input_ids)。
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
    """返回与 input_ids 对齐的 input_token_logprobs 列表(logP(tok_i | tok_<i))。"""
    payload = {
        "input_ids": input_ids,
        "return_logprob": True,
        "logprob_start_len": 0,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
    }
    r = requests.post(f"{url}/generate", json=payload, timeout=3600)
    r.raise_for_status()
    meta = r.json()["meta_info"]
    # input_token_logprobs:由 [logprob, token_id, text|None] 组成的列表;首个 token 的 logprob 为 null
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
        # 左截断,保留尾部(assistant 回复)
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

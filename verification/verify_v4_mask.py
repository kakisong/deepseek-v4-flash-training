#!/usr/bin/env python3
"""在转换后的 V4 格式 JSONL 上验证 V4 loss mask。

对前 --num 条记录，加载 tokenizer + MultiTurnLossMaskGenerator(type='deepseek_v4')，
运行 get_loss_mask，并打印：

  - token 数量 vs 源数据声明的 token_length
  - 逐 assistant 轮次：step_loss_mask、token 区间大小、loss=1 区间的解码预览
  - sanity check：loss=1 的 token 区域是否确实位于 `assistant` 渲染文本内？
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--hf-checkpoint", default="/data_train/kaynzhang/v4-sft/models/DeepSeek-V4-Flash-bf16-unpacked")
    ap.add_argument("--num", type=int, default=3)
    args = ap.parse_args()

    # 确保 miles 可被导入
    miles_root = "/data_train/kaynzhang/v4-sft/miles"
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

            # 解码连续的 loss=1 区间；展示前 3 个预览。
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

"""
Stage A 阻塞性检查：验证 V4 chat template + loss mask。

无需 GPU，不依赖 Megatron。可在头节点或任意机器上运行。

对每个样本打印 (token_id, mask, decoded_text) 表格，供人工逐项检查：
  - 角色标签（<|user|>、<|assistant|>、<|system|>）的 mask 应为 0
  - assistant 内容（以及可选的 thinking）的 mask 应为 1
  - user / system / tool 内容的 mask 应为 0

如发现错误，请按 README §1.4 将 gen_multi_turn_loss_mask_deepseek_v4
添加到 miles/utils/mask_utils.py，然后用 --loss-mask-type deepseek_v4 重新运行。

用法：
    python verification/verify_chat_template.py \\
        --hf-checkpoint        $MODELS/DeepSeek-V4-Flash-bf16 \\
        --chat-template-path   $REPO/templates/deepseek_v4.jinja \\
        --sample-data          $DATA/openhermes_v4.parquet \\
        --num-samples 5 \\
        --loss-mask-type qwen3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from miles.utils.mask_utils import MultiTurnLossMaskGenerator


GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-checkpoint", required=True, help="HF model dir; must contain tokenizer files")
    p.add_argument(
        "--chat-template-path",
        default=None,
        help="Path to the official V4 jinja template. If omitted, the tokenizer built-in is used (may be wrong).",
    )
    p.add_argument("--sample-data", required=True, help="parquet/jsonl with a `messages` field")
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--loss-mask-type", default="qwen3", choices=["qwen", "qwen3", "distill_qwen", "deepseek_v4"])
    p.add_argument("--max-print-tokens", type=int, default=200, help="print only the first N tokens per sample")
    return p.parse_args()


def load_samples(path: str, n: int) -> list[list[dict]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
        rows = df.head(n).to_dict(orient="records")
    elif suffix in (".jsonl", ".json"):
        rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                rows.append(json.loads(line))
    else:
        raise ValueError(f"unsupported sample format: {suffix}")

    out = []
    for r in rows:
        msgs = r.get("messages")
        if msgs is None:
            msgs = r.get("conversations")
        if msgs is None:
            raise ValueError(f"sample missing messages/conversations field: keys={list(r)}")
        out.append([dict(m) for m in msgs])
    return out


def patch_chat_template(tok: AutoTokenizer, jinja_path: str | None, mask_type: str) -> None:
    if mask_type == "deepseek_v4":
        # V4 不使用 jinja；它使用模型仓库中的 encoding_dsv4.py（由 mask_utils 加载）。
        print(f"{GREEN}[ok]{RESET} V4: using the model's encoding_dsv4.py (no jinja needed)")
        return
    if jinja_path is None:
        print(f"{YELLOW}[warn]{RESET} --chat-template-path not specified; using the tokenizer built-in template."
              " If the mask is wrong, try the official V4 jinja first.")
        return
    with open(jinja_path) as f:
        tok.chat_template = f.read()
    print(f"{GREEN}[ok]{RESET} chat_template replaced with {jinja_path}")


def color_for_mask(m: int) -> str:
    return GREEN if m == 1 else DIM


def render(token_ids: list[int], mask: list[int], tok: AutoTokenizer, max_print: int) -> None:
    assert len(token_ids) == len(mask), f"len mismatch: {len(token_ids)} vs {len(mask)}"
    n_train = sum(mask)
    n_total = len(mask)
    pct = 100.0 * n_train / max(n_total, 1)
    print(f"  total tokens: {n_total}, masked-in (train): {n_train} ({pct:.1f}%)")
    head = f"  {'IDX':>4} | {'TOK_ID':>7} | {'MASK':>4} | DECODED"
    print(head)
    print("  " + "-" * (len(head) - 2))
    n_show = min(len(token_ids), max_print)
    for i in range(n_show):
        tid = token_ids[i]
        m = mask[i]
        try:
            txt = tok.decode([tid])
        except Exception:
            txt = "<decode-err>"
        txt = txt.replace("\n", "\\n")
        c = color_for_mask(m)
        print(f"  {c}{i:>4} | {tid:>7} | {m:>4} | {txt!r}{RESET}")
    if len(token_ids) > max_print:
        print(f"  ... ({len(token_ids) - max_print} more tokens; increase --max-print-tokens to see more)")


def sanity_check(messages: list[dict], token_ids: list[int], mask: list[int]) -> list[str]:
    """启发式静态检查，用于捕获最常见的不匹配问题。"""
    warnings = []
    if sum(mask) == 0:
        warnings.append("mask is all 0 — training would learn nothing!")
    if all(mask):
        warnings.append("mask is all 1 — user/system tokens also count toward loss; serious bug!")
    has_assistant = any(m.get("role") == "assistant" for m in messages)
    if has_assistant and sum(mask) == 0:
        warnings.append("assistant turn present but mask is all 0 — chat template splits assistant tag incorrectly")
    return warnings


def main() -> int:
    args = parse_args()
    print(f"{DIM}[loading tokenizer from {args.hf_checkpoint}]{RESET}")
    tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    patch_chat_template(tok, args.chat_template_path, args.loss_mask_type)

    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type=args.loss_mask_type)
    print(f"{DIM}[mask generator: type={args.loss_mask_type}, system_message_length={gen.system_message_length}, "
          f"gen_token_length={gen.gen_token_length}]{RESET}")

    samples = load_samples(args.sample_data, args.num_samples)

    failed = 0
    for idx, messages in enumerate(samples):
        print(f"\n{YELLOW}=== sample {idx} (turns={len(messages)}) ==={RESET}")
        try:
            token_ids, mask = gen.get_loss_mask(messages)
        except Exception as e:
            print(f"  {RED}[error]{RESET} get_loss_mask raised: {e!r}")
            failed += 1
            continue
        render(token_ids, mask, tok, args.max_print_tokens)
        warns = sanity_check(messages, token_ids, mask)
        if warns:
            failed += 1
            for w in warns:
                print(f"  {RED}[warn]{RESET} {w}")
        else:
            print(f"  {GREEN}[ok]{RESET} static check passed (still verify token-by-token by hand)")

    print()
    if failed:
        print(f"{RED}{failed}/{len(samples)} samples failed automated checks — do not proceed to Stage A.{RESET}")
        return 1
    print(f"{GREEN}All samples passed static checks. Still verify the role-tag boundaries on the first few by hand.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

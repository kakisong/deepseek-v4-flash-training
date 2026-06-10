"""
SFT 数据 → Miles `messages` parquet。

支持的数据源:
  - HuggingFace 数据集 id (例如 teknium/OpenHermes-2.5)
  - 本地 jsonl
  - 本地 parquet

输出 schema:
  {
    "messages": [
      {"role": "user"|"assistant"|"system"|"tool", "content": "..."},
      # 含 thinking 的 assistant 轮次:
      {"role": "assistant", "reasoning_content": "...", "content": "..."},
    ],
    # 可选的 step_loss_mask=0 会对整个轮次关闭 mask (见 mask_utils.py)。
  }

--mask-mode 控制 thinking 片段是否参与训练:
  answer-only      : 只训练最终答案; 在数据准备阶段剥离 reasoning
  include-thinking : reasoning + 答案一起训练
  passthrough      : 不做转换 (数据已是目标格式时使用)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SUPPORTED_ROLES = {"user", "assistant", "system", "tool"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source",
        required=True,
        help="HF dataset id (must contain a /), or a local .jsonl / .parquet path",
    )
    p.add_argument("--output", required=True, help="target .parquet path")
    p.add_argument(
        "--mask-mode",
        choices=["answer-only", "include-thinking", "passthrough"],
        default="answer-only",
    )
    p.add_argument("--source-split", default="train", help="used when source is an HF dataset")
    p.add_argument("--limit", type=int, default=None, help="only take the first N rows (debug)")
    p.add_argument(
        "--input-field",
        default=None,
        help="if the source uses a non-standard field name (e.g. conversations / dialog), specify it here",
    )
    return p.parse_args()


def _convert_role(raw: str) -> str:
    raw = raw.lower().strip()
    if raw in {"human", "user"}:
        return "user"
    if raw in {"gpt", "assistant", "bot", "model"}:
        return "assistant"
    if raw == "system":
        return "system"
    if raw == "tool":
        return "tool"
    raise ValueError(f"unknown role: {raw}")


def _coerce_messages(raw_list: Iterable, mask_mode: str) -> list[dict]:
    """把各种常见的会话表示归一化为 OpenAI 风格的 messages。"""
    out: list[dict] = []
    for turn in raw_list:
        if isinstance(turn, dict):
            role = turn.get("role") or turn.get("from")
            if role is None:
                raise ValueError(f"turn missing role: {turn}")
            role = _convert_role(role)
            content = turn.get("content") or turn.get("value") or ""
            reasoning = turn.get("reasoning_content") or turn.get("thinking")

            msg: dict = {"role": role, "content": content}

            if role == "assistant" and reasoning:
                if mask_mode == "answer-only":
                    pass  # 丢弃 thinking 片段
                elif mask_mode == "include-thinking":
                    msg["reasoning_content"] = reasoning
                else:  # passthrough
                    msg["reasoning_content"] = reasoning

            out.append(msg)
        else:
            raise ValueError(f"unexpected turn type: {type(turn)}")
    return out


def _load_source(args: argparse.Namespace) -> Iterable[dict]:
    src: str = args.source
    field = args.input_field

    # 本地文件。
    p = Path(src)
    if p.exists():
        if p.suffix == ".jsonl":
            with open(p) as f:
                for i, line in enumerate(f):
                    if args.limit is not None and i >= args.limit:
                        break
                    yield json.loads(line)
            return
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            if args.limit:
                df = df.head(args.limit)
            for _, row in df.iterrows():
                yield row.to_dict()
            return
        raise ValueError(f"unsupported local source suffix: {p.suffix}")

    # HF 数据集 id。
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("source looks like an HF dataset id; install with `pip install datasets`") from e

    ds = load_dataset(src, split=args.source_split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    for row in ds:
        yield row


def _select_field(row: dict, hint: str | None) -> Iterable:
    candidates = [hint, "messages", "conversations", "conversation", "dialog", "turns"]
    for k in candidates:
        if k and k in row and row[k]:
            return row[k]
    raise ValueError(f"no message-like field found in row: keys={list(row)}")


def main() -> int:
    args = parse_args()
    out: list[dict] = []

    for i, row in enumerate(_load_source(args)):
        try:
            raw = _select_field(row, args.input_field)
            if args.mask_mode == "passthrough":
                # 预期数据源已是目标 messages 格式。
                msgs = list(raw)
            else:
                msgs = _coerce_messages(raw, args.mask_mode)
        except Exception as e:
            print(f"[skip row {i}] {e!r}")
            continue

        # 训练至少需要一个 user 轮次和一个 assistant 轮次。
        roles = {m.get("role") for m in msgs}
        if "assistant" not in roles:
            continue

        out.append({"messages": msgs})

    if not out:
        raise SystemExit("zero samples after parsing — check --source / --input-field / --mask-mode")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_parquet(out_path, index=False)
    print(f"[ok] wrote {len(out)} samples to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

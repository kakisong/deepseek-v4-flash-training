#!/usr/bin/env python3
"""将 albaliang 格式的 agent SFT JSONL 转换为 V4 兼容格式。

Source schema(每条记录占一行 JSON,记录之间可能有空行):
    {
      "id", "conversation_id", "request_id", "user_id",
      "tools":  str  (JSON 编码的 OpenAI tools 列表),
      "messages": [
          {"role", "content", "loss", "tool_call_id"},
          ...
      ],
      "token_length": int
    }
    role in {system, user, assistant, tool_call, tool}
    每个 `tool_call` 携带一次调用;`tool_call_id` 与其后的一个 `tool` 行对应。

Target schema(V4 兼容,每行一条记录):
    {
      "messages": [
          {"role": "system",     "content": ...},
          {"role": "user",       "content": ...},
          {"role": "assistant",  "content": ..., "tool_calls": [OpenAI 格式 ...],
                                  "step_loss_mask": 0|1,
                                  "_loss_content": 0|1, "_loss_tool_calls": [0|1, ...]},
          {"role": "tool",       "content": ..., "tool_call_id": ...},
          ...
      ],
      "tools": [OpenAI 格式的 tools 列表,由字符串解析而来],
      "token_length": int (原始值)
    }

为什么要自定义 _loss_* 字段:
    encoding_dsv4 把一条 assistant 消息渲染为一个整体(content + tool_calls)。
    默认的 mask 生成器把整个 assistant 区间标为 loss=1(或经 step_loss_mask
    标为 0)。然而源数据在各个子片段(文本 content 和每个 tool_call)上
    携带逐消息的 `loss` 标志。为保留这一精度,我们把逐片段的 loss
    附在合并后的 assistant 消息上;V4 的 mask 生成器会在后续补丁
    (gen_multi_turn_loss_mask_deepseek_v4)中消费它们。

过滤:
    丢弃 token_length 超过 --max-tokens(默认 32768)的记录,以及存在解析错误的记录。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_tool_call_content(raw: str) -> tuple[str, str] | None:
    """源 tool_call.content = JSON('{"name":..., "arguments": <json-string>}')。返回 (name, arguments-string)。"""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    name = obj.get("name")
    arguments = obj.get("arguments", "")
    if name is None:
        return None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return name, arguments


def convert_conversation(src_msgs: list[dict], drop_stats: dict) -> list[dict] | None:
    """遍历源 messages;产出 V4 兼容的消息列表。返回 None 表示丢弃该对话。"""
    out: list[dict] = []
    pending_assistant: dict | None = None  # 最近一个正在把 tool_calls 折叠进去的 assistant

    def flush_pending():
        nonlocal pending_assistant
        if pending_assistant is not None:
            out.append(pending_assistant)
            pending_assistant = None

    for m in src_msgs:
        role = m.get("role")
        content = m.get("content", "")
        loss = int(m.get("loss", 0) or 0)

        if role == "system":
            flush_pending()
            out.append({"role": "system", "content": content})

        elif role == "user":
            flush_pending()
            out.append({"role": "user", "content": content})

        elif role == "assistant":
            flush_pending()
            pending_assistant = {
                "role": "assistant",
                "content": content,
                "tool_calls": [],
                "_loss_content": loss,
                "_loss_tool_calls": [],
            }

        elif role == "tool_call":
            parsed = parse_tool_call_content(content)
            if parsed is None:
                drop_stats["bad_tool_call_json"] += 1
                return None
            name, arguments = parsed
            tc_id = m.get("tool_call_id") or f"_synth_{len(out)}_{len(pending_assistant['tool_calls']) if pending_assistant else 0}"
            if pending_assistant is None:
                # 孤立的 tool_call,前面没有 assistant 轮次。
                # 合成一个占位 assistant 来承载它(罕见;通过 stats 记录)。
                drop_stats["orphan_tool_call_synth_assistant"] += 1
                pending_assistant = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [],
                    "_loss_content": 0,
                    "_loss_tool_calls": [],
                }
            pending_assistant["tool_calls"].append({
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
            pending_assistant["_loss_tool_calls"].append(loss)

        elif role == "tool":
            flush_pending()
            out.append({
                "role": "tool",
                "content": content,
                "tool_call_id": m.get("tool_call_id", ""),
            })

        else:
            drop_stats["unknown_role"] += 1
            return None

    flush_pending()

    # 计算每个合并后 assistant 轮次的 step_loss_mask = 所有子片段 loss 的最大值。
    # 在 albaliang 057 数据上这是精确的(每一轮内各子 loss 彼此一致;
    # 在 5k 条记录抽样上验证为 0/5900 不一致)。保留逐子片段的
    # _loss_* 字段作为留痕依据,便于将来实现精确的子区间 mask。
    for msg in out:
        if msg.get("role") != "assistant":
            continue
        subs = [msg.get("_loss_content", 0)] + list(msg.get("_loss_tool_calls", []))
        msg["step_loss_mask"] = max(subs) if subs else 0
        if not msg.get("tool_calls"):
            msg.pop("tool_calls", None)
            msg.pop("_loss_tool_calls", None)

    # 校验:至少要有一个 step_loss_mask=1 的 assistant 轮次(否则没有训练信号)。
    has_any_loss = any(
        msg.get("role") == "assistant" and msg.get("step_loss_mask", 0) == 1
        for msg in out
    )
    if not has_any_loss:
        drop_stats["no_training_signal"] += 1
        return None

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, help="Source albaliang-format jsonl")
    ap.add_argument("--out", dest="dst", required=True, help="Destination V4-compatible jsonl")
    ap.add_argument("--max-tokens", type=int, default=32768, help="Drop records with token_length above this (default 32768)")
    ap.add_argument("--limit", type=int, default=0, help="Stop after this many records (0 = all)")
    ap.add_argument("--progress-every", type=int, default=10000, help="Log progress every N records read")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"source not found: {src}", file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "read": 0,
        "blank_lines": 0,
        "bad_outer_json": 0,
        "bad_tools_json": 0,
        "bad_tool_call_json": 0,
        "orphan_tool_call_synth_assistant": 0,
        "unknown_role": 0,
        "no_training_signal": 0,
        "too_long": 0,
        "kept": 0,
        "min_kept_token_length": None,
        "max_kept_token_length": None,
        "sum_kept_token_length": 0,
    }

    with src.open() as f_in, dst.open("w") as f_out:
        for line in f_in:
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            stats["read"] += 1
            if args.limit and stats["read"] > args.limit:
                stats["read"] -= 1
                break

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["bad_outer_json"] += 1
                continue

            tlen = rec.get("token_length")
            if not isinstance(tlen, int) or tlen <= 0:
                # 不信任未知长度;为安全起见跳过
                stats["too_long"] += 1
                continue
            if tlen > args.max_tokens:
                stats["too_long"] += 1
                continue

            tools_raw = rec.get("tools", "")
            tools_list: list = []
            if isinstance(tools_raw, str) and tools_raw:
                try:
                    tools_list = json.loads(tools_raw)
                except json.JSONDecodeError:
                    stats["bad_tools_json"] += 1
                    continue
            elif isinstance(tools_raw, list):
                tools_list = tools_raw

            messages = convert_conversation(rec.get("messages", []), stats)
            if messages is None:
                continue

            out_rec = {"messages": messages, "tools": tools_list, "token_length": tlen}
            f_out.write(json.dumps(out_rec, ensure_ascii=False))
            f_out.write("\n")

            stats["kept"] += 1
            stats["sum_kept_token_length"] += tlen
            if stats["min_kept_token_length"] is None or tlen < stats["min_kept_token_length"]:
                stats["min_kept_token_length"] = tlen
            if stats["max_kept_token_length"] is None or tlen > stats["max_kept_token_length"]:
                stats["max_kept_token_length"] = tlen

            if stats["read"] % args.progress_every == 0:
                print(f"... read={stats['read']} kept={stats['kept']} too_long={stats['too_long']}", flush=True)

    if stats["kept"] > 0:
        stats["mean_kept_token_length"] = stats["sum_kept_token_length"] // stats["kept"]
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

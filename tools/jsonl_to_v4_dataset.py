#!/usr/bin/env python3
"""Convert albaliang-format agent SFT JSONL to V4-compatible format.

Source schema (per record, single JSON line, may have blank lines between records):
    {
      "id", "conversation_id", "request_id", "user_id",
      "tools":  str  (JSON-encoded list of OpenAI tools),
      "messages": [
          {"role", "content", "loss", "tool_call_id"},
          ...
      ],
      "token_length": int
    }
    role in {system, user, assistant, tool_call, tool}
    Each `tool_call` carries one call; `tool_call_id` matches a following `tool` row.

Target schema (V4-compatible, one record per line):
    {
      "messages": [
          {"role": "system",     "content": ...},
          {"role": "user",       "content": ...},
          {"role": "assistant",  "content": ..., "tool_calls": [openai-format ...],
                                  "step_loss_mask": 0|1,
                                  "_loss_content": 0|1, "_loss_tool_calls": [0|1, ...]},
          {"role": "tool",       "content": ..., "tool_call_id": ...},
          ...
      ],
      "tools": [openai-format tools list, parsed from string],
      "token_length": int (original)
    }

Why custom _loss_* fields:
    encoding_dsv4 renders an assistant message as one unit (content + tool_calls). The
    default mask generator marks the whole assistant span as loss=1 (or 0 via
    step_loss_mask). The source data, however, carries a per-message `loss` flag on the
    individual sub-pieces (text content and each tool_call). To preserve precision we
    attach the per-piece losses to the merged assistant message; the V4 mask generator
    consumes them in a follow-up patch (gen_multi_turn_loss_mask_deepseek_v4).

Filter:
    Drop records with token_length above --max-tokens (default 32768) or with parse errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_tool_call_content(raw: str) -> tuple[str, str] | None:
    """Source tool_call.content = JSON('{"name":..., "arguments": <json-string>}'). Return (name, arguments-string)."""
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
    """Walk source messages; emit V4-compatible message list. Return None to drop the conv."""
    out: list[dict] = []
    pending_assistant: dict | None = None  # the most recent assistant we're folding tool_calls into

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
                # Orphan tool_call without a preceding assistant turn.
                # Synthesize a stub assistant to hold it (rare; logged via stats).
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

    # Compute step_loss_mask per merged assistant turn = max of all sub-piece losses.
    # In the albaliang 057 data this is exact (sub-losses agree within every turn;
    # verified 0/5900 mismatch on a 5k-record sample). Keep the per-sub-piece
    # _loss_* fields as forensic provenance for any future precise sub-span mask.
    for msg in out:
        if msg.get("role") != "assistant":
            continue
        subs = [msg.get("_loss_content", 0)] + list(msg.get("_loss_tool_calls", []))
        msg["step_loss_mask"] = max(subs) if subs else 0
        if not msg.get("tool_calls"):
            msg.pop("tool_calls", None)
            msg.pop("_loss_tool_calls", None)

    # Validate: at least one assistant turn with step_loss_mask=1 (otherwise no training signal).
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
                # we don't trust unknown lengths; skip for safety
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

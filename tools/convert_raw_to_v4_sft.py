#!/usr/bin/env python3
"""将原始 agent-SFT JSONL(albaliang 格式)转换为 V4-Flash 可直接训练的 JSONL。

tools/jsonl_to_v4_dataset.py 的加固版、一站式超集:
  * 接受一个或多个原始文件,并把它们拼接进同一个 --out
  * 自动检测已经是 V4 格式的记录并原样透传,
    因此在混合 / 部分已转换的数据集上重复运行是安全的(data/ 下的 le128k/le134k
    文件名虽如此,其实已是 V4 格式 — 这样可避免二次折叠)
  * float 安全的 `loss` 解析;对空/None 的 tool_call_id 做防御性处理
  * 长度过滤与训练用的 --seq-length 对齐,并附带醒目的提醒
  * 完整的丢弃原因计数输出到 stderr,以及可选的 --stats-out json

────────────────────────────────────────────────────────────────────────────
SOURCE schema(原始 albaliang;每行一个 JSON 对象,允许空行):
    {
      "id","conversation_id","request_id","user_id",      # 丢弃
      "tools": "<JSON 编码的 OpenAI tools 列表>",          # 注意:是字符串
      "messages": [
        {"role","content","loss": 0|1 (float),"tool_call_id"},
        ...
      ],
      "token_length": int
    }
    role in {system, user, assistant, tool_call, tool}
    - `tool_call` 行的 content 是 JSON:{"name":..,"arguments":"<json-string>"}
    - 每个 `tool_call` 都由其后的一个 `tool` 行应答(共享 tool_call_id)

TARGET schema(V4-Flash;即 Miles `--prompt-data ... --input-key messages` 读取的格式):
    {
      "messages": [
        {"role":"system","content":...},
        {"role":"user","content":...},
        {"role":"assistant","content":...,
                            "tool_calls":[{"id","type":"function",
                                           "function":{"name","arguments"}}],
                            "step_loss_mask":0|1,
                            "_loss_content":0|1,"_loss_tool_calls":[0|1,...]},
        {"role":"tool","content":...,"tool_call_id":...}
      ],
      "tools": [ ...解析后的 OpenAI tools 列表... ],
      "token_length": int
    }
    - tool_call 行会被折叠进前一个 assistant 的 `tool_calls`
    - 逐消息的 `loss` 标志会合并为按 assistant 轮次的 `step_loss_mask`
      (= 该轮各子片段的最大值)。逐片段的 `_loss_content`/`_loss_tool_calls`
      是有意保留的:Miles 的 gen_multi_turn_loss_mask_deepseek_v4 会消费它们,
      以便精确地 mask 子区间(assistant 文本 vs 每个 tool_call)。

用法
    python3 tools/convert_raw_to_v4_sft.py RAW1.jsonl [RAW2.jsonl ...] \
        --out  $V4_DATA/myrun_v4_le256k.jsonl \
        --max-tokens 262144            # 必须 <= 训练的 --seq-length
                                       #   256K 训练 -> 262144 ; 128K 训练 -> 131072

纯标准库,单进程,仅用 CPU。约 18 GB / 5 万条对话,几分钟内完成。
丢弃并按原因计数:解析错误、未知 role、损坏的 tool_call JSON、
超长、缺少长度,以及没有任何参与训练的 assistant 轮次的对话。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _as_loss(v) -> int:
    """原始 `loss` 可能是 int、float(1.0)、str 或缺失。>= 0.5 视为参与训练。"""
    try:
        return 1 if float(v) >= 0.5 else 0
    except (TypeError, ValueError):
        return 0


def parse_tool_call_content(raw: str):
    """tool_call.content = JSON('{"name":..,"arguments":<json-string>}')。返回 (name, args_str) | None。"""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    name = obj.get("name")
    if name is None:
        return None
    arguments = obj.get("arguments", "")
    if not isinstance(arguments, str):
        # OpenAI 要求 function.arguments 必须是字符串;若上游给的是对象则重新编码。
        arguments = json.dumps(arguments, ensure_ascii=False)
    return name, arguments


def is_already_v4(rec: dict) -> bool:
    """对此前运行已产出的记录返回 True:没有 tool_call role,且已有
    assistant 轮次携带 step_loss_mask。把这类记录原样透传
    可避免二次折叠(那会破坏 tool_calls / step_loss_mask)。"""
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False
    if any(m.get("role") == "tool_call" for m in msgs):
        return False
    return any(m.get("role") == "assistant" and "step_loss_mask" in m for m in msgs)


def normalize_tools(tools_raw, stats) -> list | None:
    """源 `tools` 是 JSON 字符串;目标需要解析后的列表。返回 None => 丢弃该记录。"""
    if isinstance(tools_raw, list):
        return tools_raw
    if isinstance(tools_raw, str) and tools_raw.strip():
        try:
            parsed = json.loads(tools_raw)
        except json.JSONDecodeError:
            stats["bad_tools_json"] += 1
            return None
        return parsed if isinstance(parsed, list) else []
    return []


def convert_conversation(src_msgs: list, stats: dict) -> list | None:
    """遍历源 messages;产出 V4 消息列表。返回 None => 丢弃该对话。"""
    out: list = []
    pending_assistant: dict | None = None  # 正在把 tool_calls 折叠进去的 assistant

    def flush():
        nonlocal pending_assistant
        if pending_assistant is not None:
            out.append(pending_assistant)
            pending_assistant = None

    for m in src_msgs:
        role = m.get("role")
        content = m.get("content", "") or ""
        loss = _as_loss(m.get("loss", 0))

        if role == "system":
            flush()
            out.append({"role": "system", "content": content})

        elif role == "user":
            flush()
            out.append({"role": "user", "content": content})

        elif role == "assistant":
            flush()
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
                stats["bad_tool_call_json"] += 1
                return None
            name, arguments = parsed
            if pending_assistant is None:
                # tool_call 前面没有 assistant;合成一个占位 assistant 来承载它。
                stats["orphan_tool_call_synth_assistant"] += 1
                pending_assistant = {
                    "role": "assistant", "content": "",
                    "tool_calls": [], "_loss_content": 0, "_loss_tool_calls": [],
                }
            tc_id = m.get("tool_call_id") or f"_synth_{len(out)}_{len(pending_assistant['tool_calls'])}"
            pending_assistant["tool_calls"].append({
                "id": tc_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })
            pending_assistant["_loss_tool_calls"].append(loss)

        elif role == "tool":
            flush()
            out.append({
                "role": "tool",
                "content": content,
                "tool_call_id": m.get("tool_call_id") or "",
            })

        else:
            stats["unknown_role"] += 1
            return None

    flush()

    # 每个合并后 assistant 的 step_loss_mask = 其各片段的最大值(content + 每个 tool_call)。
    for msg in out:
        if msg.get("role") != "assistant":
            continue
        subs = [msg.get("_loss_content", 0)] + list(msg.get("_loss_tool_calls", []))
        msg["step_loss_mask"] = max(subs) if subs else 0
        if not msg.get("tool_calls"):
            msg.pop("tool_calls", None)
            msg.pop("_loss_tool_calls", None)

    # 至少需要一个参与训练的 assistant 轮次,否则该对话只是无用负担。
    if not any(m.get("role") == "assistant" and m.get("step_loss_mask", 0) == 1 for m in out):
        stats["no_training_signal"] += 1
        return None
    return out


def process_record(rec: dict, max_tokens: int, keep_unknown_len: bool, stats: dict) -> dict | None:
    tlen = rec.get("token_length")
    if not isinstance(tlen, int) or tlen <= 0:
        if not keep_unknown_len:
            stats["unknown_length"] += 1
            return None
        tlen = 0
    elif tlen > max_tokens:
        stats["too_long"] += 1
        return None

    if is_already_v4(rec):
        tools = normalize_tools(rec.get("tools", []), stats)
        if tools is None:
            return None
        stats["already_v4_passthrough"] += 1
        return {"messages": rec["messages"], "tools": tools, "token_length": tlen}

    tools = normalize_tools(rec.get("tools", ""), stats)
    if tools is None:
        return None
    messages = convert_conversation(rec.get("messages", []), stats)
    if messages is None:
        return None
    return {"messages": messages, "tools": tools, "token_length": tlen}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="one or more raw albaliang jsonl files (concatenated into --out)")
    ap.add_argument("--out", required=True, help="destination V4-format jsonl")
    ap.add_argument("--max-tokens", type=int, default=262144,
                    help="drop convs with token_length above this; MUST be <= training "
                         "--seq-length (256K->262144, 128K->131072). default 262144")
    ap.add_argument("--keep-unknown-length", action="store_true",
                    help="keep records whose token_length is missing/invalid (default: drop)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N input records (debug)")
    ap.add_argument("--progress-every", type=int, default=20000)
    ap.add_argument("--stats-out", default=None, help="also write the stats dict to this json path")
    args = ap.parse_args()

    for p in args.inputs:
        if not Path(p).exists():
            print(f"[fatal] input not found: {p}", file=sys.stderr)
            return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seq_hint = {262144: "256K", 131072: "128K", 65536: "64K"}.get(args.max_tokens, f"{args.max_tokens} tok")
    print(f"[align] --max-tokens={args.max_tokens} (~{seq_hint}); "
          f"training --seq-length MUST be >= this, else long convs truncate/error", file=sys.stderr)

    stats = {k: 0 for k in (
        "read", "blank_lines", "bad_outer_json", "bad_tools_json", "bad_tool_call_json",
        "orphan_tool_call_synth_assistant", "unknown_role", "no_training_signal",
        "too_long", "unknown_length", "already_v4_passthrough", "kept")}
    tl_sum, tl_max, tl_min = 0, 0, None

    t0 = time.time()
    stop = False
    with out_path.open("w") as f_out:
        for src in args.inputs:
            if stop:
                break
            print(f"[read] {src}", file=sys.stderr)
            with open(src) as f_in:
                for line in f_in:
                    if not line.strip():
                        stats["blank_lines"] += 1
                        continue
                    if args.limit and stats["read"] >= args.limit:
                        stop = True
                        break
                    stats["read"] += 1
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_outer_json"] += 1
                        continue
                    out_rec = process_record(rec, args.max_tokens, args.keep_unknown_length, stats)
                    if out_rec is None:
                        continue
                    f_out.write(json.dumps(out_rec, ensure_ascii=False))
                    f_out.write("\n")
                    stats["kept"] += 1
                    tl = out_rec["token_length"]
                    tl_sum += tl
                    tl_max = max(tl_max, tl)
                    tl_min = tl if tl_min is None else min(tl_min, tl)
                    if stats["read"] % args.progress_every == 0:
                        print(f"... read={stats['read']} kept={stats['kept']} "
                              f"too_long={stats['too_long']} elapsed={time.time()-t0:.0f}s",
                              file=sys.stderr, flush=True)

    stats["min_kept_token_length"] = tl_min or 0
    stats["max_kept_token_length"] = tl_max
    stats["mean_kept_token_length"] = (tl_sum // stats["kept"]) if stats["kept"] else 0
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["out"] = str(out_path)

    report = json.dumps(stats, indent=2, ensure_ascii=False)
    print(report)
    if args.stats_out:
        Path(args.stats_out).write_text(report)
    if stats["kept"] == 0:
        print("[warn] zero kept — check --max-tokens and input format", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

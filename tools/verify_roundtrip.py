"""
Verification harness for the BF16 -> dpsk FP8/FP4 round-trip.

Subcommands:
  micro     : per-tensor closed loop on the ORIGINAL dir. For sampled tensors:
              (a) dequant original (q,s) -> bf16_a, and confirm bf16_a == the
                  bf16-unpacked source tensor (validates name map + dequant);
              (b) requant bf16_src with the forward quantizers -> dequant again
                  -> bf16_b, assert bf16_b == bf16_a BITWISE (idempotency);
              (c) report byte-match ratio between requant (q,s) and original.
  keymap    : every dpsk weight name in TEMPLATE maps to an HF name present in
              the bf16-unpacked index (coverage of the name map).
  struct    : compare two dpsk dirs (A: new vs original) — tensor name set,
              per-tensor dtype + shape, and index weight_map keys must be equal.
  cmp-bf16  : compare two HF bf16 dirs (B: re-dequant'd new vs bf16-unpacked)
              tensor-by-tensor, BITWISE; report mismatches + max abs diff.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_bf16_to_megablocks import quant_fp4, quant_fp8  # noqa: E402
from megablocks_to_hf_bf16 import (  # noqa: E402
    _is_routed_expert,
    dequant_fp4,
    dequant_fp8,
    map_name_megablocks_to_hf,
)


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def _index(d):
    return json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]


class Dir:
    def __init__(self, path):
        self.path = path
        self.map = _index(path)
        self._h = {}

    def get(self, name):
        shard = self.map[name]
        if shard not in self._h:
            self._h[shard] = safe_open(os.path.join(self.path, shard), framework="pt")
        return self._h[shard].get_tensor(name)


def cmd_micro(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig = Dir(args.template)
    hf = Dir(args.hf_src)

    # representative dpsk weight bases across all quant classes
    samples = [
        "layers.0.attn.wq_a",            # fp8
        "layers.0.attn.wq_b",            # fp8 (other block shape)
        "layers.0.attn.wkv",             # fp8
        "layers.0.attn.wo_a",            # fp8
        "layers.0.attn.wo_b",            # fp8
        "layers.0.ffn.shared_experts.w1",# fp8 (shared)
        "layers.0.ffn.experts.0.w1",     # fp4 (routed)
        "layers.0.ffn.experts.0.w2",     # fp4 (routed, K=2048)
        "layers.0.ffn.experts.7.w3",     # fp4 (routed)
        "layers.21.ffn.experts.100.w1",  # fp4 deeper layer/expert
        "mtp.0.e_proj",                  # fp8 (mtp)
        "mtp.0.ffn.experts.0.w1",        # fp4 (mtp routed)
    ]
    n_ok = 0
    for base in samples:
        wn, sn = base + ".weight", base + ".scale"
        if wn not in orig.map:
            print(f"[skip] {wn} not in template"); continue
        q = orig.get(wn).to(dev)
        s = orig.get(sn).to(dev)
        is_fp4 = (q.dtype == torch.int8 and _is_routed_expert(wn))
        deq = dequant_fp4 if is_fp4 else dequant_fp8
        bf16_a = deq(q, s)                                   # original dequant

        hf_name = map_name_megablocks_to_hf(wn)
        bf16_src = hf.get(hf_name).to(dev)
        eq_src = torch.equal(bf16_a, bf16_src)

        qw, sc = (quant_fp4 if is_fp4 else quant_fp8)(bf16_src)
        assert tuple(qw.shape) == tuple(q.shape), f"{wn} requant weight shape {qw.shape} != {q.shape}"
        assert tuple(sc.shape) == tuple(s.shape), f"{wn} requant scale shape {sc.shape} != {s.shape}"
        assert qw.dtype == q.dtype and sc.dtype == s.dtype
        bf16_b = deq(qw.to(dev), sc.to(dev))                 # requant -> dequant
        eq_round = torch.equal(bf16_b, bf16_a)

        wbyte = (qw.view(torch.uint8) == q.view(torch.uint8)).float().mean().item()
        sbyte = (sc.view(torch.uint8) == s.view(torch.uint8)).float().mean().item()
        maxdiff = (bf16_b.float() - bf16_a.float()).abs().max().item()
        tag = "fp4" if is_fp4 else "fp8"
        status = "OK" if (eq_src and eq_round) else "FAIL"
        print(f"[{status}] {base:32s} {tag} shape={tuple(bf16_a.shape)} "
              f"deq==src:{eq_src} round-trip-bitwise:{eq_round} maxdiff:{maxdiff:.3e} "
              f"w-byte-match:{wbyte:.3f} s-byte-match:{sbyte:.3f}")
        if not (eq_src and eq_round):
            raise SystemExit(f"micro FAILED on {base}")
        n_ok += 1
    print(f"\n[micro] {n_ok} tensors passed (deq==src AND requant->deq bitwise-identical)")


def cmd_keymap(args):
    tmpl_map = _index(args.template)
    hf_map = _index(args.hf_src)
    missing = []
    for name in tmpl_map:
        if name.endswith(".scale"):
            continue  # scales are regenerated, not sourced
        hf_name = map_name_megablocks_to_hf(name)
        if hf_name not in hf_map:
            missing.append((name, hf_name))
    print(f"[keymap] template weight keys: {sum(1 for k in tmpl_map if not k.endswith('.scale'))}")
    if missing:
        print(f"[keymap] MISSING {len(missing)} mappings in hf-src, e.g.:")
        for a, b in missing[:20]:
            print(f"    {a}  ->  {b}   (NOT FOUND)")
        raise SystemExit("keymap FAILED")
    print("[keymap] OK — every template weight maps to an existing hf-src tensor")


def cmd_struct(args):
    a, b = args.a, args.b
    ia, ib = _index(a), _index(b)
    if set(ia) != set(ib):
        print(f"[struct] index key mismatch: only-in-a={len(set(ia)-set(ib))} only-in-b={len(set(ib)-set(ia))}")
        for k in list(set(ia) ^ set(ib))[:20]:
            print("   ", k)
        raise SystemExit("struct FAILED (index keys)")
    # per-shard header dtype+shape
    bad = 0
    shards = sorted(set(ia.values()))
    for sh in shards:
        ha = _header(os.path.join(a, sh))
        hb = _header(os.path.join(b, sh))
        if set(ha) != set(hb):
            print(f"[struct] {sh}: tensor set differs"); bad += 1; continue
        for k in ha:
            if ha[k]["dtype"] != hb[k]["dtype"] or ha[k]["shape"] != hb[k]["shape"]:
                print(f"[struct] {sh}:{k}: a={ha[k]['dtype']}{ha[k]['shape']} b={hb[k]['dtype']}{hb[k]['shape']}")
                bad += 1
    if bad:
        raise SystemExit(f"struct FAILED ({bad} mismatches)")
    print(f"[struct] OK — {len(ia)} tensors across {len(shards)} shards: identical names/dtype/shape")


def cmd_cmp_dequant(args):
    """Verification B (cheap, full-coverage, no 543GB write):

    Dequant every weight of the NEW dpsk ckpt with the SAME functions the reverse
    tool (megablocks_to_hf_bf16.py) uses, map to HF name, and compare BITWISE to
    the bf16-unpacked dir. If this is identical for all tensors, running the real
    reverse tool is guaranteed to reproduce bf16-unpacked exactly.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new = Dir(args.new)
    hf = Dir(args.hf)
    names = [k for k in new.map if not k.endswith(".scale")]
    names.sort()
    if args.sample and args.sample < len(names):
        step = len(names) // args.sample
        names = names[::step][: args.sample]
    n_mismatch = 0
    max_abs = 0.0
    for i, wn in enumerate(names):
        base = wn[: -len(".weight")] if wn.endswith(".weight") else wn
        sn = base + ".scale"
        q = new.get(wn).to(dev)
        if sn in new.map:
            s = new.get(sn).to(dev)
            is_fp4 = (q.dtype == torch.int8 and _is_routed_expert(wn))
            deq = (dequant_fp4 if is_fp4 else dequant_fp8)(q, s)
        else:
            deq = q  # copy tensor
        ref = hf.get(map_name_megablocks_to_hf(wn)).to(dev)
        if deq.dtype != ref.dtype or deq.shape != ref.shape:
            print(f"[cmp-dequant] {wn}: dtype/shape {deq.dtype}{tuple(deq.shape)} vs {ref.dtype}{tuple(ref.shape)}")
            n_mismatch += 1; continue
        if not torch.equal(deq, ref):
            d = (deq.float() - ref.float()).abs().max().item()
            max_abs = max(max_abs, d); n_mismatch += 1
            if n_mismatch <= 20:
                print(f"[cmp-dequant] DIFF {wn}: max_abs={d:.3e}")
        if (i + 1) % 2000 == 0:
            print(f"  ...{i+1}/{len(names)} checked, mismatches={n_mismatch}")
    print(f"\n[cmp-dequant] checked {len(names)} weights; mismatches={n_mismatch}; max_abs_diff={max_abs:.3e}")
    if n_mismatch:
        raise SystemExit("cmp-dequant: NOT bitwise identical")
    print("[cmp-dequant] OK — re-dequant of NEW ckpt is bitwise identical to bf16-unpacked")


def cmd_cmp_bf16(args):
    a, b = Dir(args.a), Dir(args.b)
    if set(a.map) != set(b.map):
        print(f"[cmp-bf16] key set differs: only-a={len(set(a.map)-set(b.map))} only-b={len(set(b.map)-set(a.map))}")
        raise SystemExit("cmp-bf16 FAILED (keys)")
    names = sorted(a.map)
    if args.sample and args.sample < len(names):
        step = len(names) // args.sample
        names = names[::step][: args.sample]
    n_mismatch = 0
    max_abs = 0.0
    for i, name in enumerate(names):
        ta, tb = a.get(name), b.get(name)
        if ta.dtype != tb.dtype or ta.shape != tb.shape:
            print(f"[cmp-bf16] {name}: dtype/shape differ {ta.dtype}{ta.shape} vs {tb.dtype}{tb.shape}")
            n_mismatch += 1; continue
        if not torch.equal(ta, tb):
            d = (ta.float() - tb.float()).abs().max().item()
            max_abs = max(max_abs, d)
            n_mismatch += 1
            if n_mismatch <= 20:
                print(f"[cmp-bf16] DIFF {name}: max_abs={d:.3e}")
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{len(names)} checked, mismatches={n_mismatch}")
    print(f"\n[cmp-bf16] checked {len(names)} tensors; mismatches={n_mismatch}; max_abs_diff={max_abs:.3e}")
    if n_mismatch:
        raise SystemExit("cmp-bf16: NOT bitwise identical")
    print("[cmp-bf16] OK — bitwise identical")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("micro"); p.add_argument("--template", required=True); p.add_argument("--hf-src", required=True); p.set_defaults(fn=cmd_micro)
    p = sub.add_parser("keymap"); p.add_argument("--template", required=True); p.add_argument("--hf-src", required=True); p.set_defaults(fn=cmd_keymap)
    p = sub.add_parser("struct"); p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.set_defaults(fn=cmd_struct)
    p = sub.add_parser("cmp-dequant"); p.add_argument("--new", required=True); p.add_argument("--hf", required=True); p.add_argument("--sample", type=int, default=0); p.set_defaults(fn=cmd_cmp_dequant)
    p = sub.add_parser("cmp-bf16"); p.add_argument("--a", required=True); p.add_argument("--b", required=True); p.add_argument("--sample", type=int, default=0); p.set_defaults(fn=cmd_cmp_bf16)

    args = ap.parse_args()
    args.fn(args)

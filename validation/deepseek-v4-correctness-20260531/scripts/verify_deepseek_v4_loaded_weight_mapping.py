#!/usr/bin/env python3
"""将选定的已加载 Miles DeepSeek-V4 权重与原始 checkpoint tensor 进行比较。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.training import get_model

import miles_plugins.mbridge  # noqa: F401
from miles.backends.megatron_utils.arguments import set_default_megatron_args
from miles.backends.megatron_utils.checkpoint import load_checkpoint
from miles.backends.megatron_utils.initialize import init
from miles.backends.megatron_utils.model_provider import get_model_provider_func
from miles.utils.logging_utils import configure_logger


SELECTED = {
    "module.embedding.word_embeddings.weight": "embedding.word_embeddings.weight",
    "module.decoder.layers.0.self_attention.attn_sink": "decoder.layers.0.self_attention.attn_sink",
    "module.decoder.layers.0.self_attention.wq_a.weight": "decoder.layers.0.self_attention.wq_a.weight",
    "module.decoder.layers.0.self_attention.q_norm.weight": "decoder.layers.0.self_attention.q_norm.weight",
    "module.decoder.layers.0.self_attention.wq_b.weight": "decoder.layers.0.self_attention.wq_b.weight",
    "module.decoder.layers.0.self_attention.wkv.weight": "decoder.layers.0.self_attention.wkv.weight",
    "module.decoder.layers.0.self_attention.kv_norm.weight": "decoder.layers.0.self_attention.kv_norm.weight",
    "module.decoder.layers.0.self_attention.wo_a.weight": "decoder.layers.0.self_attention.wo_a.weight",
    "module.decoder.layers.0.self_attention.wo_b.weight": "decoder.layers.0.self_attention.wo_b.weight",
    "module.decoder.layers.0.pre_mlp_layernorm.weight": "decoder.layers.0.pre_mlp_layernorm.weight",
    "module.decoder.layers.0.mlp.router.weight": "decoder.layers.0.mlp.router.weight",
    "module.decoder.layers.0.mlp.router.tid2eid": "decoder.layers.0.mlp.router.tid2eid",
    "module.decoder.layers.0.mlp.shared_experts.linear_fc1.weight": "decoder.layers.0.mlp.shared_experts.linear_fc1.weight",
    "module.decoder.layers.0.mlp.shared_experts.linear_fc2.weight": "decoder.layers.0.mlp.shared_experts.linear_fc2.weight",
    "module.decoder.final_layernorm.weight": "decoder.final_layernorm.weight",
    "module.output_layer.weight": "output_layer.weight",
}


GROUPED_EXPERTS = {
    "module.decoder.layers.0.mlp.experts.linear_fc1": "decoder.layers.0.mlp.experts.experts.linear_fc1.weight",
    "module.decoder.layers.0.mlp.experts.linear_fc2": "decoder.layers.0.mlp.experts.experts.linear_fc2.weight",
}


def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--hf-checkpoint", type=str, required=False)
    parser.add_argument("--qkv-format", choices=["thd", "bshd"], default="thd")
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--checkpoint-release-dir", type=Path, required=True)
    return parser


def _init_distributed() -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )


def _parse_args():
    args = parse_args(extra_args_provider=add_args)
    args = set_default_megatron_args(args)
    args.no_load_optim = True
    args.no_load_rng = True
    args.finetune = True
    args.__dict__.setdefault("custom_megatron_before_log_prob_hook_path", None)
    args.save_interval = getattr(args, "save_interval", None) or 1
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    validate_args(args)
    args.variable_seq_lengths = True
    return args


def _load_raw_state(checkpoint_release_dir: Path) -> dict[str, torch.Tensor]:
    spec = importlib.util.spec_from_file_location("official_full_forward", "/tmp/verify_deepseek_v4_official_full_forward.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._load_megatron_state(checkpoint_release_dir, num_layers=1)


def _diff_stats(loaded: torch.Tensor, raw: torch.Tensor, chunk_size: int = 16 * 1024 * 1024) -> dict[str, Any]:
    loaded_flat = loaded.flatten()
    raw_flat = raw.flatten()
    max_abs = 0.0
    sum_abs = 0.0
    nonzero = 0
    numel = loaded_flat.numel()
    for start in range(0, numel, chunk_size):
        end = min(start + chunk_size, numel)
        diff = (loaded_flat[start:end].float() - raw_flat[start:end].float()).abs()
        if diff.numel():
            max_abs = max(max_abs, float(diff.max().item()))
            sum_abs += float(diff.sum().item())
            nonzero += int((diff != 0).sum().item())
    return {
        "max_abs": max_abs,
        "mean_abs": 0.0 if numel == 0 else sum_abs / numel,
        "nonzero_abs_count": nonzero,
    }


def _compare_tensor(name: str, loaded: torch.Tensor, raw: torch.Tensor) -> dict[str, Any]:
    loaded_cpu = loaded.detach().cpu().contiguous()
    raw_cpu = raw.detach().cpu().contiguous()
    if loaded_cpu.shape != raw_cpu.shape:
        return {
            "name": name,
            "status": "FAIL",
            "shape_mismatch": True,
            "loaded_shape": list(loaded_cpu.shape),
            "raw_shape": list(raw_cpu.shape),
            "loaded_dtype": str(loaded_cpu.dtype).replace("torch.", ""),
            "raw_dtype": str(raw_cpu.dtype).replace("torch.", ""),
        }
    exact = torch.equal(loaded_cpu, raw_cpu)
    row = {
        "name": name,
        "status": "PASS" if exact else "FAIL",
        "shape": list(loaded_cpu.shape),
        "loaded_dtype": str(loaded_cpu.dtype).replace("torch.", ""),
        "raw_dtype": str(raw_cpu.dtype).replace("torch.", ""),
        "exact_equal": bool(exact),
        "max_abs": 0.0,
        "mean_abs": 0.0,
        "nonzero_abs_count": 0,
    }
    if not exact:
        row.update(_diff_stats(loaded_cpu, raw_cpu))
    return row


def _compare_grouped_experts(name: str, params: dict[str, torch.Tensor], raw: torch.Tensor) -> dict[str, Any]:
    raw_cpu = raw.detach().cpu().contiguous()
    rows = []
    max_abs = 0.0
    sum_abs = 0.0
    nonzero = 0
    total_numel = 0
    exact_all = True
    for expert_id in range(raw_cpu.shape[0]):
        loaded_name = f"{name}.weight{expert_id}"
        if loaded_name not in params:
            rows.append({"expert_id": expert_id, "status": "FAIL", "missing_loaded": True})
            exact_all = False
            continue
        loaded_cpu = params[loaded_name].detach().cpu().contiguous()
        raw_expert = raw_cpu[expert_id]
        if loaded_cpu.shape != raw_expert.shape:
            rows.append(
                {
                    "expert_id": expert_id,
                    "status": "FAIL",
                    "shape_mismatch": True,
                    "loaded_shape": list(loaded_cpu.shape),
                    "raw_shape": list(raw_expert.shape),
                }
            )
            exact_all = False
            continue
        exact = torch.equal(loaded_cpu, raw_expert)
        exact_all = exact_all and exact
        total_numel += loaded_cpu.numel()
        if not exact:
            stats = _diff_stats(loaded_cpu, raw_expert)
            max_abs = max(max_abs, stats["max_abs"])
            sum_abs += stats["mean_abs"] * loaded_cpu.numel()
            nonzero += stats["nonzero_abs_count"]
        rows.append({"expert_id": expert_id, "status": "PASS" if exact else "FAIL", "exact_equal": bool(exact)})
    failed = [row for row in rows if row["status"] != "PASS"]
    return {
        "name": name,
        "status": "PASS" if exact_all else "FAIL",
        "shape": list(raw_cpu.shape),
        "loaded_dtype": str(params[f"{name}.weight0"].dtype).replace("torch.", "") if f"{name}.weight0" in params else None,
        "raw_dtype": str(raw_cpu.dtype).replace("torch.", ""),
        "num_experts": int(raw_cpu.shape[0]),
        "num_checked_experts": len(rows),
        "num_failed_experts": len(failed),
        "first_failed_expert": failed[0] if failed else None,
        "exact_equal": bool(exact_all),
        "max_abs": max_abs,
        "mean_abs": 0.0 if total_numel == 0 else sum_abs / total_numel,
        "nonzero_abs_count": nonzero,
    }


def main() -> int:
    configure_logger()
    _init_distributed()
    args = _parse_args()
    init(args)
    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)
    load_checkpoint(model, None, None, checkpointing_context={}, skip_load_to_model_and_opt=False)

    if dist.get_rank() == 0:
        params = dict(model[0].named_parameters())
        raw_state = _load_raw_state(args.checkpoint_release_dir)
        rows = []
        for loaded_name, raw_name in SELECTED.items():
            if loaded_name not in params:
                rows.append({"name": loaded_name, "status": "FAIL", "missing_loaded": True})
                continue
            if raw_name not in raw_state:
                rows.append({"name": loaded_name, "status": "FAIL", "missing_raw": True, "raw_name": raw_name})
                continue
            rows.append(_compare_tensor(loaded_name, params[loaded_name], raw_state[raw_name]))
        for loaded_name, raw_name in GROUPED_EXPERTS.items():
            if raw_name not in raw_state:
                rows.append({"name": loaded_name, "status": "FAIL", "missing_raw": True, "raw_name": raw_name})
                continue
            rows.append(_compare_grouped_experts(loaded_name, params, raw_state[raw_name]))
        payload = {
            "label": "loaded_miles_weights_vs_raw_megatron_checkpoint",
            "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
            "coverage": {
                "num_selected": len(rows),
                "attention_weights": 8,
                "mlp_weights": 7,
                "top_level_weights": 3,
            },
            "selected": rows,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

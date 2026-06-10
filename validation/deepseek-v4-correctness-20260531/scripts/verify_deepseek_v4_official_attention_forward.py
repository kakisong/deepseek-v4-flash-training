#!/usr/bin/env python3
"""DeepSeek-V4 官方推理 attention 前向一致性校验。

本脚本使用相同的随机初始化权重和相同输入,将 Miles 的 ``DeepSeekV4Attention``
与官方 DeepSeek-V4 推理 ``Attention`` 模块进行对比。这不是完整的 checkpoint
一致性测试,而是基于外部代码的模块级一致性检查,同时覆盖非压缩的
sliding-window attention 路径和 compress_ratio=4 的压缩 KV/indexer 路径。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig

from miles_plugins.models.deepseek_v4.deepseek_v4 import DeepSeekV4Attention


SEED = 20260531


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_distributed() -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    torch.cuda.set_device(0)
    torch.set_default_device("cuda")
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=0,
            world_size=1,
            init_method=f"tcp://127.0.0.1:{_free_port()}",
            device_id=torch.device("cuda:0"),
        )
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        )
    model_parallel_cuda_manual_seed(SEED)


def _destroy_distributed() -> None:
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _load_official_model_module(inference_dir: Path):
    model_py = inference_dir / "model.py"
    kernel_py = inference_dir / "kernel.py"
    if not model_py.exists() or not kernel_py.exists():
        raise FileNotFoundError(f"{inference_dir} must contain model.py and kernel.py")
    import sys

    sys.path.insert(0, str(inference_dir))
    spec = importlib.util.spec_from_file_location("deepseek_v4_official_inference_model", model_py)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _init_method(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.02)


def _build_miles_config(compress_ratio: int) -> TransformerConfig:
    config = TransformerConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        sequence_parallel=False,
        perform_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=64,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        init_method=_init_method,
        output_layer_init_method=_init_method,
        dsv4_compress_ratios=[compress_ratio],
        dsv4_compress_rope_theta=160000,
        dsv4_o_groups=8,
        dsv4_o_lora_rank=1024,
        dsv4_window_size=128,
        dsa_indexer_n_heads=64,
        dsa_indexer_head_dim=128,
        dsa_indexer_topk=512,
    )
    for key, value in {
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_pos_emb_head_dim": 64,
        "rotary_base": 10000,
        "rotary_scaling_factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    }.items():
        setattr(config, key, value)
    return config


def _prepare_state_for_miles(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state = dict(state)
    if "indexer.wq_b.weight" in state:
        state["indexer.linear_wq_b.weight"] = state.pop("indexer.wq_b.weight")
    if "indexer.weights_proj.weight" in state:
        state["indexer.linear_weights_proj.weight"] = state.pop("indexer.weights_proj.weight")
    return state


def _tensor_rel_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = (xf.square().sum() + yf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (xf * yf).sum().item() / denom)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-inference-dir", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--seqlen", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compress-ratios", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--max-rel-gap", type=float, default=1e-3)
    parser.add_argument("--max-abs", type=float, default=5e-3)
    args = parser.parse_args()

    _init_distributed()
    try:
        torch.set_default_dtype(torch.bfloat16)
        official_model = _load_official_model_module(args.official_inference_dir)
        official_model.world_size = 1
        official_model.rank = 0
        official_model.default_dtype = torch.bfloat16
        official_model.scale_fmt = None
        official_model.scale_dtype = torch.float32

        cases: list[dict[str, Any]] = []
        failures: list[str] = []

        for compress_ratio in args.compress_ratios:
            if compress_ratio not in (0, 4):
                raise ValueError(f"unsupported compress_ratio for this verifier: {compress_ratio}")
            torch.manual_seed(SEED + compress_ratio)
            torch.cuda.manual_seed_all(SEED + compress_ratio)
            official_args = official_model.ModelArgs(
                dtype="bf16",
                scale_dtype="fp32",
                n_layers=1,
                n_mtp_layers=0,
                n_hash_layers=0,
                n_heads=64,
                q_lora_rank=1024,
                compress_ratios=(compress_ratio,),
                max_seq_len=128,
                max_batch_size=args.batch_size,
                rope_factor=16,
                original_seq_len=65536,
                compress_rope_theta=160000,
                index_topk=512,
            )

            official = official_model.Attention(0, official_args)
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
            miles = DeepSeekV4Attention(
                _build_miles_config(compress_ratio),
                layer_number=1,
                pg_collection=pg_collection,
            ).cuda()

            for _, param in official.named_parameters():
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            missing, unexpected = miles.load_state_dict(_prepare_state_for_miles(official.state_dict()), strict=False)
            allowed_missing = {
                "wq_a._extra_state",
                "q_norm._extra_state",
                "wkv._extra_state",
                "kv_norm._extra_state",
                "indexer.linear_wq_b._extra_state",
                "indexer.linear_weights_proj._extra_state",
            }
            assert set(missing).issubset(allowed_missing), f"unexpected missing keys: {missing}"
            assert not unexpected, f"unexpected keys: {unexpected}"

            hidden_states = torch.randn(
                args.batch_size,
                args.seqlen,
                4096,
                device="cuda",
                dtype=torch.bfloat16,
            )
            with torch.no_grad():
                official_out = official(hidden_states.clone(), 0)
            _assert_finite(f"official.compress_{compress_ratio}.output", official_out)

            comparisons: list[dict[str, Any]] = []
            for impl in ["dense", "sparse", "tilelang"]:
                os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = impl
                with torch.no_grad():
                    miles_out = miles(hidden_states.clone().transpose(0, 1), None).transpose(0, 1)
                _assert_finite(f"miles.compress_{compress_ratio}.{impl}.output", miles_out)
                max_abs = float((official_out.float() - miles_out.float()).abs().max().item())
                rel_gap = _tensor_rel_gap(official_out, miles_out)
                item = {
                    "label": f"official_vs_miles_{impl}",
                    "max_abs": max_abs,
                    "relative_gap": rel_gap,
                }
                comparisons.append(item)
                if max_abs > args.max_abs:
                    failures.append(f"compress_{compress_ratio}.{item['label']}.max_abs")
                if rel_gap > args.max_rel_gap:
                    failures.append(f"compress_{compress_ratio}.{item['label']}.relative_gap")
            cases.append(
                {
                    "name": f"official_attention_compress_{compress_ratio}",
                    "seqlen": args.seqlen,
                    "batch_size": args.batch_size,
                    "hidden_size": 4096,
                    "heads": 64,
                    "head_dim": 512,
                    "rope_head_dim": 64,
                    "normalization": "RMSNorm",
                    "compress_ratio": compress_ratio,
                    "kv_qat": True,
                    "state_dict_load": {"missing": missing, "unexpected": unexpected},
                    "comparisons": comparisons,
                }
            )

        max_abs = max(
            item["max_abs"]
            for case in cases
            for item in case["comparisons"]
        )
        max_rel_gap = max(
            item["relative_gap"]
            for case in cases
            for item in case["comparisons"]
        )

        payload = {
            "seed": SEED,
            "status": "PASS" if not failures else "FAIL",
            "reference": "DeepSeek-V4 official inference Attention",
            "thresholds": {"max_rel_gap": args.max_rel_gap, "max_abs": args.max_abs},
            "max_abs": max_abs,
            "max_relative_gap": max_rel_gap,
            "cases": cases,
            "failures": failures,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"wrote {args.json_output}")
        return 0 if not failures else 1
    finally:
        _destroy_distributed()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    os.environ.setdefault("MEGATRON_USE_KV_QAT", "1")
    raise SystemExit(main())

#!/usr/bin/env python3
"""DeepSeek-V4 attention module training-step parity.

This is a module-level gate between the pure PyTorch dense/sparse paths and the
TileLang sparse path. It builds three identical DeepSeekV4Attention modules,
runs one deterministic forward/backward/update step for each implementation,
and compares:

* forward output
* loss
* input gradient
* parameter gradients (finite check)
* post-SGD parameter state

The module uses production DeepSeek-V4 attention dimensions (H=8 local heads,
D=512, RoPE=64, O-LoRA=1024, hidden=4096), but disables compressed KV so the
check isolates the dense/sparse/tilelang attention implementation swap.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import socket
from dataclasses import asdict, dataclass
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


@dataclass
class ImplResult:
    impl: str
    loss: float
    output: torch.Tensor
    input_grad: torch.Tensor
    state_after_step: dict[str, torch.Tensor]
    max_param_grad_norm: float
    num_params_with_grad: int
    params_without_grad: list[str]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _init_distributed() -> None:
    assert torch.cuda.is_available(), "CUDA is required for TileLang/Megatron module parity"
    torch.cuda.set_device(0)
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


def _init_method(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.02)


def _build_config(compress_ratio: int) -> TransformerConfig:
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
        num_attention_heads=8,
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
    # These DeepSeek-V4 fields are carried by the runtime HF/mbridge config but
    # are not dataclass fields in Megatron's generic TransformerConfig.
    for key, value in {
        "q_lora_rank": 1024,
        "kv_lora_rank": 512,
        "qk_pos_emb_head_dim": 64,
        "rotary_base": 10000,
        "rotary_scaling_factor": 4,
        "beta_fast": 32,
        "beta_slow": 1,
        "original_max_position_embeddings": 65536,
    }.items():
        setattr(config, key, value)
    return config


def _tensor_rel_gap(x: torch.Tensor, y: torch.Tensor) -> float:
    xf = x.detach().flatten().float()
    yf = y.detach().flatten().float()
    denom = (xf.square().sum() + yf.square().sum()).item()
    if denom == 0.0:
        return 0.0
    return float(1.0 - 2.0 * (xf * yf).sum().item() / denom)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"


def _run_impl(
    impl: str,
    compress_ratio: int,
    base_state: dict[str, Any],
    hidden_states: torch.Tensor,
    upstream_grad: torch.Tensor,
    *,
    lr: float,
) -> ImplResult:
    os.environ["MEGATRON_SPARSE_ATTN_IMPL"] = impl
    pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    module = DeepSeekV4Attention(
        _build_config(compress_ratio),
        layer_number=1,
        pg_collection=pg_collection,
    ).cuda()
    module.load_state_dict(base_state)
    module.train()

    x = hidden_states.detach().clone().requires_grad_(True)
    output = module(x)
    _assert_finite(f"{impl}.output", output)
    loss = (output.float() * upstream_grad.float()).mean()
    loss.backward()

    assert x.grad is not None
    _assert_finite(f"{impl}.input_grad", x.grad)
    max_grad_norm = 0.0
    num_params_with_grad = 0
    params_without_grad: list[str] = []
    for name, param in module.named_parameters():
        if param.grad is None:
            params_without_grad.append(name)
            continue
        _assert_finite(f"{impl}.{name}.grad", param.grad)
        max_grad_norm = max(max_grad_norm, float(param.grad.detach().float().norm().item()))
        num_params_with_grad += 1
        param.data.add_(param.grad.to(param.dtype), alpha=-lr)

    state_after_step = {
        name: tensor.detach().cpu().float()
        for name, tensor in module.state_dict().items()
        if torch.is_tensor(tensor) and tensor.numel() > 0
    }
    return ImplResult(
        impl=impl,
        loss=float(loss.item()),
        output=output.detach().cpu(),
        input_grad=x.grad.detach().cpu(),
        state_after_step=state_after_step,
        max_param_grad_norm=max_grad_norm,
        num_params_with_grad=num_params_with_grad,
        params_without_grad=params_without_grad,
    )


def _compare(a: ImplResult, b: ImplResult) -> dict[str, Any]:
    state_max_abs = 0.0
    state_max_abs_name = ""
    state_max_rel = 0.0
    for name, tensor_a in a.state_after_step.items():
        tensor_b = b.state_after_step[name]
        max_abs = float((tensor_a - tensor_b).abs().max().item())
        rel = _tensor_rel_gap(tensor_a, tensor_b)
        if max_abs > state_max_abs:
            state_max_abs = max_abs
            state_max_abs_name = name
        state_max_rel = max(state_max_rel, rel)

    return {
        "label": f"{a.impl}_vs_{b.impl}",
        "loss_abs": abs(a.loss - b.loss),
        "output_max_abs": float((a.output.float() - b.output.float()).abs().max().item()),
        "output_rel_gap": _tensor_rel_gap(a.output, b.output),
        "input_grad_max_abs": float((a.input_grad.float() - b.input_grad.float()).abs().max().item()),
        "input_grad_rel_gap": _tensor_rel_gap(a.input_grad, b.input_grad),
        "state_after_step_max_abs": state_max_abs,
        "state_after_step_max_abs_name": state_max_abs_name,
        "state_after_step_max_rel_gap": state_max_rel,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--seqlen", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compress-ratios", type=int, nargs="+", default=[0, 4])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--skip-tilelang", action="store_true")
    parser.add_argument("--max-rel-gap", type=float, default=5e-5)
    parser.add_argument("--max-output-abs", type=float, default=4e-2)
    parser.add_argument("--max-input-grad-abs", type=float, default=2e-6)
    parser.add_argument("--max-state-abs", type=float, default=1e-7)
    args = parser.parse_args()

    _init_distributed()
    try:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        impls = ["dense", "sparse"] if args.skip_tilelang else ["dense", "sparse", "tilelang"]
        cases: list[dict[str, Any]] = []
        failures: list[str] = []

        for compress_ratio in args.compress_ratios:
            if compress_ratio not in (0, 4, 128):
                raise ValueError(f"unsupported compress_ratio for this verifier: {compress_ratio}")
            if compress_ratio and args.seqlen % compress_ratio != 0:
                raise ValueError(f"--seqlen must be divisible by compress_ratio={compress_ratio}")
            os.environ["V4_INDEXER_IMPL"] = "tilelang"
            torch.manual_seed(SEED + compress_ratio)
            torch.cuda.manual_seed_all(SEED + compress_ratio)
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
            base_module = DeepSeekV4Attention(
                _build_config(compress_ratio),
                layer_number=1,
                pg_collection=pg_collection,
            ).cuda()
            base_state = copy.deepcopy(base_module.state_dict())
            hidden_states = torch.randn(
                args.seqlen,
                args.batch_size,
                4096,
                device="cuda",
                dtype=torch.bfloat16,
            )
            upstream_grad = torch.randn_like(hidden_states)
            results = {
                impl: _run_impl(impl, compress_ratio, base_state, hidden_states, upstream_grad, lr=args.lr)
                for impl in impls
            }
            comparisons = []
            for left, right in [("dense", "sparse"), ("dense", "tilelang"), ("sparse", "tilelang")]:
                if left in results and right in results:
                    comparisons.append(_compare(results[left], results[right]))

            allowed_without_grad = set()
            if compress_ratio == 4:
                allowed_without_grad = {
                    "indexer.linear_wq_b.weight",
                    "indexer.linear_weights_proj.weight",
                    "indexer.compressor.ape",
                    "indexer.compressor.wkv.weight",
                    "indexer.compressor.wgate.weight",
                    "indexer.compressor.norm.weight",
                }
            for impl, result in results.items():
                if result.num_params_with_grad <= 0:
                    failures.append(f"compress_{compress_ratio}.{impl}: no parameter gradients")
                unexpected_without_grad = set(result.params_without_grad) - allowed_without_grad
                missing_without_grad = allowed_without_grad - set(result.params_without_grad)
                if unexpected_without_grad:
                    failures.append(
                        f"compress_{compress_ratio}.{impl}.unexpected_params_without_grad={sorted(unexpected_without_grad)}"
                    )
                if missing_without_grad:
                    failures.append(
                        f"compress_{compress_ratio}.{impl}.expected_indexer_no_grad_missing={sorted(missing_without_grad)}"
                    )
            for item in comparisons:
                checks = {
                    "output_rel_gap": item["output_rel_gap"] <= args.max_rel_gap,
                    "input_grad_rel_gap": item["input_grad_rel_gap"] <= args.max_rel_gap,
                    "state_after_step_max_rel_gap": item["state_after_step_max_rel_gap"] <= args.max_rel_gap,
                    "output_max_abs": item["output_max_abs"] <= args.max_output_abs,
                    "input_grad_max_abs": item["input_grad_max_abs"] <= args.max_input_grad_abs,
                    "state_after_step_max_abs": item["state_after_step_max_abs"] <= args.max_state_abs,
                }
                for check_name, passed in checks.items():
                    if not passed:
                        failures.append(f"compress_{compress_ratio}.{item['label']}.{check_name}")

            cases.append(
                {
                    "name": f"DeepSeekV4Attention_compress_{compress_ratio}",
                    "seqlen": args.seqlen,
                    "batch_size": args.batch_size,
                    "hidden_size": 4096,
                    "local_heads": 8,
                    "head_dim": 512,
                    "rope_head_dim": 64,
                    "normalization": "RMSNorm",
                    "dsv4_o_lora_rank": 1024,
                    "compress_ratio": compress_ratio,
                    "lr": args.lr,
                    "impls": {
                        impl: {
                            "loss": result.loss,
                            "max_param_grad_norm": result.max_param_grad_norm,
                            "num_params_with_grad": result.num_params_with_grad,
                            "params_without_grad": result.params_without_grad,
                        }
                        for impl, result in results.items()
                    },
                    "comparisons": comparisons,
                }
            )

        payload = {
            "seed": SEED,
            "status": "PASS" if not failures else "FAIL",
            "thresholds": {
                "max_rel_gap": args.max_rel_gap,
                "max_output_abs": args.max_output_abs,
                "max_input_grad_abs": args.max_input_grad_abs,
                "max_state_abs": args.max_state_abs,
            },
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
    raise SystemExit(main())

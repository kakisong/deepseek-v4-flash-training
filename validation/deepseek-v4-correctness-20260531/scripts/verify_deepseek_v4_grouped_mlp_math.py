#!/usr/bin/env python3
"""Verify DeepSeek-V4 routed expert grouped MLP math.

This is a focused training-path verifier for the Megatron/TE grouped expert
kernel used by the Miles DeepSeek-V4 MoE path.  It compares TEGroupedMLP
forward/backward/update against an explicit per-expert BF16 SwiGLU reference
with the DeepSeek-V4 0415 activation clamp.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.extensions.transformer_engine import TEGroupedLinear


SEED = 20260531


def _init_dist() -> None:
    if dist.is_initialized():
        return
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        create_gloo_process_groups=False,
    )


def _destroy_dist() -> None:
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _build_config(hidden_size: int, ffn_hidden_size: int, num_experts: int, clamp: float) -> TransformerConfig:
    def init_method(tensor: torch.Tensor) -> None:
        torch.nn.init.normal_(tensor, mean=0.0, std=0.03)

    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        ffn_hidden_size=ffn_hidden_size,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=ffn_hidden_size,
        moe_router_topk=6,
        moe_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=False,
        moe_token_dispatcher_type="alltoall",
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_clamp_value=clamp,
        activation_func_clamp_shared_expert=False,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        perform_initialization=True,
        init_method=init_method,
        output_layer_init_method=init_method,
        sequence_parallel=False,
        gradient_accumulation_fusion=False,
        use_te_activation_func=False,
    )


def _pg_collection() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "ep", "expt_tp", "expt_dp"])


def _weight_list(grouped_linear: TEGroupedLinear, num_experts: int) -> list[torch.Tensor]:
    return [getattr(grouped_linear, f"weight{i}") for i in range(num_experts)]


def _manual_grouped_mlp(
    hidden: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    probs: torch.Tensor,
    fc1_weights: list[torch.Tensor],
    fc2_weights: list[torch.Tensor],
    *,
    clamp: float,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    offset = 0
    for expert_id, n_tokens in enumerate(tokens_per_expert.tolist()):
        n_tokens = int(n_tokens)
        if n_tokens == 0:
            continue
        x = hidden[offset : offset + n_tokens]
        p = probs[offset : offset + n_tokens].unsqueeze(-1)
        fc1 = F.linear(x, fc1_weights[expert_id])
        gate, up = fc1.chunk(2, dim=-1)
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
        intermediate = F.silu(gate) * up
        intermediate = intermediate * p.to(intermediate.dtype)
        intermediate = intermediate.to(intermediate.dtype)
        pieces.append(F.linear(intermediate, fc2_weights[expert_id]))
        offset += n_tokens
    if not pieces:
        return torch.empty(0, fc2_weights[0].shape[0], dtype=hidden.dtype, device=hidden.device)
    return torch.cat(pieces, dim=0)


def _compare(name: str, left: torch.Tensor, right: torch.Tensor, rtol: float, atol: float) -> dict[str, Any]:
    left_f = left.detach().float().flatten()
    right_f = right.detach().float().flatten()
    diff = (left_f - right_f).abs()
    close = torch.isclose(left_f, right_f, rtol=rtol, atol=atol)
    denom = float((left_f.square().sum() + right_f.square().sum()).item())
    rel_gap = 0.0 if denom == 0.0 else float(1.0 - 2.0 * (left_f * right_f).sum().item() / denom)
    return {
        "name": name,
        "status": "PASS" if bool(close.all().item()) else "FAIL",
        "shape": list(left.shape),
        "numel": int(left_f.numel()),
        "mismatches": int((~close).sum().item()),
        "nonzero_abs_count": int((diff != 0).sum().item()),
        "exact_equal": bool((diff == 0).all().item()),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "p95_abs": float(diff.quantile(0.95).item()) if diff.numel() else 0.0,
        "p99_abs": float(diff.quantile(0.99).item()) if diff.numel() else 0.0,
        "relative_l2_gap": rel_gap,
        "rtol": rtol,
        "atol": atol,
    }


def _sgd_step(params: list[torch.Tensor], lr: float) -> list[torch.Tensor]:
    updated = []
    with torch.no_grad():
        for param in params:
            grad = torch.zeros_like(param) if param.grad is None else param.grad
            updated.append((param - lr * grad).detach().clone())
    return updated


def _grad_or_zero(param: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(param) if param.grad is None else param.grad


def run_case(
    *,
    name: str,
    tokens_per_expert: torch.Tensor,
    hidden_size: int,
    ffn_hidden_size: int,
    clamp: float,
    rtol: float,
    atol: float,
    lr: float,
) -> dict[str, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    num_experts = int(tokens_per_expert.numel())
    config = _build_config(hidden_size, ffn_hidden_size, num_experts, clamp)
    expert_module, expert_submodules = TESpecProvider().grouped_mlp_modules(
        moe_use_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=False,
    )
    assert expert_module is TEGroupedMLP
    module = TEGroupedMLP(
        num_local_experts=num_experts,
        config=config,
        submodules=expert_submodules,
        pg_collection=_pg_collection(),
    ).to(device)
    module.train()

    total_tokens = int(tokens_per_expert.sum().item())
    torch.manual_seed(SEED + total_tokens + num_experts)
    hidden = torch.randn(total_tokens, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(total_tokens, device=device, dtype=torch.float32).mul_(1.25).add_(0.05).to(torch.bfloat16)
    target = torch.randn(total_tokens, hidden_size, device=device, dtype=torch.bfloat16)

    fc1_params = _weight_list(module.linear_fc1, num_experts)
    fc2_params = _weight_list(module.linear_fc2, num_experts)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_fc1 = [w.detach().clone().requires_grad_(True) for w in fc1_params]
    ref_fc2 = [w.detach().clone().requires_grad_(True) for w in fc2_params]

    te_output, te_bias = module(hidden, tokens_per_expert.to(device=device), probs)
    assert te_bias is None
    ref_output = _manual_grouped_mlp(
        ref_hidden,
        tokens_per_expert.to(device=device),
        probs.detach(),
        ref_fc1,
        ref_fc2,
        clamp=clamp,
    )

    loss_weight = target.float() / max(1, total_tokens * hidden_size)
    te_loss = (te_output.float() * loss_weight).sum()
    ref_loss = (ref_output.float() * loss_weight).sum()
    te_loss.backward()
    ref_loss.backward()

    te_params = fc1_params + fc2_params
    ref_params = ref_fc1 + ref_fc2
    te_updated = _sgd_step(te_params, lr)
    ref_updated = _sgd_step(ref_params, lr)

    comparisons = [
        _compare("forward_output", te_output, ref_output, rtol, atol),
        _compare("input_grad", hidden.grad, ref_hidden.grad, rtol, atol),
    ]
    for expert_id, (left, right) in enumerate(zip(fc1_params, ref_fc1, strict=True)):
        comparisons.append(
            _compare(
                f"fc1_weight_grad_expert_{expert_id}",
                _grad_or_zero(left),
                _grad_or_zero(right),
                rtol,
                atol,
            )
        )
    for expert_id, (left, right) in enumerate(zip(fc2_params, ref_fc2, strict=True)):
        comparisons.append(
            _compare(
                f"fc2_weight_grad_expert_{expert_id}",
                _grad_or_zero(left),
                _grad_or_zero(right),
                rtol,
                atol,
            )
        )
    for expert_id, (left, right) in enumerate(zip(te_updated[:num_experts], ref_updated[:num_experts], strict=True)):
        comparisons.append(_compare(f"fc1_sgd_update_expert_{expert_id}", left, right, rtol, atol))
    for expert_id, (left, right) in enumerate(zip(te_updated[num_experts:], ref_updated[num_experts:], strict=True)):
        comparisons.append(_compare(f"fc2_sgd_update_expert_{expert_id}", left, right, rtol, atol))

    return {
        "name": name,
        "status": "PASS" if all(row["status"] == "PASS" for row in comparisons) else "FAIL",
        "tokens_per_expert": [int(x) for x in tokens_per_expert.tolist()],
        "total_tokens": total_tokens,
        "hidden_size": hidden_size,
        "moe_ffn_hidden_size": ffn_hidden_size,
        "num_experts": num_experts,
        "activation": "SwiGLU",
        "activation_func_clamp_value": clamp,
        "probability_application": "after_activation_before_fc2",
        "expert_module": type(module).__name__,
        "linear_module": type(module.linear_fc1).__name__,
        "loss": {
            "te": float(te_loss.detach().cpu().item()),
            "reference": float(ref_loss.detach().cpu().item()),
            "abs_diff": float(abs(te_loss.detach().cpu().item() - ref_loss.detach().cpu().item())),
        },
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ffn-hidden-size", type=int, default=48)
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    _init_dist()
    try:
        cases = [
            run_case(
                name="imbalanced_with_empty_experts",
                tokens_per_expert=torch.tensor([0, 1, 7, 3, 9], dtype=torch.int64),
                hidden_size=args.hidden_size,
                ffn_hidden_size=args.ffn_hidden_size,
                clamp=args.activation_clamp,
                rtol=args.rtol,
                atol=args.atol,
                lr=args.lr,
            ),
            run_case(
                name="balanced_all_experts_active",
                tokens_per_expert=torch.tensor([4, 4, 4, 4, 4], dtype=torch.int64),
                hidden_size=args.hidden_size,
                ffn_hidden_size=args.ffn_hidden_size,
                clamp=args.activation_clamp,
                rtol=args.rtol,
                atol=args.atol,
                lr=args.lr,
            ),
        ]
        payload = {
            "status": "PASS" if all(case["status"] == "PASS" for case in cases) else "FAIL",
            "seed": SEED,
            "dtype": "bfloat16",
            "runtime": {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "transformer_engine_grouped_linear": "TEGroupedLinear",
            },
            "threshold": {"rtol": args.rtol, "atol": args.atol},
            "cases": cases,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return 0 if payload["status"] == "PASS" else 1
    finally:
        _destroy_dist()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

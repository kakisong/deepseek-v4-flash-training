#!/usr/bin/env python3
"""Verify DeepSeek-V4 EP=8 real MoE layer against an external reference.

This verifier closes the gap between the focused EP=8 dispatcher math test and
the one-rank score-routed MoE block reference.  It runs Megatron's real
``MoELayer`` on 8 ranks with one expert per rank, real all-to-all dispatch,
TE grouped expert GEMMs, shared experts, sqrtsoftplus top-k routing, expert
bias, backward, and a manual SGD update.  The comparison target is an explicit
PyTorch formula built from tensors gathered across all ranks.
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
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig


SEED = 20260531


def _init_dist(expert_parallel_size: int) -> None:
    assert torch.cuda.is_available(), "CUDA is required"
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert dist.get_world_size() == expert_parallel_size, (
        f"this verifier expects world_size={expert_parallel_size}, got {dist.get_world_size()}"
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=1,
        create_gloo_process_groups=False,
    )
    model_parallel_cuda_manual_seed(SEED)


def _destroy_dist() -> None:
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _init_method(tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.init.normal_(tensor, mean=0.0, std=0.02)


def _build_config(expert_parallel_size: int) -> TransformerConfig:
    config = TransformerConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        perform_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        num_layers=1,
        hidden_size=4096,
        num_attention_heads=8,
        ffn_hidden_size=1024,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        init_method=_init_method,
        output_layer_init_method=_init_method,
        num_moe_experts=expert_parallel_size,
        moe_ffn_hidden_size=2048,
        moe_router_topk=6,
        moe_router_pre_softmax=True,
        moe_router_score_function="sqrtsoftplus",
        moe_router_topk_scaling_factor=1.5,
        moe_router_enable_expert_bias=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=False,
        moe_shared_expert_intermediate_size=2048,
        moe_shared_expert_overlap=False,
        moe_shared_expert_gate=False,
        moe_permute_fusion=False,
        moe_expert_capacity_factor=None,
        moe_router_padding_for_quantization=False,
        moe_pad_expert_input_to_capacity=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_clamp_value=10.0,
        activation_func_clamp_shared_expert=False,
        bias_activation_fusion=False,
        use_te_activation_func=False,
        gradient_accumulation_fusion=False,
    )
    config.dsv4_mode = True
    config.dsv4_n_hash_layers = 0
    return config


def _pg_collection() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups(
        required_pgs=["tp", "cp", "tp_cp", "tp_dp_cp", "pp", "ep", "expt_tp", "expt_dp", "tp_ep"]
    )


def _build_moe_layer(config: TransformerConfig):
    spec = get_moe_module_spec_for_backend(
        backend=TESpecProvider(),
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=config.moe_use_legacy_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )
    layer = spec.module(
        config=config,
        submodules=spec.submodules,
        layer_number=1,
        pg_collection=_pg_collection(),
    ).cuda()
    layer.train()
    return layer


def _normal_(tensor: torch.Tensor, seed: int, std: float = 0.02) -> None:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(seed)
    values = torch.randn(tensor.shape, generator=generator, device=tensor.device, dtype=torch.float32)
    tensor.copy_((values * std).to(tensor.dtype))


def _initialize_state(layer) -> None:
    rank = dist.get_rank()
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if name == "router.weight" or name.startswith("shared_experts."):
                if rank == 0:
                    _normal_(param, SEED + 10 + len(name), std=0.02)
                dist.broadcast(param.data, src=0)
            elif name.startswith("experts.linear_fc1.weight"):
                _normal_(param, SEED + 1000 + rank, std=0.02)
            elif name.startswith("experts.linear_fc2.weight"):
                _normal_(param, SEED + 2000 + rank, std=0.02)
            else:
                if rank == 0:
                    _normal_(param, SEED + 3000 + len(name), std=0.02)
                dist.broadcast(param.data, src=0)
        for name, buffer in layer.named_buffers():
            if name == "router.expert_bias":
                values = torch.linspace(-0.03, 0.04, buffer.numel(), device=buffer.device, dtype=torch.float32)
                buffer.copy_(values.view_as(buffer))


def _get_param(layer, name: str) -> torch.Tensor:
    params = dict(layer.named_parameters())
    if name not in params:
        raise KeyError(f"parameter not found: {name}; available={sorted(params)}")
    return params[name]


def _get_buffer(layer, name: str) -> torch.Tensor:
    buffers = dict(layer.named_buffers())
    if name not in buffers:
        raise KeyError(f"buffer not found: {name}; available={sorted(buffers)}")
    return buffers[name]


def _gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor.contiguous()) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.stack(gathered, dim=0)


def _swiglu(x: torch.Tensor, clamp: float | None) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    if clamp is not None:
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    return F.silu(gate) * up


def _moe_topk(
    x_flat: torch.Tensor,
    router_weight: torch.Tensor,
    expert_bias: torch.Tensor,
    *,
    topk: int,
    scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = F.linear(x_flat, router_weight)
    scores = F.softplus(logits.float()).sqrt().to(logits.dtype)
    _, top_indices = torch.topk(scores + expert_bias.to(scores.dtype), k=topk, dim=1)
    top_scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
    top_probs = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)
    top_probs = top_probs * scaling_factor
    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, top_probs)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
    return routing_probs, routing_map, top_indices


def _reference_moe(
    hidden_by_rank: torch.Tensor,
    expert_fc1: list[torch.Tensor],
    expert_fc2: list[torch.Tensor],
    router_weight: torch.Tensor,
    expert_bias: torch.Tensor,
    shared_fc1: torch.Tensor,
    shared_fc2: torch.Tensor,
    *,
    topk: int,
    scaling_factor: float,
    clamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    world_size, local_tokens, batch_size, hidden_size = hidden_by_rank.shape
    x_flat = hidden_by_rank.contiguous().view(world_size * local_tokens * batch_size, hidden_size)
    routing_probs, routing_map, _ = _moe_topk(
        x_flat,
        router_weight,
        expert_bias,
        topk=topk,
        scaling_factor=scaling_factor,
    )

    routed = torch.zeros_like(x_flat)
    for expert_id in range(world_size):
        selected = routing_map[:, expert_id]
        if not bool(selected.any().item()):
            continue
        expert_input = x_flat[selected]
        probs = routing_probs[selected, expert_id].unsqueeze(-1)
        fc1 = F.linear(expert_input, expert_fc1[expert_id])
        intermediate = _swiglu(fc1, clamp=clamp)
        intermediate = intermediate * probs.to(intermediate.dtype)
        routed[selected] = routed[selected] + F.linear(intermediate, expert_fc2[expert_id])

    shared = F.linear(x_flat, shared_fc1)
    shared = _swiglu(shared, clamp=None)
    shared = F.linear(shared, shared_fc2)
    output = (routed + shared).view(world_size, local_tokens, batch_size, hidden_size)
    return output, routing_map


def _compare(name: str, left: torch.Tensor, right: torch.Tensor, threshold: float) -> dict[str, Any]:
    left_f = left.detach().float().flatten()
    right_f = right.detach().float().flatten()
    diff = (left_f - right_f).abs()
    left_abs = left_f.abs()
    right_abs = right_f.abs()
    denom = float((left_f.square().sum() + right_f.square().sum()).item())
    rel_gap = 0.0 if denom == 0.0 else float(1.0 - 2.0 * (left_f * right_f).sum().item() / denom)
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    return {
        "name": name,
        "status": "PASS" if max_abs <= threshold else "FAIL",
        "shape": list(left.shape),
        "numel": int(left_f.numel()),
        "nonzero_abs_count": int((diff != 0).sum().item()),
        "left_nonzero_abs_count": int((left_abs != 0).sum().item()),
        "right_nonzero_abs_count": int((right_abs != 0).sum().item()),
        "left_max_abs": float(left_abs.max().item()) if left_abs.numel() else 0.0,
        "right_max_abs": float(right_abs.max().item()) if right_abs.numel() else 0.0,
        "max_abs": max_abs,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "p95_abs": float(diff.quantile(0.95).item()) if diff.numel() else 0.0,
        "p99_abs": float(diff.quantile(0.99).item()) if diff.numel() else 0.0,
        "relative_l2_gap": rel_gap,
        "threshold": threshold,
    }


def _grad_or_zero(param: torch.Tensor) -> torch.Tensor:
    if param.grad is None:
        return torch.zeros_like(param)
    return param.grad


def _manual_sgd(param: torch.Tensor, lr: float) -> torch.Tensor:
    return (param.detach() - lr * _grad_or_zero(param).detach()).detach()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    config = _build_config(world_size)
    layer = _build_moe_layer(config)
    _initialize_state(layer)

    torch.manual_seed(SEED + 4000 + rank)
    hidden = torch.randn(
        args.local_tokens,
        args.batch_size,
        config.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    torch.manual_seed(SEED + 5000 + rank)
    upstream = torch.randn_like(hidden)

    local_fc1 = _get_param(layer, "experts.linear_fc1.weight0")
    local_fc2 = _get_param(layer, "experts.linear_fc2.weight0")
    router_weight = _get_param(layer, "router.weight").detach()
    expert_bias = _get_buffer(layer, "router.expert_bias").detach()
    shared_fc1 = _get_param(layer, "shared_experts.linear_fc1.weight").detach()
    shared_fc2 = _get_param(layer, "shared_experts.linear_fc2.weight").detach()

    hidden_all = _gather_tensor(hidden.detach())
    upstream_all = _gather_tensor(upstream.detach())
    fc1_all = _gather_tensor(local_fc1.detach())
    fc2_all = _gather_tensor(local_fc2.detach())

    ref_hidden = hidden_all.clone().requires_grad_(True)
    ref_fc1 = [fc1_all[i].clone().requires_grad_(True) for i in range(world_size)]
    ref_fc2 = [fc2_all[i].clone().requires_grad_(True) for i in range(world_size)]
    ref_router_weight = router_weight.clone().requires_grad_(True)
    ref_shared_fc1 = shared_fc1.clone().requires_grad_(True)
    ref_shared_fc2 = shared_fc2.clone().requires_grad_(True)

    output, bias = layer(hidden)
    assert bias is None
    loss = (output.float() * upstream.float()).mean()
    loss.backward()

    ref_output, routing_map = _reference_moe(
        ref_hidden,
        ref_fc1,
        ref_fc2,
        ref_router_weight,
        expert_bias,
        ref_shared_fc1,
        ref_shared_fc2,
        topk=config.moe_router_topk,
        scaling_factor=config.moe_router_topk_scaling_factor,
        clamp=float(config.activation_func_clamp_value),
    )
    ref_local_losses = [
        (ref_output[src_rank].float() * upstream_all[src_rank].float()).mean()
        for src_rank in range(world_size)
    ]
    ref_loss = sum(ref_local_losses)
    ref_loss.backward()

    actual_fc1_updated = _manual_sgd(local_fc1, args.lr)
    actual_fc2_updated = _manual_sgd(local_fc2, args.lr)
    ref_fc1_updated = (ref_fc1[rank].detach() - args.lr * ref_fc1[rank].grad.detach()).detach()
    ref_fc2_updated = (ref_fc2[rank].detach() - args.lr * ref_fc2[rank].grad.detach()).detach()
    local_expert_grad_nonzero_count = int((_grad_or_zero(local_fc1).detach().float().abs() != 0).sum().item())
    local_expert_grad_nonzero_count += int((_grad_or_zero(local_fc2).detach().float().abs() != 0).sum().item())

    comparisons = [
        _compare("forward_output", output, ref_output[rank], args.max_output_abs),
        _compare("loss", loss.detach().view(1), ref_local_losses[rank].detach().view(1), args.max_loss_abs),
        _compare("input_grad", hidden.grad, ref_hidden.grad[rank], args.max_input_grad_abs),
        _compare("local_expert_fc1_grad", _grad_or_zero(local_fc1), ref_fc1[rank].grad, args.max_expert_grad_abs),
        _compare("local_expert_fc2_grad", _grad_or_zero(local_fc2), ref_fc2[rank].grad, args.max_expert_grad_abs),
        _compare("local_expert_fc1_sgd_update", actual_fc1_updated, ref_fc1_updated, args.max_state_abs),
        _compare("local_expert_fc2_sgd_update", actual_fc2_updated, ref_fc2_updated, args.max_state_abs),
    ]

    checks = {
        "local_expert_grad_nonzero": local_expert_grad_nonzero_count > 0,
    }
    local_row = {
        "rank": rank,
        "status": "PASS" if all(item["status"] == "PASS" for item in comparisons) and all(checks.values()) else "FAIL",
        "local_expert_id": rank,
        "local_tokens": args.local_tokens,
        "batch_size": args.batch_size,
        "routing_tokens_selected_for_local_expert": int(routing_map[:, rank].sum().item()),
        "routing_tokens_selected_global": int(routing_map.sum().item()),
        "local_expert_grad_nonzero_count": local_expert_grad_nonzero_count,
        "checks": checks,
        "comparisons": comparisons,
        "skipped_gradient_comparisons": {
            "router_weight": (
                "Router weights are replicated but this verifier is not wrapped in DDP; each rank "
                "receives only its local-source router gradient in the module graph."
            ),
            "shared_experts": (
                "Shared experts are replicated and not DDP-reduced in this isolated MoELayer run; "
                "they are included in forward parity but local expert gradients are the strict "
                "backward/update target for EP all-to-all."
            ),
        },
    }
    return local_row


def _max_nested_value(rows: list[dict[str, Any]], key: str) -> float:
    values = []

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            if key in item and isinstance(item[key], (int, float)):
                values.append(float(item[key]))
            for value in item.values():
                walk(value)
        elif isinstance(item, list):
            for value in item:
                walk(value)

    walk(rows)
    return max(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--expert-parallel-size", type=int, default=8)
    parser.add_argument("--local-tokens", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-loss-abs", type=float, default=1e-4)
    parser.add_argument("--max-output-abs", type=float, default=0.04)
    parser.add_argument("--max-input-grad-abs", type=float, default=2e-5)
    parser.add_argument("--max-expert-grad-abs", type=float, default=2e-4)
    parser.add_argument("--max-state-abs", type=float, default=1e-7)
    args = parser.parse_args()

    _init_dist(args.expert_parallel_size)
    try:
        local_row = _run(args)
        gathered: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, local_row)
        if dist.get_rank() == 0:
            rows = list(gathered)
            failures = [
                f"rank{row['rank']}:{comp['name']}"
                for row in rows
                for comp in row["comparisons"]
                if comp["status"] != "PASS"
            ]
            failures.extend(
                f"rank{row['rank']}:local_expert_grad_nonzero"
                for row in rows
                if not row["checks"]["local_expert_grad_nonzero"]
            )
            payload = {
                "date": "2026-05-31",
                "seed": SEED,
                "status": "PASS" if not failures else "FAIL",
                "scope": "DeepSeek-V4 real EP=8 MoELayer external-reference forward/backward/update parity",
                "reference": "explicit PyTorch score-routed MoE formula over all gathered ranks",
                "dtype": "bfloat16",
                "world_size": dist.get_world_size(),
                "config": {
                    "hidden_size": 4096,
                    "num_moe_experts": args.expert_parallel_size,
                    "expert_parallel_size": args.expert_parallel_size,
                    "num_local_experts_per_rank": 1,
                    "moe_ffn_hidden_size": 2048,
                    "moe_router_topk": 6,
                    "moe_router_score_function": "sqrtsoftplus",
                    "moe_router_pre_softmax": True,
                    "moe_router_topk_scaling_factor": 1.5,
                    "moe_router_enable_expert_bias": True,
                    "moe_token_dispatcher_type": "alltoall",
                    "moe_grouped_gemm": True,
                    "moe_shared_expert_intermediate_size": 2048,
                    "activation": "SwiGLU",
                    "activation_func_clamp_value": 10.0,
                },
                "manual_update": {"rule": "sgd", "lr": args.lr},
                "thresholds": {
                    "max_loss_abs": args.max_loss_abs,
                    "max_output_abs": args.max_output_abs,
                    "max_input_grad_abs": args.max_input_grad_abs,
                    "max_expert_grad_abs": args.max_expert_grad_abs,
                    "max_state_abs": args.max_state_abs,
                },
                "global_summary": {
                    "output_max_abs_global_max": _max_nested_value(
                        [
                            comp
                            for row in rows
                            for comp in row["comparisons"]
                            if comp["name"] == "forward_output"
                        ],
                        "max_abs",
                    ),
                    "input_grad_max_abs_global_max": _max_nested_value(
                        [
                            comp
                            for row in rows
                            for comp in row["comparisons"]
                            if comp["name"] == "input_grad"
                        ],
                        "max_abs",
                    ),
                    "expert_grad_max_abs_global_max": _max_nested_value(
                        [
                            comp
                            for row in rows
                            for comp in row["comparisons"]
                            if comp["name"] in ("local_expert_fc1_grad", "local_expert_fc2_grad")
                        ],
                        "max_abs",
                    ),
                    "expert_state_after_step_max_abs_global_max": _max_nested_value(
                        [
                            comp
                            for row in rows
                            for comp in row["comparisons"]
                            if comp["name"] in ("local_expert_fc1_sgd_update", "local_expert_fc2_sgd_update")
                        ],
                        "max_abs",
                    ),
                    "loss_abs_global_max": _max_nested_value(
                        [
                            comp
                            for row in rows
                            for comp in row["comparisons"]
                            if comp["name"] == "loss"
                        ],
                        "max_abs",
                    ),
                    "total_routed_assignments": sum(row["routing_tokens_selected_global"] for row in rows)
                    // max(1, dist.get_world_size()),
                    "per_expert_selected_tokens": [
                        row["routing_tokens_selected_for_local_expert"] for row in sorted(rows, key=lambda item: item["rank"])
                    ],
                    "ranks_with_nonzero_local_expert_grad": sum(
                        1 for row in rows if row["local_expert_grad_nonzero_count"] > 0
                    ),
                },
                "rank_summaries": rows,
                "failures": failures,
                "boundary": (
                    "This validates the real EP=8 MoELayer path with all-to-all dispatch, TE grouped "
                    "routed experts, shared expert forward contribution, sqrtsoftplus routing, expert "
                    "bias, local expert gradients, and one-step updates. It is narrower than a loaded "
                    "4-layer mini-checkpoint external SFT reference, but stronger than isolated "
                    "dispatcher-only or one-rank MoE checks."
                ),
                "numerical_note": (
                    "The loss comparison is a scalar dot-reduction over BF16 MoE outputs and random "
                    "upstream gradients. It uses a 1e-4 bound because repeated TE grouped GEMM/all-to-all "
                    "runs show BF16 reduction-level variation while forward output, input gradients, "
                    "local expert gradients, and one-step expert updates remain within tighter bounds."
                ),
                "runtime": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
            }
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return 0 if local_row["status"] == "PASS" else 1
    finally:
        _destroy_dist()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

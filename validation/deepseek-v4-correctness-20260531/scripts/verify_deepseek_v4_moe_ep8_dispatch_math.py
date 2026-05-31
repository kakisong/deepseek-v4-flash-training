#!/usr/bin/env python3
"""Verify EP=8 all-to-all MoE dispatch/combine math.

This verifier isolates Megatron's all-to-all token dispatcher used by the
DeepSeek-V4 MoE path.  It runs on 8 ranks, dispatches deterministic routed
tokens to expert-parallel ranks, applies a hand-checkable expert formula, then
combines tokens back and compares forward/backward/update against a direct
reference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


SEED = 20260531


def _init_dist(expert_parallel_size: int) -> None:
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    assert dist.get_world_size() == expert_parallel_size, (
        f"this verifier expects world_size={expert_parallel_size}, got {dist.get_world_size()}"
    )
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=expert_parallel_size,
        expert_tensor_parallel_size=1,
        create_gloo_process_groups=False,
    )


def _destroy_dist() -> None:
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _pg_collection() -> ProcessGroupCollection:
    return ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "ep", "expt_tp", "tp_ep"])


def _config(hidden_size: int, num_experts: int, topk: int) -> TransformerConfig:
    return TransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        ffn_hidden_size=hidden_size * 2,
        num_moe_experts=num_experts,
        moe_ffn_hidden_size=hidden_size * 2,
        moe_router_topk=topk,
        moe_router_pre_softmax=True,
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=False,
        expert_model_parallel_size=num_experts,
        expert_tensor_parallel_size=1,
        add_bias_linear=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        sequence_parallel=False,
        moe_permute_fusion=False,
        moe_expert_capacity_factor=None,
        moe_router_padding_for_quantization=False,
        moe_pad_expert_input_to_capacity=False,
    )


def _expert_scale(expert_id: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    # Powers of two and small integers are exactly representable in BF16.
    value = 1.0 + 0.5 * (expert_id % 4)
    return torch.full((hidden_size,), value, device=device, dtype=torch.bfloat16)


def _expert_bias(expert_id: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    value = 0.25 * (expert_id % 3)
    return torch.full((hidden_size,), value, device=device, dtype=torch.bfloat16)


def _build_hidden(rank: int, local_tokens: int, hidden_size: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(local_tokens, device=device, dtype=torch.int64) + rank * local_tokens
    feature = torch.arange(hidden_size, device=device, dtype=torch.int64)
    values = ((token_ids[:, None] + feature[None, :]) % 7) - 3
    return values.to(torch.bfloat16).view(local_tokens, 1, hidden_size).requires_grad_(True)


def _build_routing(
    *,
    rank: int,
    local_tokens: int,
    num_experts: int,
    topk: int,
    case_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    routing_map = torch.zeros(local_tokens, num_experts, device=device, dtype=torch.bool)
    probs = torch.zeros(local_tokens, num_experts, device=device, dtype=torch.bfloat16)
    global_token_ids = torch.arange(local_tokens, device=device, dtype=torch.int64) + rank * local_tokens
    prob_values = [1.0, 0.5, 0.25, 0.125]

    for local_idx in range(local_tokens):
        gid = int(global_token_ids[local_idx].item())
        if case_name == "top1_with_empty_expert":
            experts = [gid % (num_experts - 1)]
        elif case_name == "top3_balanced_multiroute":
            experts = [gid % num_experts, (gid + 3) % num_experts, (gid + 5) % num_experts]
        else:
            raise ValueError(f"unknown case: {case_name}")
        assert len(experts) == topk
        assert len(set(experts)) == topk
        for top_idx, expert_id in enumerate(experts):
            routing_map[local_idx, expert_id] = True
            probs[local_idx, expert_id] = prob_values[top_idx]
    return routing_map, probs


def _direct_reference(hidden: torch.Tensor, routing_map: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    hidden_flat = hidden.view(-1, hidden.shape[-1])
    out = torch.zeros_like(hidden_flat)
    hidden_size = hidden_flat.shape[-1]
    device = hidden_flat.device
    for expert_id in range(routing_map.shape[1]):
        selected = routing_map[:, expert_id]
        if not bool(selected.any().item()):
            continue
        p = probs[selected, expert_id].unsqueeze(-1)
        scale = _expert_scale(expert_id, hidden_size, device)
        bias = _expert_bias(expert_id, hidden_size, device)
        out[selected] = out[selected] + (hidden_flat[selected] * scale + bias) * p
    return out.view_as(hidden)


def _gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.stack(gathered, dim=0)


def _expected_local_expert_grads(
    hidden: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    local_expert_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_all = _gather_tensor(hidden.detach().view(-1, hidden.shape[-1]))
    routing_all = _gather_tensor(routing_map.to(torch.int8)).bool()
    probs_all = _gather_tensor(probs.detach())

    selected = routing_all[:, :, local_expert_id]
    p = probs_all[:, :, local_expert_id].unsqueeze(-1).to(torch.bfloat16)
    scale_grad = torch.where(selected.unsqueeze(-1), hidden_all * p, torch.zeros_like(hidden_all)).sum(dim=(0, 1))
    bias_grad = torch.where(
        selected.unsqueeze(-1),
        p.expand_as(hidden_all),
        torch.zeros_like(hidden_all),
    ).sum(dim=(0, 1))
    return scale_grad.to(torch.bfloat16), bias_grad.to(torch.bfloat16)


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
        "relative_l2_gap": rel_gap,
        "rtol": rtol,
        "atol": atol,
    }


def run_case(
    *,
    case_name: str,
    topk: int,
    hidden_size: int,
    local_tokens: int,
    rtol: float,
    atol: float,
    lr: float,
) -> dict[str, Any]:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    config = _config(hidden_size=hidden_size, num_experts=world_size, topk=topk)
    dispatcher = MoEAlltoAllTokenDispatcher(
        num_local_experts=1,
        local_expert_indices=[rank],
        config=config,
        pg_collection=_pg_collection(),
    )

    hidden = _build_hidden(rank, local_tokens, hidden_size, device)
    routing_map, probs = _build_routing(
        rank=rank,
        local_tokens=local_tokens,
        num_experts=world_size,
        topk=topk,
        case_name=case_name,
        device=device,
    )
    reference = _direct_reference(hidden.detach(), routing_map, probs)

    scale = _expert_scale(rank, hidden_size, device).requires_grad_(True)
    bias = _expert_bias(rank, hidden_size, device).requires_grad_(True)

    pre_hidden, pre_probs = dispatcher.dispatch_preprocess(hidden, routing_map, probs)
    dispatched_hidden, dispatched_probs = dispatcher.token_dispatch(pre_hidden, pre_probs)
    local_tokens_for_expert, tokens_per_expert, local_probs = dispatcher.dispatch_postprocess(
        dispatched_hidden, dispatched_probs
    )
    assert tokens_per_expert.numel() == 1
    expert_output = (local_tokens_for_expert * scale.unsqueeze(0) + bias.unsqueeze(0)) * local_probs.unsqueeze(-1)
    combined = dispatcher.combine_preprocess(expert_output)
    combined = dispatcher.token_combine(combined)
    output = dispatcher.combine_postprocess(combined)

    loss = output.float().sum()
    loss.backward()

    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_output = _direct_reference(ref_hidden, routing_map, probs)
    ref_loss = ref_output.float().sum()
    ref_loss.backward()

    expected_scale_grad, expected_bias_grad = _expected_local_expert_grads(hidden, routing_map, probs, rank)
    with torch.no_grad():
        scale_updated = scale - lr * scale.grad
        bias_updated = bias - lr * bias.grad
        expected_scale_updated = _expert_scale(rank, hidden_size, device) - lr * expected_scale_grad
        expected_bias_updated = _expert_bias(rank, hidden_size, device) - lr * expected_bias_grad

    comparisons = [
        _compare("forward_output", output, reference, rtol, atol),
        _compare("input_grad", hidden.grad, ref_hidden.grad, rtol, atol),
        _compare("expert_scale_grad", scale.grad, expected_scale_grad, rtol, atol),
        _compare("expert_bias_grad", bias.grad, expected_bias_grad, rtol, atol),
        _compare("expert_scale_sgd_update", scale_updated, expected_scale_updated, rtol, atol),
        _compare("expert_bias_sgd_update", bias_updated, expected_bias_updated, rtol, atol),
    ]
    local_summary = {
        "rank": rank,
        "case": case_name,
        "status": "PASS" if all(row["status"] == "PASS" for row in comparisons) else "FAIL",
        "topk": topk,
        "local_tokens": local_tokens,
        "hidden_size": hidden_size,
        "local_expert_id": rank,
        "tokens_received_by_local_expert": int(tokens_per_expert[0].item()),
        "selected_tokens_from_local_rank": int(routing_map.sum().item()),
        "comparisons": comparisons,
    }
    return local_summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--expert-parallel-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--local-tokens", type=int, default=6)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.25)
    args = parser.parse_args()

    _init_dist(args.expert_parallel_size)
    try:
        torch.manual_seed(SEED)
        cases = [
            run_case(
                case_name="top1_with_empty_expert",
                topk=1,
                hidden_size=args.hidden_size,
                local_tokens=args.local_tokens,
                rtol=args.rtol,
                atol=args.atol,
                lr=args.lr,
            ),
            run_case(
                case_name="top3_balanced_multiroute",
                topk=3,
                hidden_size=args.hidden_size,
                local_tokens=args.local_tokens,
                rtol=args.rtol,
                atol=args.atol,
                lr=args.lr,
            ),
        ]
        gathered: list[Any] = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, cases)
        if dist.get_rank() == 0:
            rank_cases = [case for rank_rows in gathered for case in rank_rows]
            payload = {
                "status": "PASS" if all(case["status"] == "PASS" for case in rank_cases) else "FAIL",
                "seed": SEED,
                "expert_parallel_size": args.expert_parallel_size,
                "dispatcher": "MoEAlltoAllTokenDispatcher",
                "dtype": "bfloat16",
                "hidden_size": args.hidden_size,
                "local_tokens_per_rank": args.local_tokens,
                "threshold": {"rtol": args.rtol, "atol": args.atol},
                "cases": [
                    {
                        "name": name,
                        "status": (
                            "PASS"
                            if all(case["status"] == "PASS" for case in rank_cases if case["case"] == name)
                            else "FAIL"
                        ),
                        "topk": next(case["topk"] for case in rank_cases if case["case"] == name),
                        "rank_summaries": [case for case in rank_cases if case["case"] == name],
                    }
                    for name in ["top1_with_empty_expert", "top3_balanced_multiroute"]
                ],
                "runtime": {
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                },
            }
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return 0 if all(case["status"] == "PASS" for case in cases) else 1
    finally:
        _destroy_dist()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    raise SystemExit(main())

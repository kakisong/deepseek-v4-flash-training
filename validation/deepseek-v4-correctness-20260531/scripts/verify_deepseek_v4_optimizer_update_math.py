#!/usr/bin/env python3
"""Verify DeepSeek-V4 optimizer path and AdamW update math evidence."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch


SEED = 20260531


def _check(condition: bool, failures: List[str], name: str) -> None:
    if not condition:
        failures.append(name)


def _compare(name: str, left: torch.Tensor, right: torch.Tensor, atol: float, rtol: float) -> Dict[str, Any]:
    diff = (left.detach().float() - right.detach().float()).abs()
    mismatches = int((diff > (atol + rtol * right.detach().float().abs())).sum().item())
    return {
        "name": name,
        "status": "PASS" if mismatches == 0 else "FAIL",
        "shape": list(left.shape),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "mismatches": mismatches,
        "atol": atol,
        "rtol": rtol,
    }


def _adamw_step_reference(
    param: torch.Tensor,
    grad: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> torch.Tensor:
    m = (1.0 - beta1) * grad.float()
    v = (1.0 - beta2) * grad.float().square()
    m_hat = m / (1.0 - beta1)
    v_hat = v / (1.0 - beta2)
    return param.float() * (1.0 - lr * weight_decay) - lr * m_hat / (v_hat.sqrt() + eps)


def _adamw_step_closed_form(
    param: torch.Tensor,
    grad: torch.Tensor,
    *,
    lr: float,
    eps: float,
    weight_decay: float,
) -> torch.Tensor:
    return param.float() * (1.0 - lr * weight_decay) - lr * grad.float() / (grad.float().abs() + eps)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _grep_required(path: Path, needles: List[str]) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "file": path.name,
        "checks": {needle: (needle in text) for needle in needles},
    }


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("docs/en/advanced"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-state-bound", type=float, default=2e-5)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    failures: List[str] = []

    repo = args.repo_root
    train_path = _grep_required(
        repo / "miles/backends/megatron_utils/model.py",
        [
            "get_megatron_optimizer",
            "OptimizerConfig",
            "optimizer.prepare_grads()",
            "optimizer.step()",
            "opt_param_scheduler.step(increment=args.global_batch_size)",
        ],
    )
    script_path = _grep_required(
        repo / "scripts/run_deepseek_v4.py",
        [
            "--optimizer adam",
            "--lr 1e-6",
            "--weight-decay 0.1",
            "--adam-beta1 0.9",
            "--adam-beta2 0.98",
        ],
    )
    for check_name, passed in train_path["checks"].items():
        _check(passed, failures, "training_path.{}".format(check_name))
    for check_name, passed in script_path["checks"].items():
        _check(passed, failures, "deepseek_v4_script.{}".format(check_name))

    cases = []
    for name, shape in [
        ("attn_sink", (64,)),
        ("norm_weight", (1024,)),
        ("attention_projection_tile", (512, 4096)),
        ("hc_scale", (4, 4096)),
    ]:
        param = (torch.randn(*shape, dtype=torch.float32) * 0.02).requires_grad_(False)
        grad = torch.randn(*shape, dtype=torch.float32) * 0.001
        reference = _adamw_step_reference(
            param,
            grad,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        closed_form = _adamw_step_closed_form(
            param,
            grad,
            lr=args.lr,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        comparison = _compare(name + "_reference_vs_closed_form", reference, closed_form, atol=1e-9, rtol=0.0)
        _check(comparison["status"] == "PASS", failures, comparison["name"])

        zero_grad = torch.zeros_like(grad)
        zero_update = _adamw_step_reference(
            param,
            zero_grad,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        wd_only = param.float() * (1.0 - args.lr * args.weight_decay)
        zero_comparison = _compare(name + "_zero_grad_weight_decay_only", zero_update, wd_only, atol=1e-9, rtol=0.0)
        _check(zero_comparison["status"] == "PASS", failures, zero_comparison["name"])

        # Worst-case first-step AdamW adaptive update term is bounded in
        # [-lr, lr] per element, so two backends with identical starting
        # parameters can differ by at most 2*lr before dtype rounding, regardless
        # of gradient magnitude or sign.
        grad_left = torch.full(shape, -1.0, dtype=torch.float32)
        grad_right = torch.full(shape, 1.0, dtype=torch.float32)
        left = _adamw_step_reference(
            param,
            grad_left,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        right = _adamw_step_reference(
            param,
            grad_right,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
        sign_flip_max_abs = float((left - right).abs().max().item())
        sign_flip_bound = 2.0 * args.lr
        _check(
            sign_flip_max_abs <= sign_flip_bound * 1.001,
            failures,
            name + "_first_step_sign_flip_bound",
        )
        _check(
            sign_flip_bound <= args.max_state_bound,
            failures,
            name + "_first_step_sign_flip_bound_within_state_threshold",
        )

        cases.append(
            {
                "name": name,
                "shape": list(shape),
                "reference_vs_closed_form": comparison,
                "zero_grad_weight_decay_only": zero_comparison,
                "first_step_sign_flip_max_abs": sign_flip_max_abs,
                "first_step_sign_flip_bound": sign_flip_bound,
                "state_threshold": args.max_state_bound,
            }
        )

    tolerance = _load_json(args.artifacts_dir / "deepseek-v4-end-to-end-bf16-tolerance-20260531.json")
    proof_ledger = _load_json(args.artifacts_dir / "deepseek-v4-proof-ledger-20260531.json")
    _check(tolerance.get("status") == "PASS", failures, "end_to_end_tolerance.status")
    _check(proof_ledger.get("status") == "PASS", failures, "proof_ledger.status")

    payload = {
        "date": "2026-05-31",
        "scope": "DeepSeek-V4 optimizer path and AdamW first-step update math",
        "status": "PASS" if not failures else "FAIL",
        "seed": SEED,
        "optimizer_hyperparameters": {
            "optimizer": "adam",
            "lr": args.lr,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "eps": args.eps,
            "weight_decay": args.weight_decay,
        },
        "training_path_static_checks": {
            "megatron_train_step": train_path,
            "deepseek_v4_script": script_path,
        },
        "adamw_math_cases": cases,
        "linked_artifacts": {
            "end_to_end_bf16_tolerance": "deepseek-v4-end-to-end-bf16-tolerance-20260531.json",
            "proof_ledger": "deepseek-v4-proof-ledger-20260531.json",
        },
        "failures": failures,
        "conclusion": (
            "Miles DeepSeek-V4 uses the Megatron optimizer path with Adam settings in the "
            "DeepSeek-V4 runner, and the first AdamW step from zero moments is exactly "
            "equivalent to the closed-form update used for the selected-update proof. "
            "The worst-case first-step adaptive update disagreement is bounded by 2*lr, "
            "which is below the selected-state threshold."
            if not failures
            else "Optimizer path or AdamW update math checks failed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(_main())

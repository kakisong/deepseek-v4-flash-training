# DeepSeek-V4 FP32 Strict Closure Attempt

This note records the follow-up question:

> Can we simply run the loaded mini checkpoint in FP32 to close
> `strict_mini_backend_logprob_parity = FAIL` and
> `external_reference_mini_checkpoint_one_step_train_parity = FAIL_DIAGNOSTIC`?

Short answer: **no, not with the current Miles DeepSeek-V4 runtime**.

## Why FP32 would have been useful

FP32 is still a good diagnostic idea in principle.

For `strict_mini_backend_logprob_parity`, the right FP32 test would compare the
same loaded checkpoint and same fixed batch across the dense, sparse, and
TileLang attention backends, with the Miles runtime itself running in FP32. If
the BF16 strict mismatch were only low-precision accumulation and routing
sensitivity, the FP32 backend gaps should shrink sharply.

For `external_reference_mini_checkpoint_one_step_train_parity`, the right FP32
test would run the full external reference verifier with both sides in FP32,
independent routing, and backward enabled. If the selected-gradient diagnostic
failure were only a BF16 forward-drift amplification, the selected-gradient
delta should fall back under its original `0.01` threshold.

Those are valid hypotheses. The problem is that the current production runtime
cannot execute them as true FP32 tests.

## What happened when we tried

Artifact:

- `artifacts/deepseek-v4-fp32-strict-closure-attempt-20260601.json`

The full external FP32 verifier forced:

- `bf16 = false`
- `fp16 = false`
- `fp8 = null`
- `MEGATRON_USE_KV_QAT = 0`

The run failed before checkpoint loading, during model construction:

```text
DeepSeekV4Attention.__init__ asserts self.wo_a.weight.dtype == torch.bfloat16
```

So the failure is not a numerical comparison failure. It means the current
Miles DeepSeek-V4 implementation is intentionally BF16-first at this level.

Static runtime checks agree with that:

- `DeepSeekV4Attention` asserts BF16 attention projection weights.
- DeepSeek-V4 helper code asserts BF16 activation input.
- TileLang sparse MLA backward kernels assert BF16 input with FP32 accumulation.
- Miles default Megatron arguments set `args.bf16 = not args.fp16`, so an
  unmodified mini forward verifier defaults back to BF16 rather than FP32.

## What This Proves

This result proves a boundary about the verifier strategy:

**A direct FP32 run cannot currently be used to close either remaining strict
gate.**

It does **not** prove that the original strict failures are real math bugs. It
also does **not** prove that FP32 would fail numerically if a separate FP32
DeepSeek-V4 runtime existed.

It only says that such a runtime does not exist in the current Miles production
path, so FP32 is not an available proof method for this checkpoint.

## Why Not Patch FP32 Just for the Proof

We should not remove the BF16 assertions or replace kernels just to make a proof
run pass.

That would test a different implementation from the one used by real training.
It might answer whether a hypothetical FP32 model can match an FP32 reference,
but it would not prove that the current BF16 Miles DeepSeek-V4 training path is
correct.

For this validation package, the proof target is the current training path, not
a one-off FP32 research variant.

## Impact on the Two Gates

| Gate | FP32 closure status | Final handling |
| --- | --- | --- |
| `strict_mini_backend_logprob_parity` | Not runnable on current Miles DeepSeek-V4 runtime | Remains recorded as strict `FAIL`; correctness claim relies on BF16 tolerance, routing replay, layer localization, and composed training evidence. |
| `external_reference_mini_checkpoint_one_step_train_parity` | Not runnable on current Miles DeepSeek-V4 runtime | Remains `FAIL_DIAGNOSTIC`; it is not used as a training PASS claim. |

Therefore the correct documentation is:

**FP32 would be a useful diagnostic if Miles had a true FP32 DeepSeek-V4 runtime,
but it cannot close the current proof. The training-correctness conclusion must
remain grounded in BF16 production-path validation.**

## Environment Recorded for the Attempt

The attempt was run in the same 8-rank validation container used for the other
mini-checkpoint checks.

| Item | Value |
| --- | --- |
| NVIDIA driver | `580.126.20` |
| CUDA reported by `nvidia-smi` | `13.0` |
| PyTorch | `2.9.1+cu129` |
| PyTorch CUDA | `12.9` |
| Miles git head in container | `34684df` |
| Megatron source | source directory is not a git worktree in the container |

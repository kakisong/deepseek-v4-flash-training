# DeepSeek-V4 Correctness Validation Snapshot

This directory archives the DeepSeek-V4 training correctness validation snapshot from 2026-05-31.

## Contents

- `docs/`: human-readable validation write-ups. Start with `docs/deepseek-v4-correctness-explained.md` for a lower-barrier explanation, then use `docs/deepseek-v4-hyperconnection-runtime.md` for the full technical detail.
- `scripts/`: verifier and trace-comparison scripts used for the validation.
- `artifacts/`: machine-readable JSON validation outputs.
- `MANIFEST.sha256`: SHA256 checksums for the archived files.

## Key Results

- HyperConnection / mHC orientation check: PASS.
- QAT, RoPE, dense/sparse attention, TileLang sparse MLA operator math: PASS.
- Attention module, TransformerBlock, Grouped MLP, EP=8 dispatch training-step math, and real EP=8 MoELayer external reference: PASS.
- Mini-checkpoint attention I/O training-step replay: PASS.
- Complete SFT one-step with attention-output straight-through replay: PASS.
- SFT loss explicit PyTorch reference: PASS. Miles loss/fused CE matches `log_softmax + gather + loss_mask` exactly at loss level, with token count exact and max per-token logprob gap `1.9073486328125e-06`.
- SFT loss backward/update reference: PASS. On the loaded 4-layer mini checkpoint, the fused CE path and explicit PyTorch loss formula produce bounded loss delta (`0.00048828125`), selected-gradient delta (`0.009491967037320137`), and one-step selected-state delta (`9.531504474580288e-10`) under the declared BF16/fused-CE tolerance.
- Mini-checkpoint framework-level correctness gate: PASS under the declared BF16 training tolerance. This includes a fresh loaded 4-layer mini-checkpoint SFT attention-output replay rerun, the SFT loss explicit forward/backward/update reference, routed MoE evidence, score-routed MoE/shared-expert external reference, real EP=8 MoELayer external reference, EP=8 dispatch math, attention I/O training-step bounds, c0/c4/c128 external attention references, optimizer math, and the BF16 envelope.
- External training-reference one-layer non-compressed block parity: PASS.
- External training-reference one-layer `compress_ratio=4` indexer block parity: PASS within BF16/indexer-selection tolerance; indexer top-k selection pressure is exercised.
- External training-reference one-layer deterministic `compress_ratio=128` compressed-attention block parity: PASS within BF16/FP32 rounding tolerance; output/loss are exact.
- External training-reference one-layer score-routed MoE/shared-expert block parity: PASS within BF16 MoE tolerance; gradients and one-step update remain within tight bounds.
- External training-reference real EP=8 MoELayer parity: PASS within BF16 MoE tolerance; real all-to-all dispatch, shared expert forward, local expert gradients, and one-step local expert updates are covered.
- Loaded 4-layer full external forward/loss reference: PASS with Miles routing replay under BF16 tolerance. Independent routing localizes the remaining branch discontinuity to the layer-3 score-routed router after hash-routed layers 0/1/2 match exactly.
- Loaded 4-layer full external one-step train delta: FAIL_DIAGNOSTIC on selected-gradient strict thresholds. This artifact is retained as a boundary; it is not used as a training PASS claim.
- FP32 strict closure attempt: NOT_RUNNABLE on the current Miles DeepSeek-V4 runtime. A true FP32 run would be the right diagnostic shape for strict logprob parity and full external selected-gradient parity, but the current production implementation is BF16-first and asserts BF16 attention/kernel inputs during model construction. Therefore FP32 does not close either gate.
- Layer-by-layer mini-checkpoint drift localization rerun: PASS_WITH_LAYERWISE_DRIFT_LOCALIZED. With dense routing replay fixed, embedding and layer-0 input norm are exact, and the first nonzero backend divergence appears at layer-0 self-attention output; attention-output replay evidence remains the explanation for strict logprob parity FAIL.
- Layer-0 self-attention internal localization: PASS_WITH_ATTENTION_CORE_LOCALIZED. Q projection, KV projection, norm, RoPE, and KV QAT tensors are exact across dense/sparse/tilelang. The first internal nonzero tensor is `attention_core`, before inverse RoPE and output projection, explaining strict logprob parity FAIL as BF16 attention-core backend numerical drift.
- Layer-0 attention-core external oracle: PASS. An in-script PyTorch FP64/FP32 masked-attention formula, using recorded Q/KV tensors and checkpoint `attn_sink`, validates dense/sparse/tilelang attention_core outputs under the declared BF16 envelope. This rules out a formula-level attention_core error.
- Official-vs-Miles full-forward BF16 tolerance gate: PASS.
- End-to-end BF16 tolerance envelope: PASS.
- Proof coverage matrix and proof ledger: PASS.

Strict official/reference logprob parity is still recorded as FAIL. Full mini-checkpoint external reference one-step train parity is no longer `MISSING_INPUT`: it was implemented and run, but remains `FAIL_DIAGNOSTIC` on selected-gradient delta because residual BF16 forward drift is amplified by backward through the complete 4-layer graph. The mini-checkpoint training correctness gate itself is PASS under the declared BF16 tolerance, using the composed SFT loss, attention-output replay, MoE, optimizer, and BF16-envelope proof chain.

The FP32 follow-up is recorded separately in `docs/deepseek-v4-fp32-strict-closure.md` and `artifacts/deepseek-v4-fp32-strict-closure-attempt-20260601.json`. It is a verifier-strategy boundary, not a new numerical PASS.

## Integrity Check

Run from the repository root:

```bash
cd validation/deepseek-v4-correctness-20260531
sha256sum -c MANIFEST.sha256
```

The scripts are archived for reproducibility. They assume the Miles/Megatron/DeepSeek-V4 runtime environment described in `docs/deepseek-v4-hyperconnection-runtime.md`.

# DeepSeek-V4 Correctness Validation Snapshot

This directory archives the DeepSeek-V4 training correctness validation snapshot from 2026-05-31.

## Contents

- `docs/`: human-readable validation write-up.
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
- Mini-checkpoint framework-level correctness gate: PASS under the declared BF16 training tolerance. This includes a fresh loaded 4-layer mini-checkpoint SFT attention-output replay rerun, the SFT loss explicit reference, routed MoE evidence, score-routed MoE/shared-expert external reference, real EP=8 MoELayer external reference, EP=8 dispatch math, attention I/O training-step bounds, c0/c4/c128 external attention references, optimizer math, and the BF16 envelope.
- External training-reference one-layer non-compressed block parity: PASS.
- External training-reference one-layer `compress_ratio=4` indexer block parity: PASS within BF16/indexer-selection tolerance; indexer top-k selection pressure is exercised.
- External training-reference one-layer deterministic `compress_ratio=128` compressed-attention block parity: PASS within BF16/FP32 rounding tolerance; output/loss are exact.
- External training-reference one-layer score-routed MoE/shared-expert block parity: PASS within BF16 MoE tolerance; gradients and one-step update remain within tight bounds.
- External training-reference real EP=8 MoELayer parity: PASS within BF16 MoE tolerance; real all-to-all dispatch, shared expert forward, local expert gradients, and one-step local expert updates are covered.
- Official-vs-Miles full-forward BF16 tolerance gate: PASS.
- End-to-end BF16 tolerance envelope: PASS.
- Proof coverage matrix and proof ledger: PASS.

Strict official/reference logprob parity is still recorded as FAIL. Full mini-checkpoint external reference one-step train parity remains `MISSING_INPUT` only for the stronger monolithic-reference claim: SFT loss, one-layer score-routed MoE, and real EP=8 MoELayer are now independently covered, but we have not rewritten the entire 4-layer checkpoint, loaded weights, SFT loss, backward, and update as a single independent PyTorch reference. The mini-checkpoint training correctness gate itself is now PASS under the declared BF16 tolerance.

## Integrity Check

Run from the repository root:

```bash
cd validation/deepseek-v4-correctness-20260531
sha256sum -c MANIFEST.sha256
```

The scripts are archived for reproducibility. They assume the Miles/Megatron/DeepSeek-V4 runtime environment described in `docs/deepseek-v4-hyperconnection-runtime.md`.

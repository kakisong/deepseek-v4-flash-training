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
- Attention module, TransformerBlock, Grouped MLP, EP=8 dispatch training-step math: PASS.
- Mini-checkpoint attention I/O training-step replay: PASS.
- Complete SFT one-step with attention-output straight-through replay: PASS.
- External training-reference one-layer non-compressed block parity: PASS.
- External training-reference one-layer deterministic `compress_ratio=128` compressed-attention block parity: PASS within BF16/FP32 rounding tolerance; output/loss are exact.
- Official-vs-Miles full-forward BF16 tolerance gate: PASS.
- End-to-end BF16 tolerance envelope: PASS.
- Proof coverage matrix and proof ledger: PASS.

Strict official/reference logprob parity is still recorded as FAIL. Full mini-checkpoint external reference one-step train parity remains `MISSING_INPUT`; the first external training-reference gate now passes for one-layer non-compressed and deterministic `compress_ratio=128` training blocks, and the remaining work is to extend that reference to the `compress_ratio=4` indexer path, routed MoE, loaded mini-checkpoint weights, and SFT loss.

## Integrity Check

Run from the repository root:

```bash
cd validation/deepseek-v4-correctness-20260531
sha256sum -c MANIFEST.sha256
```

The scripts are archived for reproducibility. They assume the Miles/Megatron/DeepSeek-V4 runtime environment described in `docs/deepseek-v4-hyperconnection-runtime.md`.

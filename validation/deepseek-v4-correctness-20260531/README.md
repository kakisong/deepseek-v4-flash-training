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
- Official-vs-Miles full-forward BF16 tolerance gate: PASS.
- End-to-end BF16 tolerance envelope: PASS.
- Proof coverage matrix and proof ledger: PASS.

Strict official/reference logprob parity is still recorded as FAIL, and external reference one-step train parity is intentionally recorded as `SKIPPED_FORWARD_STRICT_PARITY_REQUIRED` until strict official/reference forward parity passes.

## Integrity Check

Run from the repository root:

```bash
sha256sum -c validation/deepseek-v4-correctness-20260531/MANIFEST.sha256
```

The scripts are archived for reproducibility. They assume the Miles/Megatron/DeepSeek-V4 runtime environment described in `docs/deepseek-v4-hyperconnection-runtime.md`.

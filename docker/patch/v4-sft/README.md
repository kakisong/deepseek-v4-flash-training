# DeepSeek V4 SFT Megatron Patch

`megatron-dsv4-pr28.patch` carries the Megatron-Core side of the
DeepSeek-V4-Flash integration used by this training repository.

The image build checks out `radixark/Megatron-LM` at
`d9a5080f2bafea5246add1489247f25368b8a120`, applies this patch, then installs
Megatron from `/root/Megatron-LM`. See `../../Dockerfile.v4-sft` and
`../../Dockerfile.dev`.

This patch is kept here instead of relying on a moving Megatron branch so the
V4 SFT image is reproducible. It should be updated together with the pinned
`MEGATRON_COMMIT`.

## What It Adds

- DeepSeek V4 config surface:
  - adds `experimental_attention_variant=dsv4` and `dsv4_mode`;
  - adds hyper-connection, compressor, sparse attention window, hash-routing,
    output projection grouping, and expert-bias freeze settings;
  - adds YaRN arguments needed by the V4 MLA path.
- Hyper-Connection runtime:
  - expands V4 hidden states to `[seq, batch, hc_mult, hidden]`;
  - teaches PP shape exchange and scheduling to carry 4-D activations;
  - keeps HC parameters in FP32 through Megatron's fp16/bf16 module wrapping.
- DSV4 attention and indexer support:
  - wires the DSA indexer to the Miles DeepSeek V4 compressor, RoPE helpers,
    CP-aware RoPE slicing, QAT simulation, and indexer replay;
  - fixes top-k selection when compressed key length is smaller than sequence
    length;
  - passes original hidden states and compressed query state into the `dsv4`
    attention variant.
- MoE routing support:
  - propagates `input_ids` from `GPTModel` through transformer blocks/layers to
    MoE routing;
  - supports deterministic token-id-to-expert hash routing for early V4 layers;
  - supports `sqrtsoftplus` router scores, frozen router gates, and frozen
    expert score correction bias;
  - bypasses rollout routing replay for deterministic hash-routed layers.
- Numeric and training fixes:
  - allows FP32 gradient all-reduce in tensor-parallel copy backward;
  - fixes CPU-offload optimizer copy direction and blocking synchronization
    around refreshed FP32/master parameters;
  - lets shared experts opt out of activation clamp behavior separately from
    routed experts.

## Runtime Expectations

- Miles and `miles_plugins.models.deepseek_v4` must be importable in the image.
- `PYTHONPATH` should include the patched `/root/Megatron-LM` when using the
  Megatron backend.
- The patch is not a generic Megatron upgrade. It is the V4-specific runtime
  delta required by the current SFT validation image.

## Maintenance Notes

- If the Megatron base commit changes, re-apply or regenerate this patch against
  the new base and run the V4 smoke/256K validation again.
- If the corresponding V4 changes land in the consumed Megatron branch, remove
  this patch and the `git apply` step together.
- Keep the Miles framework changes separate from this patch; this file only
  documents the Megatron-side delta baked into the training image.

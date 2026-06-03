# SFT-only image

`Dockerfile.sft-only` builds a DeepSeek V4 Flash training image for Miles SFT
runs only. It does not inherit from `lmsysorg/sglang` and intentionally avoids:

- `sglang`
- `sglang-router`
- `flashinfer`
- `sglang-kernel`

It keeps the training stack required by the current V4 Megatron path:

- torch/cu129 + cuDNN
- Ray
- Megatron-LM with `patch/v4-sft/megatron-dsv4-pr28.patch`
- TileKernels
- Transformer Engine
- Apex
- fast-hadamard-transform
- torch_memory_saver
- Miles with the `--sft-only` patch

## Build

After the Miles SFT-only patch is committed and pushed, build with an immutable
Miles commit instead of the branch default:

```bash
docker build \
  -f docker/Dockerfile.sft-only \
  --build-arg MILES_REPO=https://github.com/kakisong/miles.git \
  --build-arg MILES_COMMIT=6713301501e5401939b500b4d365cbfa3d24aa57 \
  -t radixark/miles:sft-only-v4deps-20260603 \
  .
```

The build validates that V4 Megatron/TileKernels import correctly and that
SGLang/FlashInfer packages are absent.

## Runtime

Use this image only for commands carrying both:

```bash
--sft-only --debug-train-only
```

The normal RL/rollout path still requires the full image with SGLang runtime.

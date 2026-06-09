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
- DeepEP v1.2.1 for Megatron flex MoE dispatch
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
  -t radixark/miles:sft-only-v4deps-deepep-20260606 \
  .
```

DeepEP v1.2.1 is now installed from the **known-good prebuilt wheel**
`docker/wheels/deep_ep-1.2.1+9af0e0d-cp312-cp312-linux_x86_64.whl` (was an nvcc
source build, which was silently absent from the deployed `...-20260603` tag and
forced a fsx-wheel pip-install at every ray-worker startup). The build still
fail-fast-validates that V4 Megatron/TileKernels and `deep_ep`/`deep_ep_cpp` import
correctly while SGLang/FlashInfer are absent — so a deep_ep-less image cannot ship.

**Deploy:** after building, push this tag and set it as the ray-gpu-worker
DaemonSet image (and `V4_IMAGE` in the k8s fleet env). Then run
`cluster/k8s/patch_ray_worker_storage.sh` once to re-apply the ray-worker storage
fixes (temp-dir → big overlay disk, object-store cap) that are NOT bakeable into the
image. Background: `docs/h200_bottleneck_analysis.md`.

## Runtime

Use this image only for commands carrying both:

```bash
--sft-only --debug-train-only
```

The normal RL/rollout path still requires the full image with SGLang runtime.

#!/usr/bin/env bash
# Megatron torch_dist (full iter_XXXXXXX: weights + optimizer) -> bf16 HF safetensors, with the
# optimizer state and every _extra_state stripped in the same load pass. Inverse of
# convert_hf_to_megatron.sh, and the front of the post-run FP8 eval chain:
#     iter_XXXX --(this)--> HF bf16 --(inference/convert.py FP4+FP8)--> SGLang/generate.py eval.
#
# Unlike convert_hf_to_megatron.sh (8-node distributed conversion via ssh + docker exec), this is a
# SINGLE-PROCESS CPU job: the bf16 export needs no model parallelism, only ~569GB CPU RAM. We submit
# it as a Ray job pinned to one verified-idle node (>=600GB free) so it runs inside the training
# image (which has the miles deps) without hand-managing docker exec. NO GPU, NO distributed.
#
# Run this FROM the Ray head container, where the Ray dashboard is reachable on 127.0.0.1:8201.
#
# IMPORTANT: pick CONV_NODE_IP by REAL occupancy (nvidia-smi / free on every candidate), NOT by ray
# accounting or fleet membership — fleet "excluded" nodes can still be running production, and ray's
# available-resources view is scheduler bookkeeping, not physical idleness. (See the
# gpu-node-occupancy-not-from-fleet lesson.)
#
# Params (env-overridable; defaults reproduce the iter_1164 -> hf_kaynzhang_128k_iter1164 deliverable):
#   CONV_INPUT_DIR    full torch_dist checkpoint dir (iter_XXXXXXX)
#   CONV_OUTPUT_DIR   destination HF bf16 dir
#   CONV_NODE_IP      NodeManagerAddress (10.3.x) of a verified-idle, high-RAM node
#   CONV_HF_TEMPLATE  bf16-unpacked HF dir, source of config/tokenizer assets copied into the output
#   RAY_ADDRESS       Ray dashboard endpoint (default http://127.0.0.1:8201)
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash"

MILES="${V4_MILES:-$ROOT/train/miles}"
CONV_INPUT_DIR="${CONV_INPUT_DIR:-$ROOT/outputs/stageKaynzhang077-134K-3ep-H200-20260606-172101/checkpoints/iter_0001164}"
CONV_OUTPUT_DIR="${CONV_OUTPUT_DIR:-$ROOT/outputs/hf_kaynzhang_128k_iter1164}"
CONV_HF_TEMPLATE="${CONV_HF_TEMPLATE:-$ROOT/models/DeepSeek-V4-Flash-bf16-unpacked}"
CONV_NODE_IP="${CONV_NODE_IP:-10.3.74.204}"
RAY_ADDRESS="${RAY_ADDRESS:-http://127.0.0.1:8201}"

WRAP="$SCRIPT_DIR/convert_torch_dist_to_hf_stub.py"

[[ -d "$CONV_INPUT_DIR" ]] || { echo "[err] input ckpt dir missing: $CONV_INPUT_DIR" >&2; exit 1; }
[[ -f "$CONV_INPUT_DIR/common.pt" ]] || echo "[warn] no common.pt in $CONV_INPUT_DIR — converter requires it" >&2
[[ -f "$WRAP" ]] || { echo "[err] driver missing: $WRAP" >&2; exit 1; }

# Build the Ray runtime-env: PYTHONPATH for the in-image megatron/tilelang + miles, plus the
# CONV_* params the driver reads.
RUNTIME_ENV="$(MILES="$MILES" IN="$CONV_INPUT_DIR" OUT="$CONV_OUTPUT_DIR" TPL="$CONV_HF_TEMPLATE" \
  python3 - <<'PY'
import json
import os

print(json.dumps({"env_vars": {
    "PYTHONPATH": f"/root/Megatron-LM:/root/TileKernels:{os.environ['MILES']}",
    "PYTHONUNBUFFERED": "1",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
    "V4_MILES": os.environ["MILES"],
    "CONV_INPUT_DIR": os.environ["IN"],
    "CONV_OUTPUT_DIR": os.environ["OUT"],
    "CONV_HF_TEMPLATE": os.environ["TPL"],
}}))
PY
)"

echo "[convert] $CONV_INPUT_DIR"
echo "      ->  $CONV_OUTPUT_DIR   (bf16 HF, optimizer + _extra_state stripped)"
echo "      on  node $CONV_NODE_IP via $RAY_ADDRESS"
ray job submit --address="$RAY_ADDRESS" --no-wait \
  --entrypoint-resources "{\"node:$CONV_NODE_IP\":0.001}" \
  --runtime-env-json="$RUNTIME_ENV" \
  -- env CUDA_VISIBLE_DEVICES=0 RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 \
     python3 "$WRAP"

echo "[convert] submitted (--no-wait). progress -> ${CONV_OUTPUT_DIR%/}.convert.log"

#!/usr/bin/env bash
# Launch sglang inference server for a converted HF ckpt.
# Run from inside the miles-v4-sft container on a node with 8 free GPUs.
#
#   bash launch_sglang.sh <hf_dir> [port]
#
# Default port 30000. Will block; ctrl-C to stop.

set -euo pipefail
HF_DIR="${1:?usage: $0 <hf_dir> [port]}"
PORT="${2:-30000}"

# DeepSeek-V4-Flash needs DSV4 attention; sglang upstream may not have it yet,
# so we use --attention-backend triton + --enable-mixed-chunk for safety.
# tp=8 fits one full BF16 replica on 8×H20.
exec python3 -m sglang.launch_server \
  --model-path "$HF_DIR" \
  --tp 8 \
  --trust-remote-code \
  --port "$PORT" \
  --host 0.0.0.0 \
  --mem-fraction-static 0.85 \
  --max-running-requests 8 \
  --context-length 32768 \
  --disable-cuda-graph

#!/usr/bin/env bash
# 为转换后的 HF ckpt 启动 sglang 推理服务。
# 需在拥有 8 张空闲 GPU 的节点上、miles-v4-sft 容器内部运行。
#
#   bash launch_sglang.sh <hf_dir> [port]
#
# 默认端口 30000。会阻塞前台;按 ctrl-C 停止。

set -euo pipefail
HF_DIR="${1:?usage: $0 <hf_dir> [port]}"
PORT="${2:-30000}"

# DeepSeek-V4-Flash 需要 DSV4 attention;sglang 上游可能尚未支持,
# 因此稳妥起见使用 --attention-backend triton + --enable-mixed-chunk。
# tp=8 可在 8×H20 上放下一个完整的 BF16 副本。
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

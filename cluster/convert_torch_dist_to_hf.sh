#!/usr/bin/env bash
# Megatron torch_dist(完整 iter_XXXXXXX:权重 + optimizer)-> bf16 HF safetensors,
# 并在同一次加载中顺带剥离 optimizer 状态和所有 _extra_state。是
# convert_hf_to_megatron.sh 的逆向操作,也是训练跑完后 FP8 评测链路的第一环:
#     iter_XXXX --(本脚本)--> HF bf16 --(inference/convert.py FP4+FP8)--> SGLang/generate.py 评测。
#
# 与 convert_hf_to_megatron.sh(经 ssh + docker exec 的 8 节点分布式转换)不同,这是一个
# 单进程 CPU 任务:bf16 导出不需要模型并行,只需约 569GB CPU 内存。我们把它作为
# Ray job 提交并钉到一台确认空闲的节点(空闲内存 >=600GB),让它跑在训练镜像里
# (自带 miles 依赖),省去手工管理 docker exec。无 GPU、无分布式。
#
# 请在 Ray head 容器内执行本脚本,那里可经 127.0.0.1:8201 访问 Ray dashboard。
#
# 重要:CONV_NODE_IP 必须按真实占用挑选(逐台候选机看 nvidia-smi / free),不能依据 ray
# 的记账或 fleet 成员关系 — fleet 中标记"excluded"的节点可能仍在跑生产任务,而 ray 的
# available-resources 视图只是调度器的记账,不代表物理空闲。(参见
# gpu-node-occupancy-not-from-fleet 教训。)
#
# 参数(可用环境变量覆盖;默认值可复现 iter_1164 -> hf_kaynzhang_128k_iter1164 交付物):
#   CONV_INPUT_DIR    完整 torch_dist checkpoint 目录(iter_XXXXXXX)
#   CONV_OUTPUT_DIR   目标 HF bf16 目录
#   CONV_NODE_IP      一台确认空闲、大内存节点的 NodeManagerAddress(10.3.x)
#   CONV_HF_TEMPLATE  bf16-unpacked HF 目录,输出中 config/tokenizer 资产的拷贝来源
#   RAY_ADDRESS       Ray dashboard 端点(默认 http://127.0.0.1:8201)
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

# 构建 Ray runtime-env:PYTHONPATH 指向镜像内的 megatron/tilelang + miles,外加
# 驱动脚本要读取的 CONV_* 参数。
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

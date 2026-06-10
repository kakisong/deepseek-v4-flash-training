#!/usr/bin/env bash
# Stage A — 1 节点 x 8 卡 H20, DeepSeek-V4-Flash 4 层 smoke 测试。
#
# 目标: 90 分钟内验证 chat-template + loss-mask + 前向/反向 + ckpt save/load。
#
# 必需的环境变量 (见 README §1.3):
#   BASE_FOLDER / MODELS / DATA / OUT / REPO / MEGATRON_PATH / MASTER_ADDR
#
# 数据: $DATA/openhermes_v4.parquet 或 $DATA/your_v4_sft.parquet
# Ckpt: $MODELS/DeepSeek-V4-Flash-4layer_torch_dist (prepare_megatron_ckpt.sh 4layer)

set -euo pipefail

# ---------- 0. 清理与检查 ---------------------------------------------------------
pkill -9 sglang || true
ray stop --force || true
pkill -9 ray    || true
pkill -9 python || true
sleep 3

: "${REPO:?REPO is not set}"
: "${MODELS:?MODELS is not set}"
: "${DATA:?DATA is not set}"
: "${OUT:?OUT is not set}"
: "${MEGATRON_PATH:?MEGATRON_PATH is not set}"
: "${MASTER_ADDR:?MASTER_ADDR is not set}"

NUM_NODES="${NUM_NODES:-1}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"

if [[ "$NUM_NODES" != "1" ]]; then
  echo "[warn] Stage A is meant to be a single-node smoke; current NUM_NODES=$NUM_NODES" >&2
fi

SFT_DATA="${SFT_DATA:-$DATA/openhermes_v4.parquet}"
TEMPLATE="${REPO}/templates/deepseek_v4.jinja"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-qwen3}"      # 默认 qwen3; 验证通过后切换到 deepseek_v4

if [[ ! -f "$SFT_DATA" ]]; then
  echo "[err] $SFT_DATA does not exist — run prepare_data.py first" >&2
  exit 1
fi
if [[ ! -d "$MODELS/DeepSeek-V4-Flash-4layer_torch_dist" ]]; then
  echo "[err] 4-layer torch_dist does not exist — run prepare_megatron_ckpt.sh 4layer first" >&2
  exit 1
fi

RUN_ID="stageA-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$OUT/$RUN_ID"
mkdir -p "$SAVE_DIR"
echo "[info] run id: $RUN_ID, save: $SAVE_DIR"

# ---------- 1. nvlink 探测(NCCL_NVLS_ENABLE) ------------------------------------
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)
echo "[info] HAS_NVLINK=$HAS_NVLINK (links=$NVLINK_COUNT)"

# ---------- 2. 模型参数 -----------------------------------------------------------
MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash-4layer.sh"
if [[ ! -f "$MODEL_CFG" ]]; then
  echo "[err] $MODEL_CFG does not exist — checkout PR #1045 first" >&2
  exit 2
fi
# shellcheck source=/dev/null
source "$MODEL_CFG"

# ---------- 3. 参数分组 -----------------------------------------------------------
CKPT_ARGS=(
  --hf-checkpoint  "$MODELS/DeepSeek-V4-Flash-bf16"
  --ref-load       "$MODELS/DeepSeek-V4-Flash-4layer_torch_dist"
  --load           "$SAVE_DIR"
  --save           "$SAVE_DIR"
  --save-interval  50
  --save-retain-interval 50
)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    "$SFT_DATA"
  --input-key      messages
  --rollout-shuffle
  --num-epoch              1
  --rollout-batch-size     32
  --global-batch-size      32

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --sft-only
  --debug-train-only
)

# Stage A 优先使用 jinja 模板; 缺失时回退到 tokenizer 内置模板。
if [[ -f "$TEMPLATE" ]]; then
  SFT_ARGS+=( --chat-template-path "$TEMPLATE" )
  echo "[info] using jinja template: $TEMPLATE"
else
  SFT_ARGS+=( --apply-chat-template )
  echo "[warn] $TEMPLATE does not exist; falling back to the tokenizer built-in template."
  echo "       If verification/verify_chat_template.py already found the mask is wrong, fix it before going to GPU."
fi
SFT_ARGS+=( --loss-mask-type "$LOSS_MASK_TYPE" )

# 4 层 + 单节点: 不需要 PP/CP; 只切 EP 以验证该路径。
PERF_ARGS=(
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1
  --sequence-parallel

  --recompute-granularity full
  --recompute-method      uniform
  --recompute-num-layers  1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 2048
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout    0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  # V4 是 sparse-MLA; 不要传 --attention-backend flash。
  --actor-num-nodes "$NUM_NODES"
  --actor-num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --colocate
  --dump-details "$SAVE_DIR/dump_details"
)

# ---------- 4. ray head ---------------------------------------------------------
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "$NUM_GPUS_PER_NODE" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONPATH": "${MEGATRON_PATH}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "no_proxy": "${no_proxy}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
  }
}
EOF
)

# ---------- 5. 提交 ---------------------------------------------------------------
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 "$REPO/train_async.py" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"

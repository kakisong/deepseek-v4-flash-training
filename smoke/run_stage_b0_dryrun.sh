#!/usr/bin/env bash
# Stage B0 — 8 nodes x 8 cards H20 full-scale dry-run.
#
# Goals (within 1-2 hours):
#   1. Verify the TP=4 / PP=4 / CP=2 / EP=16 partition starts on real 284B weights.
#   2. Peak memory < 92GB (leave headroom).
#   3. Cross-node IB / GPUDirect bandwidth is in place (single-step wall-time within budget).
#   4. ckpt save/load works under the real sharding.
#   5. Data prefetch does not starve training.
#
# Differences from Stage B:
#   - Only 50 steps (--num-rollout runs through once)
#   - Uses the same small data set as Stage A to avoid IO interference
#   - Explicit --debug-train-only so SGLang is not started
#   - --save-interval 25 triggers save at least twice to validate the ckpt path
#
# Pass criteria (README §6.B0):
#   - Startup < 15 minutes
#   - Step 1 < 90s, steady-state 30-50s
#   - Per-node peak memory < 92GB
#   - Both saves land on disk and reload consistently

set -euo pipefail

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

NUM_NODES="${NUM_NODES:-8}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TOTAL_GPUS=$(( NUM_NODES * NUM_GPUS_PER_NODE ))
DRY_RUN_STEPS="${DRY_RUN_STEPS:-50}"

if [[ "$TOTAL_GPUS" -ne 64 ]]; then
  echo "[warn] TOTAL_GPUS=$TOTAL_GPUS; the default PERF_ARGS targets 64 GPUs." >&2
fi

SFT_DATA="${SFT_DATA:-$DATA/openhermes_v4.parquet}"
LOCAL_DATA="${LOCAL_DATA:-/root/local_data}"
HF_BF16="${LOCAL_DATA}/DeepSeek-V4-Flash-bf16"
TORCH_DIST="${LOCAL_DATA}/V4-Flash_torch_dist"
TEMPLATE="${REPO}/templates/deepseek_v4.jinja"
LOSS_MASK_TYPE="${LOSS_MASK_TYPE:-deepseek_v4}"

if [[ ! -f "$SFT_DATA" ]]; then
  echo "[err] $SFT_DATA does not exist — reuse the Stage A data" >&2
  exit 1
fi
if [[ ! -d "$TORCH_DIST" ]]; then
  echo "[err] $TORCH_DIST does not exist — run prepare_megatron_ckpt.sh full and rsync first" >&2
  exit 1
fi
if [[ ! -f "$TEMPLATE" ]]; then
  echo "[err] $TEMPLATE does not exist — V4 SFT requires the official jinja template" >&2
  exit 2
fi

RUN_ID="stageB0-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$OUT/$RUN_ID"
mkdir -p "$SAVE_DIR"
echo "[info] dry-run id: $RUN_ID, save: $SAVE_DIR, steps: $DRY_RUN_STEPS"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)

MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash.sh"
if [[ ! -f "$MODEL_CFG" ]]; then
  echo "[err] $MODEL_CFG does not exist — checkout PR #1045 first" >&2
  exit 3
fi
# shellcheck source=/dev/null
source "$MODEL_CFG"

CKPT_ARGS=(
  --hf-checkpoint  "$HF_BF16"
  --ref-load       "$TORCH_DIST"
  --load           "$SAVE_DIR"
  --save           "$SAVE_DIR"
  --save-interval  25                 # fires twice within 50 steps to validate save/load
  --save-retain-interval 25
)

# Key idea: dry-run uses a small batch and few steps; the only goal is to verify the
# path can handle real-scale weights.
SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    "$SFT_DATA"
  --input-key      messages
  --rollout-shuffle
  --num-rollout            "$DRY_RUN_STEPS"
  --rollout-batch-size     128         # half of Stage B for faster steps
  --global-batch-size      128

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --sft-only
  --debug-train-only

  --chat-template-path "$TEMPLATE"
  --loss-mask-type     "$LOSS_MASK_TYPE"
)

# Identical to Stage B — the whole point of this stage is to validate this partition.
PERF_ARGS=(
  --tensor-model-parallel-size 4
  --sequence-parallel
  --pipeline-model-parallel-size 4
  --context-parallel-size 2
  --expert-model-parallel-size 16
  --expert-tensor-parallel-size 1

  --decoder-first-pipeline-num-layers 13
  --decoder-last-pipeline-num-layers  13

  --recompute-granularity full
  --recompute-method      uniform
  --recompute-num-layers  1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style constant            # dry-run does not need cosine
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95

  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout    0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --actor-num-nodes "$NUM_NODES"
  --actor-num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --num-gpus-per-node "$NUM_GPUS_PER_NODE"
  --colocate
  --use-fault-tolerance
  --dump-details "$SAVE_DIR/dump_details"
  --disable-weights-backuper
)

# ray head + workers (same as Stage B).
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "$NUM_GPUS_PER_NODE" --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -f /root/mpi_rack_hostfile ]]; then
  for WORKER_IP in $(awk '{print $1}' /root/mpi_rack_hostfile); do
    if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then continue; fi
    echo "[info] starting Ray worker on ${WORKER_IP}"
    ssh root@"${WORKER_IP}" \
      "pkill -9 sglang ; ray stop --force ; pkill -9 python ; \
       ray start --address=${MASTER_ADDR}:6379 \
                 --num-gpus ${NUM_GPUS_PER_NODE} \
                 --node-ip-address ${WORKER_IP} \
                 --disable-usage-stats \
                 --dashboard-host=0.0.0.0 --dashboard-port=8265" &
  done
  wait
fi

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

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="$RUNTIME_ENV_JSON" \
   -- python3 "$REPO/train_async.py" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}"

echo
echo "[done] dry-run finished. Checklist:"
echo "  - $SAVE_DIR/iter_0000025 and iter_0000050 should both exist"
echo "  - $SAVE_DIR/dump_details contains no NaN loss / grad_norm"
echo "  - Per-node nvidia-smi shows peak training memory < 92GB"
echo "  - Steady-state step time 30-50s (visible in ray dashboard or wandb)"

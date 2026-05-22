#!/usr/bin/env bash
# HF (BF16) → Megatron torch_dist.
#
# Usage:
#   bash examples/deepseek_v4_sft/prepare_megatron_ckpt.sh 4layer   # Stage A
#   bash examples/deepseek_v4_sft/prepare_megatron_ckpt.sh full     # Stage B
#
# Prerequisites:
#   - $MODELS/DeepSeek-V4-Flash-bf16 exists (fp8_cast_bf16 has been run)
#   - Current branch contains PR #1045 (scripts/models/deepseek-v4-flash{,-4layer}.sh)
#   - $MEGATRON_PATH points at the Megatron-LM source tree

set -euo pipefail

VARIANT="${1:-}"
case "$VARIANT" in
  4layer|full) ;;
  *)
    echo "usage: $0 {4layer|full}" >&2
    exit 1
    ;;
esac

: "${REPO:?REPO is not set (must point at the miles repo root)}"
: "${MODELS:?MODELS is not set}"
: "${MEGATRON_PATH:?MEGATRON_PATH is not set}"

HF_BF16="${MODELS}/DeepSeek-V4-Flash-bf16"
if [[ ! -d "$HF_BF16" ]]; then
  echo "[err] $HF_BF16 does not exist. Run first:" >&2
  echo "      python $REPO/tools/fp8_cast_bf16.py \\" >&2
  echo "          --input-fp8-hf-path  \$MODELS/DeepSeek-V4-Flash \\" >&2
  echo "          --output-bf16-hf-path $HF_BF16" >&2
  exit 1
fi

cd "$REPO"

if [[ "$VARIANT" == "4layer" ]]; then
  MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash-4layer.sh"
  SAVE="${MODELS}/DeepSeek-V4-Flash-4layer_torch_dist"
  TP=1
  PP=1
  EP=1
  EXTRA=""
else
  MODEL_CFG="$REPO/scripts/models/deepseek-v4-flash.sh"
  SAVE="${MODELS}/DeepSeek-V4-Flash_torch_dist"
  TP=1
  PP=8
  EP=4
  # If PP first/last stages are unbalanced, the convert tool auto-balances them.
  # To override manually, add --decoder-first-pipeline-num-layers / -last-... to EXTRA.
  EXTRA="--expert-tensor-parallel-size 1"
fi

if [[ ! -f "$MODEL_CFG" ]]; then
  echo "[err] $MODEL_CFG does not exist — checkout PR #1045 first" >&2
  exit 2
fi

if [[ -f "$SAVE/latest_checkpointed_iteration.txt" ]]; then
  echo "[skip] $SAVE already exists; skipping conversion. Delete the directory to force a rerun."
  exit 0
fi

# shellcheck source=/dev/null
source "$MODEL_CFG"

echo "[info] variant=$VARIANT  TP=$TP PP=$PP EP=$EP"
echo "[info] HF input: $HF_BF16"
echo "[info] save to: $SAVE"

PYTHONPATH="$MEGATRON_PATH" \
python "$REPO/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_BF16" \
    --save "$SAVE" \
    --tensor-model-parallel-size "$TP" \
    --pipeline-model-parallel-size "$PP" \
    --expert-model-parallel-size "$EP" \
    $EXTRA

echo "[ok] $SAVE"

#!/usr/bin/env bash
# One-click: trained Megatron `torch_dist` checkpoint  ->  original dpsk FP8+FP4.
#
#   (1) convert_torch_dist_to_hf.py   torch_dist (BF16)  ->  HF BF16
#   (1.5) verify_roundtrip.py keymap  (sanity: HF key set vs the dpsk template)
#   (2) hf_bf16_to_megablocks.py      HF BF16            ->  dpsk FP8/FP4 (46 shards)
#   (3) verify_roundtrip.py struct    (names/dtype/shape == original layout)
#
# Step (1) runs in the TRAINING image (it is a miles/Megatron op that pulls in
# sglang); steps (2)/(3) run in the lean conversion image (docker/Dockerfile.convert,
# pure PyTorch). Code + data live on the V4 work root, bind-mounted at /work.
#
# Usage:
#   # full pipeline from a torch_dist iter:
#   tools/ckpt_to_fp8fp4.sh --iter outputs/<stage>/checkpoints/iter_NNNNNNN --name <prefix>
#   # OR skip step (1) and start from an existing HF bf16 dir
#   # (e.g. produced by cluster/convert_torch_dist_to_hf.sh):
#   tools/ckpt_to_fp8fp4.sh --from-hf <existing HF bf16 dir> --name <prefix>
#
# Produces  $V4_MODELS/<prefix>-fp8fp4. With --iter it writes a ~540 GB
# intermediate $V4_MODELS/<prefix>-hf-bf16 and removes it afterwards unless
# --keep-bf16. With --from-hf the supplied dir is never deleted.
#
# Flags: --no-mtp (don't graft original MTP), --cpu, --force, --image, --template.
# Env overrides: V4_WORK, V4_CONVERT_IMAGE, V4_TRAIN_IMAGE, V4_MILES_REPO,
#                V4_TRAINING_REPO, V4_TEMPLATE, V4_BF16_DIR.
set -euo pipefail

# ---------------------------------------------------------------- config
V4_WORK="${V4_WORK:-/data_train/kaynzhang/v4-sft}"
IMAGE="${V4_CONVERT_IMAGE:-v4-convert:latest}"                            # lean pure-torch image (steps 2/3)
TRAIN_IMAGE="${V4_TRAIN_IMAGE:-radixark/miles:dev-fht-v4deps-20260529}"   # full miles+sglang stack (step 1)
MILES="${V4_MILES_REPO:-$V4_WORK/miles}"                                  # has megatron_to_hf/deepseekv4
TRAIN_REPO="${V4_TRAINING_REPO:-$V4_WORK/deepseek-v4-flash-training}"
TEMPLATE="${V4_TEMPLATE:-$V4_WORK/models/DeepSeek-V4-Flash}"              # original dpsk FP8/FP4 = layout template
ORIGIN_HF="${V4_BF16_DIR:-$V4_WORK/models/DeepSeek-V4-Flash-bf16-unpacked}"  # config/tokenizer source + keymap ref
VOCAB=129280

ITER="" ; NAME="" ; FROM_HF="" ; KEEP_BF16=0 ; WITH_MTP=1 ; USE_GPU=1 ; FORCE=0

usage() { sed -n '2,/^set -euo pipefail/{/^set -euo pipefail/!p}' "$0"; exit "${1:-0}"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --iter)        ITER="$2"; shift 2;;
    --name)        NAME="$2"; shift 2;;
    --from-hf)     FROM_HF="$2"; shift 2;;      # skip step 1: start from an existing HF bf16 dir
    --template)    TEMPLATE="$2"; shift 2;;
    --image)       IMAGE="$2"; shift 2;;
    --keep-bf16)   KEEP_BF16=1; shift;;
    --no-mtp)      WITH_MTP=0; shift;;          # do not graft the original MTP layer
    --cpu)         USE_GPU=0; shift;;
    --force)       FORCE=1; shift;;
    -h|--help)     usage 0;;
    *) echo "unknown arg: $1" >&2; usage 1;;
  esac
done
[[ -n "$NAME" ]] || { echo "ERROR: --name is required" >&2; usage 1; }
[[ -n "$ITER" || -n "$FROM_HF" ]] || { echo "ERROR: pass --iter <torch_dist> or --from-hf <HF bf16 dir>" >&2; usage 1; }

FP_OUT="$V4_WORK/models/${NAME}-fp8fp4"
if [[ -n "$FROM_HF" ]]; then
  # Start from an existing HF bf16 dir (e.g. produced by cluster/convert_torch_dist_to_hf.sh).
  [[ "$FROM_HF" = /* ]] || FROM_HF="$V4_WORK/$FROM_HF"
  HF_OUT="$FROM_HF" ; SKIP_STEP1=1 ; KEEP_BF16=1   # never delete a user-supplied input
else
  [[ "$ITER" = /* ]] || ITER="$V4_WORK/$ITER"
  HF_OUT="$V4_WORK/models/${NAME}-hf-bf16" ; SKIP_STEP1=0
fi

rel() {  # host abs path under $V4_WORK -> /work/...
  case "$1" in
    "$V4_WORK"/*) echo "/work/${1#$V4_WORK/}";;
    *) echo "ERROR: path not under V4_WORK ($V4_WORK): $1" >&2; exit 2;;
  esac
}

# ---------------------------------------------------------------- preflight
[[ -d "$TEMPLATE" ]] || { echo "ERROR: template not found: $TEMPLATE" >&2; exit 2; }
if [[ $SKIP_STEP1 -eq 1 ]]; then
  [[ -f "$HF_OUT/model.safetensors.index.json" ]] || { echo "ERROR: --from-hf is not an HF dir (no index.json): $HF_OUT" >&2; exit 2; }
else
  [[ -f "$ITER/common.pt" ]] || { echo "ERROR: $ITER is not a torch_dist ckpt (no common.pt)" >&2; exit 2; }
  [[ -d "$ORIGIN_HF" ]] || { echo "ERROR: origin-hf dir not found: $ORIGIN_HF" >&2; exit 2; }
  [[ -f "$MILES/miles/backends/megatron_utils/megatron_to_hf/deepseekv4.py" ]] \
    || { echo "ERROR: miles checkout lacks v4 megatron_to_hf: $MILES" >&2; exit 2; }
fi
if [[ -e "$FP_OUT" ]]; then
  [[ $FORCE -eq 1 ]] || { echo "ERROR: output exists ($FP_OUT); use --force" >&2; exit 2; }
  docker run --rm -v "$V4_WORK:/work" "$IMAGE" rm -rf "$(rel "$FP_OUT")"
fi

GPU=() ; [[ $USE_GPU -eq 1 ]] && GPU=(--gpus all)
dock() { docker run --rm "${GPU[@]}" -v "$V4_WORK:/work" "$@"; }
TOOLS="$(rel "$TRAIN_REPO")/tools"
echo "[ckpt_to_fp8fp4] -> $FP_OUT  (mtp=$WITH_MTP)"
echo "[ckpt_to_fp8fp4] HF bf16: $HF_OUT  $([[ $SKIP_STEP1 -eq 1 ]] && echo '(provided, step 1 skipped)' || echo "(from $ITER, keep=$KEEP_BF16)")"

# ---------------------------------------------------------------- (1) torch_dist -> HF BF16
if [[ $SKIP_STEP1 -eq 0 ]]; then
  echo "== [1/3] torch_dist -> HF BF16  (training image) =="
  dock -e PYTHONPATH="$(rel "$MILES")" -w "$(rel "$MILES")" "$TRAIN_IMAGE" \
    python3 tools/convert_torch_dist_to_hf.py \
      --input-dir "$(rel "$ITER")" --output-dir "$(rel "$HF_OUT")" \
      --model-name deepseekv4 --origin-hf-dir "$(rel "$ORIGIN_HF")" --vocab-size "$VOCAB"
else
  echo "== [1/3] skipped (--from-hf $HF_OUT) =="
fi

# ---------------------------------------------------------------- (1.5) key-set sanity
echo "== [1.5] keymap sanity (missing keys are expected to be the MTP layer) =="
dock -w "$TOOLS" "$IMAGE" \
  python3 verify_roundtrip.py keymap --template "$(rel "$TEMPLATE")" --hf-src "$(rel "$HF_OUT")" || \
  echo "   (keymap reported gaps; if they are mtp.* they are grafted in step 2 unless --no-mtp)"

# ---------------------------------------------------------------- (2) HF BF16 -> dpsk FP8/FP4
echo "== [2/3] HF BF16 -> dpsk FP8/FP4 =="
FILL=() ; [[ $WITH_MTP -eq 1 ]] && FILL=(--fill-missing-from-template)
dock -w "$TOOLS" "$IMAGE" \
  python3 hf_bf16_to_megablocks.py \
    --hf-src "$(rel "$HF_OUT")" --template "$(rel "$TEMPLATE")" --dst "$(rel "$FP_OUT")" "${FILL[@]}"

# ---------------------------------------------------------------- (3) structural verify
echo "== [3/3] structural verify vs original layout =="
dock -w "$TOOLS" "$IMAGE" \
  python3 verify_roundtrip.py struct --a "$(rel "$FP_OUT")" --b "$(rel "$TEMPLATE")"

# ---------------------------------------------------------------- cleanup
if [[ $KEEP_BF16 -eq 0 ]]; then
  echo "== cleanup intermediate HF BF16 =="
  docker run --rm -v "$V4_WORK:/work" "$IMAGE" rm -rf "$(rel "$HF_OUT")"
fi
echo "[ckpt_to_fp8fp4] DONE -> $FP_OUT"

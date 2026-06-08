#!/usr/bin/env bash
# Unified launch entry.
#
# 用法 (任选):
#   bash run.sh
#   # 默认等价于当前 H200 42-node P3-derived 配置:
#   #   --fleet h200_k8s_42node --scale tp8pp7cp6ep8 --workload sft_kaynzhang_077_134k_3epoch
#   bash run.sh --control current --fleet h20_16node --scale tp8pp16ep8_layout --workload sft_prod
#   V4_CONTROL=current V4_FLEET=h20_16node V4_SCALE=tp8pp16ep8_layout V4_WORKLOAD=sft_prod bash run.sh
#
# Common overrides (each maps to a PRESET_* / HW_* variable):
#   --num-rollout N           training steps
#   --num-epoch N             full passes over prompt data (Miles derives rollout count)
#   --lr X                    learning rate
#   --lr-decay-style S        constant / cosine / linear
#   --save-interval N         save ckpt every N steps
#   --save-retain-interval N  keep a permanent ckpt every N steps (rolling)
#   --seq-length N            Megatron sequence length
#   --max-tokens-per-gpu N    per-CP-rank token cap (trade memory for batch)
#   --global-batch-size N
#   --attn-impl IMPL          tilelang / dense
#   --recompute-granularity G none / selective / full
#   --recompute-method M      uniform / block (when granularity=full)
#   --recompute-num-layers N
#   --no-cpu-offload          disable optimizer CPU offload for fit probes
#
# --dry-run: generate launch_in_container.sh but do NOT submit it to ray.
# --no-wait: submit the ray job and return immediately instead of streaming logs until completion.
# --submit-mode ssh|k8s: run the generated launch script through SSH/docker or
#   through kubectl exec against the configured Ray head pod.
#
# Load order (later overrides earlier):
#   cluster/fleet/$V4_FLEET.env  →  cluster/control/$V4_CONTROL.env  →  cluster/base.env  →
#   cluster/hw/$V4_GPU_MODEL.env  →  cluster/scale/$V4_SCALE.env  →
#   cluster/workload/$V4_WORKLOAD.env  →  CLI overrides

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CLUSTER_DIR="$REPO_ROOT/cluster"

# ---------- parse args -------------------------------------------------------
DRY_RUN=0
PROFILE_ENABLED=0
NO_SAVE_OPTIM=0
NO_WAIT=0
declare -A OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --no-wait) NO_WAIT=1; shift ;;
    --control)               export V4_CONTROL="$2"; shift 2 ;;
    --fleet)                 export V4_FLEET="$2"; shift 2 ;;
    --scale)                 export V4_SCALE="$2"; shift 2 ;;
    --workload)              export V4_WORKLOAD="$2"; shift 2 ;;
    --submit-mode)           export V4_SUBMIT_MODE="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
      echo "Available controls:  $(ls "$CLUSTER_DIR/control/" 2>/dev/null | sed 's/\.env$//' | tr '\n' ' ')"
      echo "Available fleets:    $(ls "$CLUSTER_DIR/fleet/" | sed 's/\.env$//' | tr '\n' ' ')"
      echo "Available scales:    $(ls "$CLUSTER_DIR/scale/" | sed 's/\.env$//' | tr '\n' ' ')"
      echo "Available workloads: $(ls "$CLUSTER_DIR/workload/" | sed 's/\.env$//' | tr '\n' ' ')"
      exit 0 ;;
    --num-rollout)           OVERRIDES[PRESET_NUM_ROLLOUT]="$2"; OVERRIDES[PRESET_NUM_EPOCH]=""; shift 2 ;;
    --num-epoch)             OVERRIDES[PRESET_NUM_EPOCH]="$2"; OVERRIDES[PRESET_NUM_ROLLOUT]=""; shift 2 ;;
    --lr)                    OVERRIDES[PRESET_LR]="$2"; shift 2 ;;
    --lr-decay-style)        OVERRIDES[PRESET_LR_DECAY_STYLE]="$2"; shift 2 ;;
    --lr-warmup-iters)       OVERRIDES[PRESET_LR_WARMUP_ITERS]="$2"; shift 2 ;;
    --profile)               PROFILE_ENABLED=1; shift ;;
    --no-save-optim)         NO_SAVE_OPTIM=1; shift ;;
    --save-interval)         OVERRIDES[PRESET_SAVE_INTERVAL]="$2"; shift 2 ;;
    --save-retain-interval)  OVERRIDES[PRESET_SAVE_RETAIN_INTERVAL]="$2"; shift 2 ;;
    --seq-length)            OVERRIDES[HW_SEQ_LENGTH]="$2"; shift 2 ;;
    --max-tokens-per-gpu)    OVERRIDES[HW_MAX_TOKENS_PER_GPU]="$2"; shift 2 ;;
    --global-batch-size)     OVERRIDES[PRESET_GLOBAL_BATCH_SIZE]="$2"; shift 2 ;;
    --rollout-batch-size)    OVERRIDES[PRESET_ROLLOUT_BATCH_SIZE]="$2"; shift 2 ;;
    --attn-impl)             OVERRIDES[PRESET_ATTN_IMPL]="$2"; shift 2 ;;
    --recompute-granularity) OVERRIDES[HW_RECOMPUTE_GRANULARITY]="$2"; shift 2 ;;
    --recompute-method)      OVERRIDES[HW_RECOMPUTE_METHOD]="$2"; shift 2 ;;
    --recompute-num-layers)  OVERRIDES[HW_RECOMPUTE_NUM_LAYERS]="$2"; shift 2 ;;
    --no-cpu-offload)        OVERRIDES[PRESET_CPU_OFFLOAD_FLAGS]=""; shift ;;
    --cpu-offload)           OVERRIDES[PRESET_CPU_OFFLOAD_FLAGS]="--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer"; shift ;;
    --)                      shift; break ;;
    *) echo "[err] unknown arg: $1 (see $0 --help)" >&2; exit 1 ;;
  esac
done

# Current default: H200 42-node P3-derived config for kaynzhang_077 134K SFT.
# Explicit CLI/env values still override these.
: "${V4_FLEET:=h200_k8s_42node}"
: "${V4_SCALE:=tp8pp7cp6ep8}"
: "${V4_WORKLOAD:=sft_kaynzhang_077_134k_3epoch}"
export V4_FLEET V4_SCALE V4_WORKLOAD

# ---------- layered source ---------------------------------------------------
# env.sh handles: fleet → base → hw → scale → workload.
source "$CLUSTER_DIR/env.sh"

# CLI overrides (applied last in the source chain)
for k in "${!OVERRIDES[@]}"; do
  printf '[info] override: %s=%s\n' "$k" "${OVERRIDES[$k]}"
  export "$k=${OVERRIDES[$k]}"
done

# Preset can set PRESET_CPU_OFFLOAD_FLAGS="" to disable cpu-offload optimizer (saves Memcpy% but
# costs ~40 GB GPU mem for optimizer state — V4 single-rank exceeds 95 GB, so default keeps offload).
: "${PRESET_CPU_OFFLOAD_FLAGS=--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer}"

# ZeRO-1: shard optimizer state along DP. Only meaningful when DP > 1
# (64 GPU TP=8 PP=8 CP=1 → DP=1, no-op; 128 GPU same shape → DP=2, ~20 GB Adam/rank).
DIST_OPT_FLAGS=""
if [[ "${PRESET_USE_DIST_OPT:-0}" == "1" ]]; then
  DIST_OPT_FLAGS="--use-distributed-optimizer"
fi

# Recompute flags depend on granularity: selective rejects --recompute-method; full needs it; none omits both.
# selective + HW_RECOMPUTE_MODULES emits --recompute-modules (Megatron defaults to core_attn when omitted;
# we want explicit moe_act/layernorm/mla_up_proj subset for cheap-recompute big-activation wins).
case "${HW_RECOMPUTE_GRANULARITY:-full}" in
  none|"") HW_RECOMPUTE_FLAGS="" ;;
  selective)
    HW_RECOMPUTE_FLAGS="--recompute-granularity selective"
    if [[ -n "${HW_RECOMPUTE_MODULES:-}" ]]; then
      HW_RECOMPUTE_FLAGS="$HW_RECOMPUTE_FLAGS --recompute-modules $HW_RECOMPUTE_MODULES"
    fi
    ;;
  *) HW_RECOMPUTE_FLAGS="--recompute-granularity $HW_RECOMPUTE_GRANULARITY --recompute-method $HW_RECOMPUTE_METHOD --recompute-num-layers $HW_RECOMPUTE_NUM_LAYERS" ;;
esac

# Optional --tool-key (datasets with separate tool spec column, e.g. albaliang agent SFT).
# Empty by default; presets needing it set PRESET_TOOL_KEY=<column-name>.
SFT_TOOL_KEY_FLAGS=""
if [[ -n "${PRESET_TOOL_KEY:-}" ]]; then
  SFT_TOOL_KEY_FLAGS="--tool-key $PRESET_TOOL_KEY"
fi

SFT_DEBUG_FLAGS=""
if [[ "${PRESET_DEBUG_TRAIN_ONLY:-1}" == "1" ]]; then
  SFT_DEBUG_FLAGS="--debug-train-only"
fi

DUMP_DETAILS_FLAGS=""

# Miles supports either explicit rollout count or epoch-derived rollout count.
# Do not emit both: when both are present Miles ignores num_epoch, which is
# surprising for post-train configs that are meant to be epoch-aligned.
SFT_LENGTH_FLAGS=""
if [[ -n "${PRESET_NUM_EPOCH:-}" ]]; then
  SFT_LENGTH_FLAGS="--num-epoch $PRESET_NUM_EPOCH"
elif [[ -n "${PRESET_NUM_ROLLOUT:-}" ]]; then
  SFT_LENGTH_FLAGS="--num-rollout $PRESET_NUM_ROLLOUT"
else
  echo "[err] either PRESET_NUM_ROLLOUT or PRESET_NUM_EPOCH must be set" >&2
  exit 1
fi

SEQ_LENGTH_FLAGS=""
if [[ -n "${HW_SEQ_LENGTH:-}" ]]; then
  SEQ_LENGTH_FLAGS="--seq-length $HW_SEQ_LENGTH"
fi

# Optional --lr-warmup-iters. Megatron's default with this unset is 0; we only emit
# the flag when caller asked for warmup so existing presets stay byte-for-byte equal.
LR_WARMUP_FLAGS=""
if [[ "${PRESET_LR_WARMUP_ITERS:-0}" -gt 0 ]]; then
  LR_WARMUP_FLAGS="--lr-warmup-iters $PRESET_LR_WARMUP_ITERS"
fi

# Pipeline layer layout. Two modes:
#   1. PRESET_PIPELINE_LAYOUT set → emit --pipeline-model-parallel-layout <string> (Megatron
#      explicit layout, e.g. 'Etttt|(tttttt|)*6,tttL' for 43-layer V4 PP=8). Mutually exclusive
#      with decoder-first/last in Megatron; we DROP those when layout is set.
#   2. Otherwise → emit --decoder-first/last-pipeline-num-layers (existing behavior).
#
# Layout-special chars (| ( ) *) get backslash-escaped, then wrapped in single quotes
# inside the launch script. Reason: ray job submit does NOT shlex.quote argv when joining
# back to a shell command on the cluster side; without escaping, sh parses '(' as syntax
# error. The single-quotes survive heredoc text-substitution; the backslashes survive sh's
# parse on cluster (\|→|, \(→(, etc.) so argparse + Megatron see the clean layout string.
if [[ -n "${PRESET_PIPELINE_LAYOUT:-}" ]]; then
  ESCAPED_LAYOUT=$(printf '%s' "$PRESET_PIPELINE_LAYOUT" | sed 's/[|()*]/\\&/g')
  PIPELINE_PP_FLAGS="--pipeline-model-parallel-layout '$ESCAPED_LAYOUT'"
else
  PIPELINE_PP_FLAGS="--decoder-first-pipeline-num-layers $PRESET_DECODER_FIRST_PIPELINE_NUM_LAYERS --decoder-last-pipeline-num-layers $PRESET_DECODER_LAST_PIPELINE_NUM_LAYERS"
fi

# Virtual pipeline parallel. Two paths in Megatron (mutually exclusive — see arguments.py:503):
#   1. Layout mode (PRESET_PIPELINE_LAYOUT set): VPP is auto-derived as
#      num_stages_in_layout / pipeline_model_parallel_size. PRESET_VPP is documentation
#      only; we DO NOT emit --num-virtual-stages-per-pipeline-rank (Megatron asserts).
#   2. Decoder-first/last mode: PRESET_VPP >= 2 emits --num-virtual-stages-per-pipeline-rank.
#      Won't work for 43-layer V4 due to divisibility (see §6.5), but keep for future models.
VPP_FLAGS=""
if [[ -z "${PRESET_PIPELINE_LAYOUT:-}" ]] && [[ "${PRESET_VPP:-1}" -gt 1 ]]; then
  VPP_FLAGS="--num-virtual-stages-per-pipeline-rank $PRESET_VPP"
fi

# EP all-to-all overlap (batch-level MoE A2A overlapped with computation). Requires:
#   - VPP enabled when PP > 1 (we satisfy via layout)
#   - EP > 1 (we have EP=8)
#   - --moe-token-dispatcher-type in [alltoall, flex] (V4 uses alltoall)
#   - torch >= 2.6.0 (image torch 2.12 satisfies)
# Megatron arguments.py:941 warns that on Hopper (H20) combining TP/CP with EP overlap is
# suboptimal — TP wants CUDA_DEVICE_MAX_CONNECTIONS=1, EP overlap prefers 32. Keep =1 for
# now (TP=8 is heavier in our config); revisit only if EP overlap shows minimal gain.
EP_OVERLAP_FLAGS=""
if [[ "${PRESET_EP_OVERLAP:-0}" == "1" ]]; then
  EP_OVERLAP_FLAGS="--overlap-moe-expert-parallel-comm --delay-wgrad-compute"
fi

# DeepEP backend (kernel-level NVL + IB pipelining inside MoE a2a). Different from EP_OVERLAP
# which is schedule-level (1F1B combined). DeepEP overlaps inside the dispatcher kernel and
# doesn't need attention to expose backward_dw.
# Switches dispatcher: V4 model script sets --moe-token-dispatcher-type alltoall in MODEL_ARGS;
# this flag emits flex + enable-deepep AFTER it in argparse (last-wins). Default SM budget = 20
# (Megatron transformer_config.moe_deepep_num_sms default); raise via PRESET_MOE_DEEPEP_NUM_SMS.
DEEPEP_FLAGS=""
if [[ "${PRESET_MOE_DEEPEP:-0}" == "1" ]]; then
  # --moe-enable-deepep is deprecated in mcore 0.16 (auto-rewrites to backend=deepep with
  # a per-rank warning); use the current --moe-flex-dispatcher-backend deepep directly.
  DEEPEP_FLAGS="--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep"
  if [[ -n "${PRESET_MOE_DEEPEP_NUM_SMS:-}" ]]; then
    DEEPEP_FLAGS="$DEEPEP_FLAGS --moe-deepep-num-sms $PRESET_MOE_DEEPEP_NUM_SMS"
  fi
fi

# MoE router dtype (Megatron warns at num_experts>=32 without fp32; DeepEP token_dispatcher.py:1178
# also emits "DeepEP only supports float32 probs" on bf16 path → internal cast overhead).
# Set PRESET_MOE_ROUTER_DTYPE=fp32 to enable. Off by default to keep baselines byte-equal.
ROUTER_DTYPE_FLAGS=""
if [[ -n "${PRESET_MOE_ROUTER_DTYPE:-}" ]]; then
  ROUTER_DTYPE_FLAGS="--moe-router-dtype $PRESET_MOE_ROUTER_DTYPE"
fi

# Launch-overhead reducers. This workload is overhead/launch-bound (94% of credited FLOPs are
# phantom dense attention; ~128 microbatches x 43 MoE layers x 336 ranks per step), so cutting
# CPU-side jitter and kernel-launch count moves measured TFLOPS ~1:1.
#   - manual-gc: aligns Python GC across all ranks; otherwise one rank's stop-the-world GC
#     straggles the whole 7-stage pipeline every step. (Megatron MoE perf recipe default.)
#   - router/permute fusion: collapse MoE router top-k/softmax and permute/unpermute into single
#     kernels (needs TE>=2.1; image has 2.10). Zero-comm, numerics-safe.
# Default OFF to keep A/B baselines byte-equal; flip to 1 in the workload env once validated.
MANUAL_GC_FLAGS=""
if [[ "${PRESET_MANUAL_GC:-0}" == "1" ]]; then
  MANUAL_GC_FLAGS="--manual-gc --manual-gc-interval ${PRESET_MANUAL_GC_INTERVAL:-10}"
fi
MOE_FUSION_FLAGS=""
if [[ "${PRESET_MOE_FUSION:-0}" == "1" ]]; then
  # NOTE: --moe-router-fusion only supports softmax/sigmoid score functions; V4 uses
  # sqrtsoftplus -> "score_function must be softmax or sigmoid for router fusion". So we
  # enable only --moe-permute-fusion (permute/unpermute around grouped-GEMM, score-agnostic).
  MOE_FUSION_FLAGS="--moe-permute-fusion"
fi

# Profile flags are finalized below after RUN_ID is set (needs $SAVE_DIR for tb_dir).
PROFILE_FLAGS=""
PROFILE_TBDIR=""

RAY_JOB_WAIT_FLAGS=""
if (( NO_WAIT == 1 )); then
  RAY_JOB_WAIT_FLAGS="--no-wait"
fi
RAY_JOB_ENTRYPOINT_FLAGS=""
if [[ -n "${V4_RAY_ENTRYPOINT_RESOURCES:-}" ]]; then
  RAY_JOB_ENTRYPOINT_FLAGS="--entrypoint-resources '$V4_RAY_ENTRYPOINT_RESOURCES'"
fi
RAY_JOB_ENTRYPOINT_ENV_PREFIX=""
if [[ -n "${V4_RAY_DRIVER_CUDA_VISIBLE_DEVICES:-}" || -n "${V4_RAY_DRIVER_NOSET_CUDA_VISIBLE_DEVICES:-}" ]]; then
  RAY_JOB_ENTRYPOINT_ENV_PREFIX="env"
  [[ -n "${V4_RAY_DRIVER_CUDA_VISIBLE_DEVICES:-}" ]] && RAY_JOB_ENTRYPOINT_ENV_PREFIX+=" CUDA_VISIBLE_DEVICES=$V4_RAY_DRIVER_CUDA_VISIBLE_DEVICES"
  [[ -n "${V4_RAY_DRIVER_NOSET_CUDA_VISIBLE_DEVICES:-}" ]] && RAY_JOB_ENTRYPOINT_ENV_PREFIX+=" RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=$V4_RAY_DRIVER_NOSET_CUDA_VISIBLE_DEVICES"
fi

# ---------- preflight ----------------------------------------------------------
source "$CLUSTER_DIR/lib/preflight.sh"
if [[ "$V4_WORKLOAD" == "sft_smoke" ]]; then
  preflight_64gpu_strict || exit 1
else
  preflight_64gpu || exit 1
fi

# ---------- run id / save dir --------------------------------------------------
RUN_ID="${PRESET_RUN_ID_PREFIX}-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$V4_OUT/$RUN_ID"
CKPT_REF_LOAD_DIR="${PRESET_REF_LOAD_DIR:-$V4_TORCH_DIST}"
# Throughput-probe shortcut: PRESET_REF_LOAD_DIR=none skips --ref-load (random init,
# no dist-ckpt reshard) -- tok/s is weight-value-independent, so config sweeps run fast.
REF_LOAD_FLAGS="--ref-load $CKPT_REF_LOAD_DIR"
if [[ "${PRESET_REF_LOAD_DIR:-}" == "none" ]]; then REF_LOAD_FLAGS=""; CKPT_REF_LOAD_DIR="(none: random init)"; fi
CKPT_LOAD_DIR="${PRESET_LOAD_DIR:-$SAVE_DIR/checkpoints}"
mkdir -p "$SAVE_DIR"
DUMP_DETAILS_FLAGS=""
if [[ "${PRESET_DUMP_DETAILS:-1}" == "1" ]]; then
  DUMP_DETAILS_FLAGS="--dump-details $SAVE_DIR/dump_details"
fi
# SFT rollout/train overlap: prefetch next batch's CPU tokenization during the current
# train step (hides the ~30% idle-GPU train_wait). Safe only for weight-independent SFT rollout.
ASYNC_PREFETCH_FLAGS=""
if [[ "${PRESET_ASYNC_ROLLOUT_PREFETCH:-0}" == "1" ]]; then
  ASYNC_PREFETCH_FLAGS="--async-rollout-prefetch"
fi
echo "[info] config   : control=${V4_CONTROL:-<legacy>} fleet=$V4_FLEET scale=$V4_SCALE workload=$V4_WORKLOAD"
echo "[info] cluster  : $V4_CLUSTER_NAME ($V4_GPU_MODEL × $V4_NUM_NODES nodes × $V4_NUM_GPUS_PER_NODE gpus)"
echo "[info] run id    : $RUN_ID"
echo "[info] save dir  : $SAVE_DIR"
echo "[info] ref load  : $CKPT_REF_LOAD_DIR"
echo "[info] load dir  : $CKPT_LOAD_DIR"
echo "[info] dashboard : http://$V4_RAY_HEAD_IP:$V4_DASHBOARD_PORT"

# Finalize profile flags (needs SAVE_DIR).
if (( PROFILE_ENABLED == 1 )); then
  : "${PROFILE_STEP_START:=5}"
  : "${PROFILE_STEP_END:=$((PROFILE_STEP_START + 1))}"   # active=1 (PP=4 trace-safe)
  PROFILE_TBDIR="$SAVE_DIR/profiler_traces"
  mkdir -p "$PROFILE_TBDIR"
  PROFILE_FLAGS="--use-pytorch-profiler --profile-step-start $PROFILE_STEP_START --profile-step-end $PROFILE_STEP_END --profile-target train_overall --tensorboard-dir $PROFILE_TBDIR"
  echo "[info] profile  : ON  active steps [$PROFILE_STEP_START,$PROFILE_STEP_END)  tb_dir=$PROFILE_TBDIR"
fi

# Wandb flags (workload sets PRESET_USE_WANDB=1 to enable).
WANDB_FLAGS=""
if [[ "${PRESET_USE_WANDB:-0}" == "1" ]]; then
  : "${PRESET_WANDB_PROJECT:?PRESET_WANDB_PROJECT required when PRESET_USE_WANDB=1}"
  WANDB_DIR="$SAVE_DIR/wandb"
  mkdir -p "$WANDB_DIR"
  if [[ -z "${PRESET_WANDB_API_KEY_FILE:-}" ]]; then
    if [[ -n "${PRESET_WANDB_API_KEY:-${WANDB_API_KEY:-}}" ]]; then
      export PRESET_WANDB_API_KEY_FILE="$SAVE_DIR/.wandb_api_key"
      umask 077
      printf '%s' "${PRESET_WANDB_API_KEY:-$WANDB_API_KEY}" > "$PRESET_WANDB_API_KEY_FILE"
    else
      echo "[err] PRESET_WANDB_API_KEY_FILE or PRESET_WANDB_API_KEY required when PRESET_USE_WANDB=1" >&2
      exit 1
    fi
  fi
  [[ -f "$PRESET_WANDB_API_KEY_FILE" ]] || {
    echo "[err] W&B key file not found: $PRESET_WANDB_API_KEY_FILE" >&2
    exit 1
  }
  WANDB_FLAGS="--use-wandb --wandb-project $PRESET_WANDB_PROJECT --wandb-dir $WANDB_DIR --wandb-group $RUN_ID"
  [[ -n "${PRESET_WANDB_TEAM:-}" ]] && WANDB_FLAGS="$WANDB_FLAGS --wandb-team $PRESET_WANDB_TEAM"
  echo "[info] wandb    : ON  project=$PRESET_WANDB_PROJECT team=${PRESET_WANDB_TEAM:-<default>} group=$RUN_ID key_source=file"
  echo "[info]            MFU on wandb: derive as actor_train_tflops / ${HW_GPU_PEAK_TFLOPS_BF16:-148} (${V4_GPU_MODEL:-gpu} BF16 peak)"
fi

# ---------- EFA / aws-ofi-nccl env -------------------------------------------
# When V4_EFA_ENABLE=1 (set in the fleet env), point NCCL at the fsx-staged
# aws-ofi-nccl plugin + libfabric so inter-node collectives use EFA RDMA instead
# of the TCP-socket fallback. These values are baked into the generated launch
# script and injected into the Ray runtime_env so all ranks pick them up.
V4_EFA_LD_LIBRARY_PATH=""
V4_NCCL_NET_PLUGIN_VAL=""
V4_FI_PROVIDER_VAL=""
V4_FI_EFA_USE_DEVICE_RDMA_VAL=""
V4_FI_EFA_FORK_SAFE_VAL=""
V4_NCCL_PROTO_VAL=""
if [[ "${V4_EFA_ENABLE:-0}" == "1" ]]; then
  V4_EFA_LD_LIBRARY_PATH="$V4_EFA_ROOT/opt/amazon/ofi-nccl/lib:$V4_EFA_ROOT/opt/amazon/efa/lib:$V4_EFA_ROOT/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
  V4_NCCL_NET_PLUGIN_VAL="$V4_EFA_ROOT/opt/amazon/ofi-nccl/lib/libnccl-net.so"
  V4_FI_PROVIDER_VAL="${V4_FI_PROVIDER:-efa}"
  V4_FI_EFA_USE_DEVICE_RDMA_VAL="${V4_FI_EFA_USE_DEVICE_RDMA:-1}"
  V4_FI_EFA_FORK_SAFE_VAL="${V4_FI_EFA_FORK_SAFE:-1}"
  V4_NCCL_PROTO_VAL="${V4_NCCL_PROTO:-simple}"
  # hostNetwork pods have per-node NIC names (enp74s0/enp75s0/...); prefix-match the
  # routable host NIC for NCCL OOB bootstrap and avoid the eni*/veth link-local ones.
  : "${V4_NCCL_SOCKET_IFNAME:=enp}"
  echo "[info] EFA      : ON  plugin=$V4_NCCL_NET_PLUGIN_VAL provider=$V4_FI_PROVIDER_VAL ifname=$V4_NCCL_SOCKET_IFNAME"
else
  echo "[info] EFA      : OFF (NCCL will fall back to TCP sockets)"
fi

# ---------- FP8 (TransformerEngine MoE grouped GEMM) -------------------------
# HW_FP8_ENABLED=1 routes the MoE expert GEMMs (bulk of the FLOPs) through TE
# blockwise FP8 (~2x matmul) while the tilelang sparse-MLA attention stays bf16.
# Mirrors the tested smoke/run_stage_a_fp8_smoke.sh recipe. Params stay bf16
# (no --fp8-param-gather) so the deepseek_v4 bf16 weight assert holds.
FP8_FLAGS=""
V4_FP8_NVTE_VALUE=""
if [[ "${HW_FP8_ENABLED:-0}" == "1" ]]; then
  FP8_FLAGS="--transformer-impl transformer_engine --bf16 --fp8-format ${HW_FP8_FORMAT:-e4m3} --fp8-recipe ${HW_FP8_RECIPE:-blockwise}"
  V4_FP8_NVTE_VALUE="1"
  echo "[info] FP8      : ON  format=${HW_FP8_FORMAT:-e4m3} recipe=${HW_FP8_RECIPE:-blockwise} (MoE grouped GEMM)"
else
  echo "[info] FP8      : OFF (bf16 GEMMs)"
fi

# ---------- generate in-container launch script ------------------------------
LAUNCH=$SAVE_DIR/launch_in_container.sh
cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
set -e
cd $V4_MILES
source scripts/models/deepseek-v4-flash.sh

CKPT_ARGS=(
  --hf-checkpoint  $V4_BF16_DIR
  $REF_LOAD_FLAGS
  --load           $CKPT_LOAD_DIR
  --save           $SAVE_DIR/checkpoints
  --save-interval  $PRESET_SAVE_INTERVAL
  --save-retain-interval $PRESET_SAVE_RETAIN_INTERVAL
)
# Workaround for module 3 Problem 8 — skip optimizer state save to avoid the
# dist_checkpointing async D2H cudaErrorInvalidValue under 64K + long runs.
# SFT terminal state does not need optimizer state for resume.
[[ "$NO_SAVE_OPTIM" == "1" ]] && CKPT_ARGS+=(--no-save-optim)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    $V4_SFT_DATA
  --input-key      messages
  --rollout-shuffle
  $SFT_LENGTH_FLAGS
  --rollout-batch-size     $PRESET_ROLLOUT_BATCH_SIZE
  --global-batch-size      $PRESET_GLOBAL_BATCH_SIZE

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --sft-only
  $SFT_DEBUG_FLAGS

  --loss-mask-type deepseek_v4
  $SFT_TOOL_KEY_FLAGS
)

PERF_ARGS=(
  --tensor-model-parallel-size $PRESET_TP
  --sequence-parallel
  --pipeline-model-parallel-size $PRESET_PP
  $PIPELINE_PP_FLAGS
  $VPP_FLAGS
  --context-parallel-size $PRESET_CP
  --expert-model-parallel-size $PRESET_EP
  --expert-tensor-parallel-size $PRESET_ETP
  $EP_OVERLAP_FLAGS
  $DEEPEP_FLAGS
  $ROUTER_DTYPE_FLAGS
  $MANUAL_GC_FLAGS
  $MOE_FUSION_FLAGS

  $HW_RECOMPUTE_FLAGS

  --micro-batch-size 1
  --use-dynamic-batch-size
  $SEQ_LENGTH_FLAGS
  --max-tokens-per-gpu $HW_MAX_TOKENS_PER_GPU
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr $PRESET_LR
  --lr-decay-style $PRESET_LR_DECAY_STYLE
  $LR_WARMUP_FLAGS
  --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.95
  $PRESET_CPU_OFFLOAD_FLAGS
  $DIST_OPT_FLAGS
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout    0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --model-name deepseekv4
  --qkv-format thd
  --moe-router-freeze-gate
  --freeze-e-score-correction-bias
  $FP8_FLAGS
  --update-weight-buffer-size 1073741824
  --train-memory-margin-bytes 3221225472
  # Raise the default 10-min NCCL/PG watchdog so slow init paths (e.g. optimizer
  # cpu-offload pinned-buffer alloc, dist-ckpt reshard reads) don't trip a 600s
  # collective timeout during startup before training even begins.
  --distributed-timeout-minutes ${HW_DIST_TIMEOUT_MINUTES:-60}

  --actor-num-nodes $V4_NUM_NODES
  --actor-num-gpus-per-node $V4_NUM_GPUS_PER_NODE
  --num-gpus-per-node $V4_NUM_GPUS_PER_NODE
  --colocate
  --no-offload-train
  --no-offload-rollout
  --use-fault-tolerance
  $ASYNC_PREFETCH_FLAGS
  $DUMP_DETAILS_FLAGS
  $PROFILE_FLAGS
  $WANDB_FLAGS
)

WANDB_API_KEY_VALUE=""
if [[ "${PRESET_USE_WANDB:-0}" == "1" ]]; then
  WANDB_API_KEY_FILE="${PRESET_WANDB_API_KEY_FILE:-}"
  if [[ -n "\$WANDB_API_KEY_FILE" && -f "\$WANDB_API_KEY_FILE" ]]; then
    WANDB_API_KEY_VALUE="\$(tr -d '\r\n' < "\$WANDB_API_KEY_FILE")"
  elif [[ -n "\${WANDB_API_KEY:-}" ]]; then
    WANDB_API_KEY_VALUE="\$WANDB_API_KEY"
  else
    echo "[err] W&B enabled but no key available in key file or WANDB_API_KEY" >&2
    exit 1
  fi
fi

RUNTIME_ENV="\$(PYTHONPATH_VALUE="$V4_RUNTIME_PYTHONPATH" \\
  MASTER_ADDR_VALUE="$V4_TRAINING_MASTER_IP" \\
  NCCL_NVLS_ENABLE_VALUE="$HW_NCCL_NVLS_ENABLE" \\
  GLOO_SOCKET_IFNAME_VALUE="${V4_GLOO_SOCKET_IFNAME:-}" \\
  NCCL_SOCKET_IFNAME_VALUE="${V4_NCCL_SOCKET_IFNAME:-}" \\
  EFA_LD_LIBRARY_PATH_VALUE="$V4_EFA_LD_LIBRARY_PATH" \\
  NCCL_NET_PLUGIN_VALUE="$V4_NCCL_NET_PLUGIN_VAL" \\
  FI_PROVIDER_VALUE="$V4_FI_PROVIDER_VAL" \\
  FI_EFA_USE_DEVICE_RDMA_VALUE="$V4_FI_EFA_USE_DEVICE_RDMA_VAL" \\
  FI_EFA_FORK_SAFE_VALUE="$V4_FI_EFA_FORK_SAFE_VAL" \\
  NCCL_PROTO_VALUE="$V4_NCCL_PROTO_VAL" \\
  NCCL_MAX_NCHANNELS_VALUE="${V4_NCCL_MAX_NCHANNELS:-}" \\
  NVTE_FP8_BLOCK_SCALING_FP32_SCALES_VALUE="$V4_FP8_NVTE_VALUE" \\
  MEGATRON_SPARSE_ATTN_IMPL_VALUE="$PRESET_ATTN_IMPL" \\
  MILES_ROLLOUT_MANAGER_RESOURCES_VALUE='${V4_ROLLOUT_MANAGER_RESOURCES:-}' \\
  PYTORCH_CUDA_ALLOC_CONF_VALUE="${HW_PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \\
  CUDA_LAUNCH_BLOCKING_VALUE="${HW_CUDA_LAUNCH_BLOCKING:-0}" \\
  TORCH_USE_CUDA_DSA_VALUE="${HW_TORCH_USE_CUDA_DSA:-0}" \\
  WANDB_API_KEY_VALUE="\$WANDB_API_KEY_VALUE" \\
  python3 - <<'PY'
import json
import os

env = {
    "PYTHONPATH": os.environ["PYTHONPATH_VALUE"],
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MASTER_ADDR": os.environ["MASTER_ADDR_VALUE"],
    "MILES_DSV4_THINKING_MODE": "chat",
    "MILES_DSV4_DROP_THINKING": "0",
    # Persist tilelang @jit kernel-compile cache to fsx (default /root/.tilelang is
    # container-ephemeral -> a pod restart re-pays the ~40min first-step recompile for
    # each new sequence shape). fsx path persists across restarts and is shared by all nodes.
    "TILELANG_CACHE_DIR": "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/.cache/tilelang",
    "NCCL_NVLS_ENABLE": os.environ["NCCL_NVLS_ENABLE_VALUE"],
    "LD_PRELOAD": "/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so",
    "MEGATRON_SPARSE_ATTN_IMPL": os.environ["MEGATRON_SPARSE_ATTN_IMPL_VALUE"],
    "PYTORCH_CUDA_ALLOC_CONF": os.environ["PYTORCH_CUDA_ALLOC_CONF_VALUE"],
    "CUDA_LAUNCH_BLOCKING": os.environ["CUDA_LAUNCH_BLOCKING_VALUE"],
    "TORCH_USE_CUDA_DSA": os.environ["TORCH_USE_CUDA_DSA_VALUE"],
}
gloo_socket_ifname = os.environ.get("GLOO_SOCKET_IFNAME_VALUE")
if gloo_socket_ifname:
    env["GLOO_SOCKET_IFNAME"] = gloo_socket_ifname
nccl_socket_ifname = os.environ.get("NCCL_SOCKET_IFNAME_VALUE")
if nccl_socket_ifname:
    env["NCCL_SOCKET_IFNAME"] = nccl_socket_ifname
# EFA / aws-ofi-nccl: only injected when the fleet enables it (values non-empty).
for _src, _dst in (
    ("EFA_LD_LIBRARY_PATH_VALUE", "LD_LIBRARY_PATH"),
    ("NCCL_NET_PLUGIN_VALUE", "NCCL_NET_PLUGIN"),
    ("FI_PROVIDER_VALUE", "FI_PROVIDER"),
    ("FI_EFA_USE_DEVICE_RDMA_VALUE", "FI_EFA_USE_DEVICE_RDMA"),
    ("FI_EFA_FORK_SAFE_VALUE", "FI_EFA_FORK_SAFE"),
    ("NCCL_PROTO_VALUE", "NCCL_PROTO"),
    ("NCCL_MAX_NCHANNELS_VALUE", "NCCL_MAX_NCHANNELS"),
    ("NVTE_FP8_BLOCK_SCALING_FP32_SCALES_VALUE", "NVTE_FP8_BLOCK_SCALING_FP32_SCALES"),
):
    _v = os.environ.get(_src)
    if _v:
        env[_dst] = _v
rollout_manager_resources = os.environ.get("MILES_ROLLOUT_MANAGER_RESOURCES_VALUE")
if rollout_manager_resources:
    env["MILES_ROLLOUT_MANAGER_RESOURCES"] = rollout_manager_resources
wandb_key = os.environ.get("WANDB_API_KEY_VALUE")
if wandb_key:
    env["WANDB_API_KEY"] = wandb_key
print(json.dumps({"env_vars": env}))
PY
)"

ray job submit --address=http://127.0.0.1:$V4_DASHBOARD_PORT \\
   $RAY_JOB_WAIT_FLAGS \\
   $RAY_JOB_ENTRYPOINT_FLAGS \\
   --runtime-env-json="\$RUNTIME_ENV" \\
   -- $RAY_JOB_ENTRYPOINT_ENV_PREFIX python3 train.py \\
   "\${MODEL_ARGS[@]}" \\
   "\${CKPT_ARGS[@]}" \\
   "\${SFT_ARGS[@]}" \\
   "\${OPTIMIZER_ARGS[@]}" \\
   "\${PERF_ARGS[@]}" \\
   "\${MISC_ARGS[@]}"
EOF

chmod +x "$LAUNCH"
echo "[info] launch script: $LAUNCH"

if (( DRY_RUN == 1 )); then
  echo "[info] --dry-run: launch_in_container.sh generated, NOT submitting to ray"
  exit 0
fi

echo
echo "=== submit ray job (live logs mirrored to $SAVE_DIR/job.log) ==="
case "${V4_SUBMIT_MODE:-ssh}" in
  k8s)
    K8S_NAMESPACE="${V4_K8S_NAMESPACE:-ray-system}"
    K8S_HEAD_DEPLOY="${V4_K8S_HEAD_DEPLOY:-ray-head-gpu-node-64}"
    K8S_HEAD_CONTAINER="${V4_K8S_HEAD_CONTAINER:-}"
    if [[ -n "$K8S_HEAD_CONTAINER" ]]; then
      kubectl exec -n "$K8S_NAMESPACE" "deploy/$K8S_HEAD_DEPLOY" -c "$K8S_HEAD_CONTAINER" -- bash "$LAUNCH" 2>&1 | tee "$SAVE_DIR/job.log"
    else
      kubectl exec -n "$K8S_NAMESPACE" "deploy/$K8S_HEAD_DEPLOY" -- bash "$LAUNCH" 2>&1 | tee "$SAVE_DIR/job.log"
    fi
    ;;
  ssh|"")
    ssh "root@$V4_RAY_HEAD_IP" "docker exec $V4_CONTAINER bash $LAUNCH" 2>&1 | tee "$SAVE_DIR/job.log"
    ;;
  *)
    echo "[err] unknown V4_SUBMIT_MODE=${V4_SUBMIT_MODE:-}" >&2
    exit 1
    ;;
esac

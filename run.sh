#!/usr/bin/env bash
# 统一启动入口。
#
# 用法 (任选):
#   bash run.sh
#   # 默认等价于当前 H200 42 节点 P3 派生配置:
#   #   --fleet h200_k8s_42node --scale tp8pp7cp6ep8 --workload sft_kaynzhang_077_134k_3epoch
#   bash run.sh --control current --fleet h20_16node --scale tp8pp16ep8_layout --workload sft_prod
#   V4_CONTROL=current V4_FLEET=h20_16node V4_SCALE=tp8pp16ep8_layout V4_WORKLOAD=sft_prod bash run.sh
#
# 常用覆盖项 (每项对应一个 PRESET_* / HW_* 变量):
#   --num-rollout N           训练步数
#   --num-epoch N             prompt 数据完整遍历的轮数 (Miles 据此推导 rollout 数)
#   --lr X                    学习率
#   --lr-decay-style S        constant / cosine / linear
#   --save-interval N         每 N 步保存一次 ckpt
#   --save-retain-interval N  每 N 步保留一个永久 ckpt (滚动)
#   --seq-length N            Megatron 序列长度
#   --max-tokens-per-gpu N    每个 CP rank 的 token 上限 (用显存换 batch)
#   --global-batch-size N
#   --attn-impl IMPL          tilelang / dense
#   --recompute-granularity G none / selective / full
#   --recompute-method M      uniform / block (granularity=full 时)
#   --recompute-num-layers N
#   --no-cpu-offload          关闭 optimizer CPU offload (用于显存容量试探)
#
# --dry-run: 只生成 launch_in_container.sh, 但不提交到 ray。
# --no-wait: 提交 ray job 后立即返回, 不再持续输出日志直到任务结束。
# --submit-mode ssh|k8s: 通过 SSH/docker 运行生成的启动脚本, 或
#   通过 kubectl exec 在配置好的 Ray head pod 中运行。
#
# 加载顺序 (后加载的覆盖先加载的):
#   cluster/fleet/$V4_FLEET.env  →  cluster/control/$V4_CONTROL.env  →  cluster/base.env  →
#   cluster/hw/$V4_GPU_MODEL.env  →  cluster/scale/$V4_SCALE.env  →
#   cluster/workload/$V4_WORKLOAD.env  →  CLI 覆盖项

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CLUSTER_DIR="$REPO_ROOT/cluster"

# ---------- 解析参数 -----------------------------------------------------------
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

# 当前默认值: 用于 kaynzhang_077 134K SFT 的 H200 42 节点 P3 派生配置。
# 显式给出的 CLI/env 取值仍会覆盖这些默认值。
: "${V4_FLEET:=h200_k8s_42node}"
: "${V4_SCALE:=tp8pp7cp6ep8}"
: "${V4_WORKLOAD:=sft_kaynzhang_077_134k_3epoch}"
export V4_FLEET V4_SCALE V4_WORKLOAD

# ---------- 分层 source --------------------------------------------------------
# env.sh 负责处理: fleet → base → hw → scale → workload。
source "$CLUSTER_DIR/env.sh"

# CLI 覆盖项 (在 source 链的最后生效)
for k in "${!OVERRIDES[@]}"; do
  printf '[info] override: %s=%s\n' "$k" "${OVERRIDES[$k]}"
  export "$k=${OVERRIDES[$k]}"
done

# Preset 可设置 PRESET_CPU_OFFLOAD_FLAGS="" 来关闭 cpu-offload optimizer (省下 Memcpy% 但
# optimizer 状态要多占 ~40 GB 显存 — V4 单 rank 会超过 95 GB, 因此默认保持 offload)。
: "${PRESET_CPU_OFFLOAD_FLAGS=--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer}"

# ZeRO-1: 沿 DP 维度切分 optimizer 状态。仅在 DP > 1 时有意义
# (64 GPU TP=8 PP=8 CP=1 → DP=1, 无效果; 128 GPU 同构型 → DP=2, 每 rank 约 20 GB Adam)。
DIST_OPT_FLAGS=""
if [[ "${PRESET_USE_DIST_OPT:-0}" == "1" ]]; then
  DIST_OPT_FLAGS="--use-distributed-optimizer"
fi

# Recompute 标志取决于 granularity: selective 不接受 --recompute-method; full 需要它; none 两者都不传。
# selective + HW_RECOMPUTE_MODULES 会发出 --recompute-modules (省略时 Megatron 默认 core_attn;
# 我们要显式指定 moe_act/layernorm/mla_up_proj 子集, 用廉价重算换大激活显存的收益)。
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

# 可选 --tool-key (用于带独立 tool 规格列的数据集, 例如 albaliang agent SFT)。
# 默认为空; 需要它的 preset 自行设置 PRESET_TOOL_KEY=<column-name>。
SFT_TOOL_KEY_FLAGS=""
if [[ -n "${PRESET_TOOL_KEY:-}" ]]; then
  SFT_TOOL_KEY_FLAGS="--tool-key $PRESET_TOOL_KEY"
fi

SFT_DEBUG_FLAGS=""
if [[ "${PRESET_DEBUG_TRAIN_ONLY:-1}" == "1" ]]; then
  SFT_DEBUG_FLAGS="--debug-train-only"
fi

DUMP_DETAILS_FLAGS=""

# Miles 支持显式指定 rollout 数, 或由 epoch 数推导 rollout 数。
# 不要两者同时传: 两者都存在时 Miles 会忽略 num_epoch, 这对
# 本应按 epoch 对齐的 post-train 配置来说很容易造成意外。
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

# 可选 --lr-warmup-iters。不设置时 Megatron 的默认值为 0; 只有调用方要求 warmup 时
# 才发出该标志, 以保证既有 preset 逐字节不变。
LR_WARMUP_FLAGS=""
if [[ "${PRESET_LR_WARMUP_ITERS:-0}" -gt 0 ]]; then
  LR_WARMUP_FLAGS="--lr-warmup-iters $PRESET_LR_WARMUP_ITERS"
fi

# Pipeline 层布局。两种模式:
#   1. 设置了 PRESET_PIPELINE_LAYOUT → 发出 --pipeline-model-parallel-layout <string> (Megatron
#      显式布局, 例如 43 层 V4 PP=8 用 'Etttt|(tttttt|)*6,tttL')。在 Megatron 中它与
#      decoder-first/last 互斥; 设置了 layout 时我们会丢弃后者。
#   2. 否则 → 发出 --decoder-first/last-pipeline-num-layers (既有行为)。
#
# 布局中的特殊字符 (| ( ) *) 先做反斜杠转义, 再在启动脚本里包进单引号。
# 原因: ray job submit 在集群侧把 argv 拼回 shell 命令时不会做 shlex.quote;
# 不转义的话 sh 会把 '(' 解析成语法错误。单引号能在 heredoc 文本替换中存活;
# 反斜杠能挺过集群上 sh 的解析 (\|→|, \(→( 等), 因此 argparse + Megatron
# 看到的是干净的 layout 字符串。
if [[ -n "${PRESET_PIPELINE_LAYOUT:-}" ]]; then
  ESCAPED_LAYOUT=$(printf '%s' "$PRESET_PIPELINE_LAYOUT" | sed 's/[|()*]/\\&/g')
  PIPELINE_PP_FLAGS="--pipeline-model-parallel-layout '$ESCAPED_LAYOUT'"
else
  PIPELINE_PP_FLAGS="--decoder-first-pipeline-num-layers $PRESET_DECODER_FIRST_PIPELINE_NUM_LAYERS --decoder-last-pipeline-num-layers $PRESET_DECODER_LAST_PIPELINE_NUM_LAYERS"
fi

# 虚拟流水线并行 (VPP)。Megatron 中有两条路径 (互斥 — 见 arguments.py:503):
#   1. Layout 模式 (设置了 PRESET_PIPELINE_LAYOUT): VPP 自动推导为
#      num_stages_in_layout / pipeline_model_parallel_size。PRESET_VPP 仅作文档用途;
#      我们不发出 --num-virtual-stages-per-pipeline-rank (否则 Megatron 会 assert)。
#   2. Decoder-first/last 模式: PRESET_VPP >= 2 时发出 --num-virtual-stages-per-pipeline-rank。
#      因整除性约束对 43 层 V4 不可用 (见 §6.5), 但为未来模型保留。
VPP_FLAGS=""
if [[ -z "${PRESET_PIPELINE_LAYOUT:-}" ]] && [[ "${PRESET_VPP:-1}" -gt 1 ]]; then
  VPP_FLAGS="--num-virtual-stages-per-pipeline-rank $PRESET_VPP"
fi

# EP all-to-all overlap (batch 级 MoE A2A 与计算重叠)。要求:
#   - PP > 1 时启用 VPP (我们通过 layout 满足)
#   - EP > 1 (我们是 EP=8)
#   - --moe-token-dispatcher-type 取值在 [alltoall, flex] 之内 (V4 用 alltoall)
#   - torch >= 2.6.0 (镜像的 torch 2.12 满足)
# Megatron arguments.py:941 警告: 在 Hopper (H20) 上把 TP/CP 与 EP overlap 组合并不
# 理想 — TP 需要 CUDA_DEVICE_MAX_CONNECTIONS=1, 而 EP overlap 偏好 32。暂时保持 =1
# (我们的配置里 TP=8 更重); 仅当 EP overlap 收益甚微时再重新评估。
EP_OVERLAP_FLAGS=""
if [[ "${PRESET_EP_OVERLAP:-0}" == "1" ]]; then
  EP_OVERLAP_FLAGS="--overlap-moe-expert-parallel-comm --delay-wgrad-compute"
fi

# DeepEP 后端 (MoE a2a 内部 kernel 级的 NVL + IB 流水)。不同于 EP_OVERLAP,
# 后者是调度级的 (1F1B combined)。DeepEP 在 dispatcher kernel 内部做 overlap,
# 不需要 attention 暴露 backward_dw。
# 切换 dispatcher: V4 模型脚本在 MODEL_ARGS 里设置 --moe-token-dispatcher-type alltoall;
# 本标志在 argparse 中于其后发出 flex + enable-deepep (后传的生效)。默认 SM 预算 = 20
# (Megatron transformer_config.moe_deepep_num_sms 的默认值); 可通过 PRESET_MOE_DEEPEP_NUM_SMS 调高。
DEEPEP_FLAGS=""
if [[ "${PRESET_MOE_DEEPEP:-0}" == "1" ]]; then
  # --moe-enable-deepep 在 mcore 0.16 中已弃用 (会自动改写为 backend=deepep 并在
  # 每个 rank 上打一条警告); 直接使用现行的 --moe-flex-dispatcher-backend deepep。
  DEEPEP_FLAGS="--moe-token-dispatcher-type flex --moe-flex-dispatcher-backend deepep"
  if [[ -n "${PRESET_MOE_DEEPEP_NUM_SMS:-}" ]]; then
    DEEPEP_FLAGS="$DEEPEP_FLAGS --moe-deepep-num-sms $PRESET_MOE_DEEPEP_NUM_SMS"
  fi
fi

# MoE router dtype (num_experts>=32 且未用 fp32 时 Megatron 会告警; DeepEP token_dispatcher.py:1178
# 在 bf16 路径上也会输出 "DeepEP only supports float32 probs" → 内部 cast 开销)。
# 设置 PRESET_MOE_ROUTER_DTYPE=fp32 启用。默认关闭, 以保持基线逐字节一致。
ROUTER_DTYPE_FLAGS=""
if [[ -n "${PRESET_MOE_ROUTER_DTYPE:-}" ]]; then
  ROUTER_DTYPE_FLAGS="--moe-router-dtype $PRESET_MOE_ROUTER_DTYPE"
fi

# 削减启动开销的开关。该 workload 属于开销/kernel 启动受限型 (计入的 FLOPs 中 94% 是
# 虚算的 dense attention; 每步约 128 个 microbatch x 43 个 MoE 层 x 336 个 rank), 因此削减
# CPU 侧抖动和 kernel 启动次数能近似 1:1 地拉高实测 TFLOPS。
#   - manual-gc: 在所有 rank 间对齐 Python GC; 否则任一 rank 的 stop-the-world GC
#     每一步都会拖慢整条 7 段流水线。(Megatron MoE 性能配方的默认做法。)
#   - router/permute fusion: 把 MoE router top-k/softmax 与 permute/unpermute 折叠成单个
#     kernel (需要 TE>=2.1; 镜像为 2.10)。零通信, 数值安全。
# 默认 OFF 以保持 A/B 基线逐字节一致; 验证通过后在 workload env 中改为 1。
MANUAL_GC_FLAGS=""
if [[ "${PRESET_MANUAL_GC:-0}" == "1" ]]; then
  MANUAL_GC_FLAGS="--manual-gc --manual-gc-interval ${PRESET_MANUAL_GC_INTERVAL:-10}"
fi
MOE_FUSION_FLAGS=""
if [[ "${PRESET_MOE_FUSION:-0}" == "1" ]]; then
  # 注意: --moe-router-fusion 仅支持 softmax/sigmoid 评分函数; V4 用的是
  # sqrtsoftplus -> "score_function must be softmax or sigmoid for router fusion"。因此
  # 只启用 --moe-permute-fusion (grouped-GEMM 前后的 permute/unpermute, 与评分函数无关)。
  MOE_FUSION_FLAGS="--moe-permute-fusion"
fi

# Profile 标志在下方 RUN_ID 确定后才最终生成 (tb_dir 需要 $SAVE_DIR)。
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

# ---------- 预检 (preflight) -----------------------------------------------------
source "$CLUSTER_DIR/lib/preflight.sh"
if [[ "$V4_WORKLOAD" == "sft_smoke" ]]; then
  preflight_64gpu_strict || exit 1
else
  preflight_64gpu || exit 1
fi

# ---------- run id / 保存目录 ----------------------------------------------------
RUN_ID="${PRESET_RUN_ID_PREFIX}-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$V4_OUT/$RUN_ID"
CKPT_REF_LOAD_DIR="${PRESET_REF_LOAD_DIR:-$V4_TORCH_DIST}"
# 吞吐试探捷径: PRESET_REF_LOAD_DIR=none 跳过 --ref-load (随机初始化,
# 不做 dist-ckpt reshard) -- tok/s 与权重数值无关, 因此配置扫描可以跑得很快。
REF_LOAD_FLAGS="--ref-load $CKPT_REF_LOAD_DIR"
if [[ "${PRESET_REF_LOAD_DIR:-}" == "none" ]]; then REF_LOAD_FLAGS=""; CKPT_REF_LOAD_DIR="(none: random init)"; fi
CKPT_LOAD_DIR="${PRESET_LOAD_DIR:-$SAVE_DIR/checkpoints}"
mkdir -p "$SAVE_DIR"
DUMP_DETAILS_FLAGS=""
if [[ "${PRESET_DUMP_DETAILS:-1}" == "1" ]]; then
  DUMP_DETAILS_FLAGS="--dump-details $SAVE_DIR/dump_details"
fi
# SFT rollout/train 重叠: 在当前 train step 期间预取下一个 batch 的 CPU tokenization
# (掩盖约 30% GPU 空闲的 train_wait)。仅对与权重无关的 SFT rollout 安全。
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

# 最终生成 profile 标志 (需要 SAVE_DIR)。
if (( PROFILE_ENABLED == 1 )); then
  : "${PROFILE_STEP_START:=5}"
  : "${PROFILE_STEP_END:=$((PROFILE_STEP_START + 1))}"   # active=1 (PP=4 下 trace 安全)
  PROFILE_TBDIR="$SAVE_DIR/profiler_traces"
  mkdir -p "$PROFILE_TBDIR"
  PROFILE_FLAGS="--use-pytorch-profiler --profile-step-start $PROFILE_STEP_START --profile-step-end $PROFILE_STEP_END --profile-target train_overall --tensorboard-dir $PROFILE_TBDIR"
  echo "[info] profile  : ON  active steps [$PROFILE_STEP_START,$PROFILE_STEP_END)  tb_dir=$PROFILE_TBDIR"
fi

# Wandb 标志 (workload 设置 PRESET_USE_WANDB=1 即启用)。
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

# ---------- EFA / aws-ofi-nccl 环境 --------------------------------------------
# 当 V4_EFA_ENABLE=1 (在 fleet env 中设置) 时, 让 NCCL 指向预置在 fsx 上的
# aws-ofi-nccl 插件 + libfabric, 使跨节点集合通信走 EFA RDMA 而不是
# 回退到 TCP socket。这些值会固化进生成的启动脚本, 并注入 Ray runtime_env,
# 让所有 rank 都能读到。
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
  # hostNetwork pod 的网卡名因节点而异 (enp74s0/enp75s0/...); 用前缀匹配可路由的
  # 宿主机网卡供 NCCL OOB bootstrap 使用, 避开 eni*/veth 这类 link-local 网卡。
  : "${V4_NCCL_SOCKET_IFNAME:=enp}"
  echo "[info] EFA      : ON  plugin=$V4_NCCL_NET_PLUGIN_VAL provider=$V4_FI_PROVIDER_VAL ifname=$V4_NCCL_SOCKET_IFNAME"
else
  echo "[info] EFA      : OFF (NCCL will fall back to TCP sockets)"
fi

# ---------- FP8 (TransformerEngine MoE grouped GEMM) -------------------------
# HW_FP8_ENABLED=1 让 MoE expert GEMM (FLOPs 的大头) 走 TE blockwise FP8
# (matmul 约 2 倍提速), 而 tilelang sparse-MLA attention 保持 bf16。
# 复刻已验证的 smoke/run_stage_a_fp8_smoke.sh 配方。参数保持 bf16
# (不加 --fp8-param-gather), 以满足 deepseek_v4 的 bf16 权重 assert。
FP8_FLAGS=""
V4_FP8_NVTE_VALUE=""
if [[ "${HW_FP8_ENABLED:-0}" == "1" ]]; then
  FP8_FLAGS="--transformer-impl transformer_engine --bf16 --fp8-format ${HW_FP8_FORMAT:-e4m3} --fp8-recipe ${HW_FP8_RECIPE:-blockwise}"
  V4_FP8_NVTE_VALUE="1"
  echo "[info] FP8      : ON  format=${HW_FP8_FORMAT:-e4m3} recipe=${HW_FP8_RECIPE:-blockwise} (MoE grouped GEMM)"
else
  echo "[info] FP8      : OFF (bf16 GEMMs)"
fi

# ---------- 生成容器内启动脚本 -------------------------------------------------
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
# 模块 3 问题 8 的临时规避 — 跳过 optimizer 状态保存, 避免 64K + 长时间运行下
# dist_checkpointing 异步 D2H 出现 cudaErrorInvalidValue。
# SFT 终态续训并不需要 optimizer 状态。
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
  # 调高默认 10 分钟的 NCCL/PG watchdog, 让慢初始化路径 (例如 optimizer
  # cpu-offload 的 pinned buffer 分配、dist-ckpt reshard 读取) 不至于在训练
  # 尚未开始的启动阶段就触发 600s 集合通信超时。
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
    # 把 tilelang @jit kernel 编译缓存持久化到 fsx (默认的 /root/.tilelang 随容器
    # 销毁而丢失 -> pod 重启后每种新的序列形状都要重付约 40 分钟的首步重编译)。
    # fsx 路径跨重启保留, 且所有节点共享。
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
# EFA / aws-ofi-nccl: 仅当 fleet 启用 (取值非空) 时才注入。
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

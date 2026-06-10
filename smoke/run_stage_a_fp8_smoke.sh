#!/usr/bin/env bash
# Stage A FP8 smoke — 单节点 8 GPU, V4 4 层 SFT, 启用 TE FP8 训练。
#
# 保持 Stage A 的小拓扑不变, 仅额外加上:
#   - TransformerEngine FP8 blockwise 训练标志
#   - DeepSeek V4 KV/indexer FP8 QAT 模拟环境变量
#
# 启动: bash smoke/run_stage_a_fp8_smoke.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
source "$REPO_ROOT/cluster/env.sh"

[[ -d "$V4_BF16_DIR" ]] || { echo "[err] BF16 dir missing: $V4_BF16_DIR" >&2; exit 1; }
[[ -f "$V4_SFT_DATA" ]] || { echo "[err] SFT data missing: $V4_SFT_DATA" >&2; exit 1; }
[[ -d "$V4_MODELS/DeepSeek-V4-Flash-bf16-4layer-stub" ]] || {
  echo "[err] 4-layer stub missing: $V4_MODELS/DeepSeek-V4-Flash-bf16-4layer-stub" >&2
  exit 1
}

ssh -o StrictHostKeyChecking=no root@"$V4_MASTER_IP" "docker exec $V4_CONTAINER ray status" 2>&1 | grep -qiE "active|HEALTHY|node_" || {
  echo "[err] ray cluster not healthy. bring_up_cluster.sh first." >&2
  exit 1
}

RUN_ID="stageA-fp8-$(date +%Y%m%d-%H%M%S)"
SAVE_DIR="$V4_OUT/$RUN_ID"
HF_SMOKE_DIR="$SAVE_DIR/hf_4layer_with_tokenizer"
mkdir -p "$SAVE_DIR"
mkdir -p "$HF_SMOKE_DIR"
cp "$V4_MODELS/DeepSeek-V4-Flash-bf16-4layer-stub/config.json" "$HF_SMOKE_DIR/config.json"
cp "$V4_BF16_DIR"/tokenizer*.json "$HF_SMOKE_DIR/"
cp "$V4_BF16_DIR/generation_config.json" "$HF_SMOKE_DIR/"
cp -r "$V4_BF16_DIR/encoding" "$HF_SMOKE_DIR/"
echo "[info] run id   : $RUN_ID"
echo "[info] save dir : $SAVE_DIR"
echo "[info] hf smoke : $HF_SMOKE_DIR"
echo "[info] dashboard: http://$V4_MASTER_IP:$V4_DASHBOARD_PORT"

LAUNCH=$SAVE_DIR/launch_in_container.sh
cat > "$LAUNCH" <<EOF
#!/usr/bin/env bash
set -e
cd $V4_MILES
source scripts/models/deepseek-v4-flash-4layer.sh

CKPT_ARGS=(
  --hf-checkpoint  $HF_SMOKE_DIR
  --ref-load       $V4_TORCH_DIST
  --load           $SAVE_DIR/checkpoints
  --save           $SAVE_DIR/checkpoints
  --save-interval  1
  --save-retain-interval 2
)

SFT_ARGS=(
  --rollout-function-path miles.rollout.sft_rollout.generate_rollout
  --prompt-data    $V4_SFT_DATA
  --input-key      messages
  --rollout-shuffle
  --num-rollout            2
  --rollout-batch-size     32
  --global-batch-size      32

  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --sft-only
  --debug-train-only

  --loss-mask-type deepseek_v4
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 8
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method      uniform
  --recompute-num-layers  1

  --micro-batch-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 5e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9 --adam-beta2 0.95
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

MISC_ARGS=(
  --transformer-impl transformer_engine
  --bf16
  --fp8-format e4m3
  --fp8-recipe blockwise

  --attention-dropout 0.0
  --hidden-dropout    0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --model-name deepseekv4
  --qkv-format thd
  --moe-router-freeze-gate
  --freeze-e-score-correction-bias
  --update-weight-buffer-size 1073741824
  --train-memory-margin-bytes 3221225472

  --actor-num-nodes 1
  --actor-num-gpus-per-node 8
  --num-gpus-per-node 8
  --colocate
  --no-offload-train
  --no-offload-rollout
  --use-fault-tolerance
  --dump-details $SAVE_DIR/dump_details
)

RUNTIME_ENV='{
  "env_vars": {
    "PYTHONPATH": "$V4_RUNTIME_PYTHONPATH:$V4_WORK/wheels",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "MASTER_ADDR": "$V4_MASTER_IP",
    "MILES_DSV4_THINKING_MODE": "chat",
    "MILES_DSV4_DROP_THINKING": "0",
    "NCCL_NVLS_ENABLE": "1",
    "GLOO_SOCKET_IFNAME": "eth0",
    "NCCL_SOCKET_IFNAME": "eth0",
    "LD_PRELOAD": "/usr/local/lib/python3.12/dist-packages/torch_memory_saver_hook_mode_preload.abi3.so",
    "MEGATRON_SPARSE_ATTN_IMPL": "sparse",
    "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
    "MEGATRON_USE_KV_QAT": "1"
  }
}'

ray job submit --address=http://127.0.0.1:$V4_DASHBOARD_PORT \\
   --runtime-env-json="\$RUNTIME_ENV" \\
   -- python3 train.py \\
   "\${MODEL_ARGS[@]}" \\
   "\${CKPT_ARGS[@]}" \\
   "\${SFT_ARGS[@]}" \\
   "\${OPTIMIZER_ARGS[@]}" \\
   "\${PERF_ARGS[@]}" \\
   "\${MISC_ARGS[@]}"
EOF

chmod +x "$LAUNCH"
echo "[info] launch script: $LAUNCH"
echo
echo "=== submit ray job (live logs mirrored to $SAVE_DIR/job.log) ==="
ssh -o StrictHostKeyChecking=no root@"$V4_MASTER_IP" "docker exec $V4_CONTAINER bash $LAUNCH" 2>&1 | tee "$SAVE_DIR/job.log"

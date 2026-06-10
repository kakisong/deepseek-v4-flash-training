#!/usr/bin/env bash
# 在当前 K8s Ray 集群上准备 DeepSeek-V4-Flash checkpoint。
#
# 流水线:
#   1. FP8 HF/MegaBlocks 源 -> BF16 HF unpacked。
#   2. BF16 HF unpacked -> Megatron torch_dist。
#
# 默认值与当前 FSX 目录布局一致:
#   source: $V4_MODELS/DeepSeek-V4-Flash
#   bf16:   $V4_MODELS/DeepSeek-V4-Flash-bf16-unpacked
#   dist:   $V4_MODELS/DeepSeek-V4-Flash-FP8_torch_dist

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

: "${V4_SUBMIT_MODE:=k8s}"
: "${V4_K8S_NAMESPACE:=ray-system}"
: "${V4_K8S_HEAD_DEPLOY:=ray-head-gpu-node-64}"
: "${V4_K8S_HEAD_CONTAINER:=ray-head}"
: "${V4_K8S_GPU_WORKER_LABEL:=app.kubernetes.io/name=ray-gpu-worker-h200-k8s-42node}"

: "${V4_FP8_HF_DIR:=$V4_MODELS/DeepSeek-V4-Flash}"
: "${V4_CONVERT_MODEL_NAME:=DeepSeek-V4-Flash-FP8}"
: "${V4_CONVERT_NUM_NODES:=8}"
: "${V4_FP8_TO_BF16_TOOL:=megablocks}"

if [[ "$V4_SUBMIT_MODE" != "k8s" ]]; then
  echo "[err] cluster/prepare_v4_ckpt_k8s.sh requires V4_SUBMIT_MODE=k8s" >&2
  exit 2
fi

kexec_head() {
  if [[ -n "${V4_K8S_HEAD_CONTAINER:-}" ]]; then
    kubectl exec -n "$V4_K8S_NAMESPACE" "deploy/$V4_K8S_HEAD_DEPLOY" -c "$V4_K8S_HEAD_CONTAINER" -- "$@"
  else
    kubectl exec -n "$V4_K8S_NAMESPACE" "deploy/$V4_K8S_HEAD_DEPLOY" -- "$@"
  fi
}

gpu_worker_pod() {
  kubectl get pod -n "$V4_K8S_NAMESPACE" \
    -l "$V4_K8S_GPU_WORKER_LABEL" \
    --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
}

kexec_gpu_worker() {
  local pod
  pod="$(gpu_worker_pod)"
  if [[ -z "$pod" ]]; then
    echo "[err] no running GPU worker pod matched label: $V4_K8S_GPU_WORKER_LABEL" >&2
    exit 1
  fi
  kubectl exec -n "$V4_K8S_NAMESPACE" "$pod" -- "$@"
}

check_ray_capacity() {
  kexec_head env EXPECTED_GPUS="$V4_EXPECTED_GPUS" RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 python3 - <<'PY'
import os
import sys
import ray

expected_gpus = int(os.environ["EXPECTED_GPUS"])
ray.init(address="auto", logging_level="ERROR")
alive = sum(1 for n in ray.nodes() if n.get("Alive"))
gpus = int(ray.cluster_resources().get("GPU", 0))
print(f"[info] ray capacity: alive_nodes={alive} gpus={gpus}/{expected_gpus}")
sys.exit(0 if gpus >= expected_gpus else 1)
PY
}

cast_fp8_to_bf16() {
  if [[ -f "$V4_BF16_DIR/model.safetensors.index.json" ]]; then
    echo "[skip] BF16 HF already exists: $V4_BF16_DIR"
    return
  fi

  [[ -d "$V4_FP8_HF_DIR" ]] || {
    echo "[err] FP8 HF source missing: $V4_FP8_HF_DIR" >&2
    echo "      set V4_FP8_HF_DIR=/path/to/DeepSeek-V4-Flash and rerun" >&2
    exit 1
  }
  [[ -f "$V4_FP8_HF_DIR/model.safetensors.index.json" ]] || {
    echo "[err] FP8 HF source is missing model.safetensors.index.json: $V4_FP8_HF_DIR" >&2
    exit 1
  }

  echo "[info] FP8 source : $V4_FP8_HF_DIR"
  echo "[info] BF16 output: $V4_BF16_DIR"
  case "$V4_FP8_TO_BF16_TOOL" in
    megablocks)
      kexec_gpu_worker bash -lc "cd '$V4_TRAINING_REPO' && python3 tools/megablocks_to_hf_bf16.py --src '$V4_FP8_HF_DIR' --dst '$V4_BF16_DIR'"
      ;;
    sglang)
      kexec_gpu_worker bash -lc "cd '$V4_MILES_REPO' && python3 tools/fp8_cast_bf16.py --input-fp8-hf-path '$V4_FP8_HF_DIR' --output-bf16-hf-path '$V4_BF16_DIR'"
      ;;
    *)
      echo "[err] unknown V4_FP8_TO_BF16_TOOL=$V4_FP8_TO_BF16_TOOL (megablocks|sglang)" >&2
      exit 2
      ;;
  esac
}

sync_dsv4_encoding() {
  local src dst
  src="$V4_TRAINING_REPO/tokenizer/encoding_dsv4.py"
  dst="$V4_BF16_DIR/encoding/encoding_dsv4.py"
  mkdir -p "$(dirname "$dst")"
  install -m 0644 "$src" "$dst"
  echo "[info] synced encoding_dsv4.py -> $dst"
}

convert_bf16_to_torch_dist() {
  local tracker conv_py conv_sh
  tracker="$V4_TORCH_DIST/latest_checkpointed_iteration.txt"
  if [[ -f "$tracker" && "$(tr -d '[:space:]' < "$tracker")" == "release" ]]; then
    echo "[skip] torch_dist already exists: $V4_TORCH_DIST"
    return
  fi

  [[ -f "$V4_BF16_DIR/model.safetensors.index.json" ]] || {
    echo "[err] BF16 not ready: $V4_BF16_DIR/model.safetensors.index.json" >&2
    exit 1
  }

  mkdir -p "$V4_OUT"
  conv_py="$V4_OUT/.convert_v4_k8s.py"
  conv_sh="$V4_OUT/.convert_v4_k8s.sh"

  cat > "$conv_py" <<EOF
import miles.utils.misc as _misc
import miles.utils.external_utils.command_utils as _cu

_DEBUG_ENV = (
    "NCCL_IB_DISABLE=1 "
    "NCCL_NET_GDR_LEVEL=0 "
    "NCCL_DEBUG=WARN "
    "NCCL_SOCKET_IFNAME=eth0 "
    "GLOO_SOCKET_IFNAME=eth0 "
    "TORCH_NCCL_BLOCKING_WAIT=1 "
    "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
)

_orig = _misc.exec_command_all_ray_node

def _patched(cmd, *a, **kw):
    cmd = cmd.replace("PYTHONPATH=", _DEBUG_ENV + "PYTHONPATH=", 1)
    return _orig(cmd, *a, **kw)

_misc.exec_command_all_ray_node = _patched
_cu.exec_command_all_ray_node = _patched

_cu.convert_checkpoint(
    model_name="$V4_CONVERT_MODEL_NAME",
    megatron_model_type="deepseek-v4-flash",
    num_gpus_per_node=$V4_NUM_GPUS_PER_NODE,
    multinode=True,
    num_nodes=$V4_CONVERT_NUM_NODES,
    extra_args=(
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 8 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--context-parallel-size 1 "
        "--decoder-first-pipeline-num-layers 7 "
        "--decoder-last-pipeline-num-layers 6 "
    ),
    dir_dst="$V4_MODELS",
    hf_checkpoint="$V4_BF16_DIR",
    megatron_path="$V4_MILES_REPO:$V4_RUNTIME_PYTHONPATH",
)
EOF

  cat > "$conv_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd '$V4_MILES_REPO'
PYTHONPATH='$V4_MILES_REPO:$V4_RUNTIME_PYTHONPATH' python3 '$conv_py'
EOF
  chmod +x "$conv_sh"

  echo "[info] launching BF16 -> torch_dist conversion on $V4_CONVERT_NUM_NODES nodes"
  kexec_head bash "$conv_sh" 2>&1 | tee "$V4_OUT/.convert_v4_k8s.log"
}

echo "[info] work         : $V4_WORK"
echo "[info] models       : $V4_MODELS"
echo "[info] fp8 source   : $V4_FP8_HF_DIR"
echo "[info] bf16 output  : $V4_BF16_DIR"
echo "[info] torch_dist   : $V4_TORCH_DIST"
echo "[info] fp8 tool     : $V4_FP8_TO_BF16_TOOL"
echo "[info] convert nodes: $V4_CONVERT_NUM_NODES"

check_ray_capacity
cast_fp8_to_bf16
sync_dsv4_encoding
convert_bf16_to_torch_dist

echo "[done] checkpoint preparation complete"

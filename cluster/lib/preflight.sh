#!/usr/bin/env bash
# Pre-launch environment checks — source from run.sh and call preflight_64gpu.
# Extracted from the old run_smoke.sh / run_sft_validation.sh / run_cp_smoke.sh.

_preflight_repo_root() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd
}

_k8s_head_exec() {
  local ns deploy container
  ns="${V4_K8S_NAMESPACE:-ray-system}"
  deploy="${V4_K8S_HEAD_DEPLOY:-ray-head-gpu-node-64}"
  container="${V4_K8S_HEAD_CONTAINER:-}"

  if [[ -n "$container" ]]; then
    kubectl exec -n "$ns" "deploy/$deploy" -c "$container" -- "$@"
  else
    kubectl exec -n "$ns" "deploy/$deploy" -- "$@"
  fi
}

_head_bash() {
  local script="$1"
  if [[ "${V4_SUBMIT_MODE:-ssh}" == "k8s" ]]; then
    _k8s_head_exec bash -lc "$script"
  else
    ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$V4_RAY_HEAD_IP" \
      "docker exec $V4_CONTAINER bash -lc $(printf '%q' "$script")"
  fi
}

sync_dsv4_encoding() {
  local src="${V4_DSV4_ENCODING_SRC:-$(_preflight_repo_root)/tokenizer/encoding_dsv4.py}"
  local dst="$V4_BF16_DIR/encoding/encoding_dsv4.py"
  local dst_dir
  dst_dir="$(dirname "$dst")"

  [[ -f "$src" ]] || {
    echo "[err] DeepSeek-V4 encoding source missing: $src" >&2
    return 1
  }
  [[ -d "$V4_BF16_DIR" ]] || {
    echo "[err] BF16 dir missing before encoding sync: $V4_BF16_DIR" >&2
    return 1
  }

  mkdir -p "$dst_dir"
  if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
    install -m 0644 "$src" "$dst"
    echo "[info] synced DeepSeek-V4 encoding: $src -> $dst"
  fi

  [[ -f "$dst" ]] || {
    echo "[err] DeepSeek-V4 encoding sync failed: $dst" >&2
    return 1
  }
}

check_runtime_framework_deps() {
  _head_bash "PYTHONPATH=\"$V4_RUNTIME_PYTHONPATH\" python -c \"import fast_hadamard_transform, tile_kernels; import megatron.core.dist_checkpointing.core as c; from megatron.core.transformer.transformer_config import TransformerConfig; assert c.CONFIG_FNAME == 'metadata.json'; assert 'dsv4_hc_mult' in TransformerConfig.__dataclass_fields__\""
}

check_ray_capacity() {
  local expected_nodes expected_gpus
  expected_nodes="$V4_EXPECTED_RAY_NODES"
  expected_gpus="$V4_EXPECTED_GPUS"

  if [[ "${V4_SUBMIT_MODE:-ssh}" == "k8s" ]]; then
    _k8s_head_exec env EXPECTED_NODES="$expected_nodes" EXPECTED_GPUS="$expected_gpus" RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 python3 -c '
import os
import sys
import ray

expected_nodes = int(os.environ["EXPECTED_NODES"])
expected_gpus = int(os.environ["EXPECTED_GPUS"])
ray.init(address="auto", logging_level="ERROR")
alive_nodes = sum(1 for n in ray.nodes() if n.get("Alive"))
gpus = int(ray.cluster_resources().get("GPU", 0))
print(f"[info] ray capacity: alive_nodes={alive_nodes}/{expected_nodes} gpus={gpus}/{expected_gpus}")
sys.exit(0 if alive_nodes >= expected_nodes and gpus >= expected_gpus else 1)
'
  else
    ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$V4_RAY_HEAD_IP" \
      "docker exec -i -e EXPECTED_NODES=$expected_nodes -e EXPECTED_GPUS=$expected_gpus -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 $V4_CONTAINER python3 - <<'PY'
import os
import sys

import ray

expected_nodes = int(os.environ['EXPECTED_NODES'])
expected_gpus = int(os.environ['EXPECTED_GPUS'])
ray.init(address='auto', logging_level='ERROR')
alive_nodes = sum(1 for n in ray.nodes() if n.get('Alive'))
gpus = int(ray.cluster_resources().get('GPU', 0))
print(f'[info] ray capacity: alive_nodes={alive_nodes}/{expected_nodes} gpus={gpus}/{expected_gpus}')
sys.exit(0 if alive_nodes >= expected_nodes and gpus >= expected_gpus else 1)
PY
"
  fi
}

preflight_64gpu() {
  local err=0

  [[ -d "$V4_BF16_DIR" ]] || { echo "[err] BF16 dir missing: $V4_BF16_DIR — cast first" >&2; err=1; }
  [[ -d "$V4_TORCH_DIST" ]] || { echo "[err] torch_dist missing: $V4_TORCH_DIST — convert first" >&2; err=1; }
  [[ -f "$V4_SFT_DATA" ]] || { echo "[err] SFT data missing: $V4_SFT_DATA" >&2; err=1; }
  if [[ -d "$V4_BF16_DIR" ]]; then
    sync_dsv4_encoding || err=1
  fi

  if (( err == 0 )); then
    _head_bash "ray status --address=127.0.0.1:$V4_RAY_PORT" 2>&1 \
      | grep -qiE "active|HEALTHY|node_" || {
        echo "[err] ray cluster not healthy. Prepare the Ray control plane first." >&2
        err=1
    }
    check_ray_capacity || {
      echo "[err] ray cluster does not have the expected fleet capacity. Run cluster/ensure_ray_workers.sh first." >&2
      err=1
    }
    _head_bash "test -f '$V4_BF16_DIR/encoding/encoding_dsv4.py'" || {
        echo "[err] DeepSeek-V4 encoding is not visible inside container: $V4_BF16_DIR/encoding/encoding_dsv4.py" >&2
        err=1
    }
    check_runtime_framework_deps || {
      echo "[err] image runtime deps invalid: need FHT, Megatron dsv4, and TileKernels baked into $V4_IMAGE" >&2
      err=1
    }
  fi

  return $err
}

# Stricter variant: smoke also requires the cast/convert completion marker files.
preflight_64gpu_strict() {
  preflight_64gpu || return 1
  local err=0
  [[ -f "$V4_BF16_DIR/model.safetensors.index.json" ]] || { echo "[err] cast not finished: $V4_BF16_DIR/model.safetensors.index.json" >&2; err=1; }
  [[ -f "$V4_TORCH_DIST/latest_checkpointed_iteration.txt" ]] || { echo "[err] convert not finished: $V4_TORCH_DIST/latest_checkpointed_iteration.txt" >&2; err=1; }
  return $err
}

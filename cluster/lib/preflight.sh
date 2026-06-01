#!/usr/bin/env bash
# Pre-launch environment checks — source from run.sh and call preflight_64gpu.
# Extracted from the old run_smoke.sh / run_sft_validation.sh / run_cp_smoke.sh.

_preflight_repo_root() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd
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
  ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$V4_MASTER_IP" \
    "docker exec $V4_CONTAINER bash -lc 'PYTHONPATH=\"$V4_RUNTIME_PYTHONPATH\" python -c \"import fast_hadamard_transform, tile_kernels; import megatron.core.dist_checkpointing.core as c; from megatron.core.transformer.transformer_config import TransformerConfig; assert c.CONFIG_FNAME == \\\"metadata.json\\\"; assert \\\"dsv4_hc_mult\\\" in TransformerConfig.__dataclass_fields__\"'"
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
    ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$V4_MASTER_IP" \
      "docker exec $V4_CONTAINER ray status" 2>&1 \
      | grep -qiE "active|HEALTHY|node_" || {
        echo "[err] ray cluster not healthy. bring_up_cluster.sh first." >&2
        err=1
    }
    ssh -o StrictHostKeyChecking=no -o LogLevel=ERROR "root@$V4_MASTER_IP" \
      "docker exec $V4_CONTAINER test -f '$V4_BF16_DIR/encoding/encoding_dsv4.py'" || {
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

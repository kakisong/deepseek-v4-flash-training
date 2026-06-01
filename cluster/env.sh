#!/usr/bin/env bash
# Layered env loader.
#
# 维度 (互相正交):
#   fleet/<name>.env       — 哪批物理机 + image/ports/redis/mounts (V4_*)
#   hw/<gpu_model>.env     — 机型默认硬件参数 (HW_*)
#   scale/<name>.env       — 并行策略 TP/PP/CP/EP/VPP/layout (PRESET_*)
#   workload/<name>.env    — data/lr/steps/save/optim 算法 (PRESET_*)
#   base.env               — 路径派生 (V4_MILES, V4_DATA, ...)
#
# 入口三种用法 (任选):
#   1. V4_FLEET=h20_16node V4_SCALE=tp8pp16ep8_layout V4_WORKLOAD=sft_prod bash run.sh
#   2. bash run.sh --fleet h20_16node --scale tp8pp16ep8_layout --workload sft_prod
#   3. bash run.sh prod                  (向后兼容: 走老 presets/<preset>.env 路径)
#
# 直接被 bring_up_cluster.sh / tear_down.sh / bring_up_caddy.sh source 时,只需要
# fleet 维度 (这些脚本不关心训练参数)。

set -u
_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

_load_or_die() {
  local kind="$1" name="$2"
  local path="$_SCRIPT_DIR/$kind/$name.env"
  if [[ ! -f "$path" ]]; then
    echo "[env.sh] $kind file not found: $path" >&2
    echo "[env.sh] available $kind: $(ls "$_SCRIPT_DIR/$kind/" 2>/dev/null | grep '\.env$' | sed 's/\.env$//' | tr '\n' ' ')" >&2
    return 1
  fi
  # shellcheck disable=SC1090
  source "$path"
}

# Fleet (always required).
_FLEET="${V4_FLEET:-h20_16node}"
_load_or_die fleet "$_FLEET" || { return 1 2>/dev/null || exit 1; }

# Project paths derive from V4_WORK set by fleet.
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/base.env"

# Hardware defaults (selected by V4_GPU_MODEL set in fleet).
_load_or_die hw "$V4_GPU_MODEL" || { return 1 2>/dev/null || exit 1; }

# Scale + workload only loaded when callers set them (run.sh sets, infra scripts don't).
if [[ -n "${V4_SCALE:-}" ]]; then
  _load_or_die scale "$V4_SCALE" || { return 1 2>/dev/null || exit 1; }
fi
if [[ -n "${V4_WORKLOAD:-}" ]]; then
  _load_or_die workload "$V4_WORKLOAD" || { return 1 2>/dev/null || exit 1; }
fi

unset _SCRIPT_DIR _FLEET
unset -f _load_or_die

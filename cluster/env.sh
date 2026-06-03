#!/usr/bin/env bash
# Layered env loader.
#
# 维度 (互相正交):
#   control/<name>.env     — 固定 Ray head/job-server + web/monitoring 入口 (V4_RAY_HEAD_IP 等)
#   fleet/<name>.env       — 本次 job 希望使用的物理机池 + image/mounts/capacity (V4_*)
#   hw/<gpu_model>.env     — 机型默认硬件参数 (HW_*)
#   scale/<name>.env       — 并行策略 TP/PP/CP/EP/VPP/layout (PRESET_*)
#   workload/<name>.env    — data/lr/steps/save/optim 算法 (PRESET_*)
#   base.env               — 路径派生 (V4_MILES, V4_DATA, ...)
#
# 入口三种用法 (任选):
#   1. V4_CONTROL=current V4_FLEET=h20_16node V4_SCALE=tp8pp16ep8_layout V4_WORKLOAD=sft_prod bash run.sh
#   2. bash run.sh --control current --fleet h20_16node --scale tp8pp16ep8_layout --workload sft_prod
#   3. bash run.sh prod                  (向后兼容: 走老 presets/<preset>.env 路径)
#
# 直接被 prepare_ray_head.sh / ensure_ray_workers.sh / bring_up_caddy.sh source 时,
# 只需要 control + fleet 维度 (这些脚本不关心训练参数)。

set -u
_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export V4_HOST_RAY_LOCAL_DIR="${V4_HOST_RAY_LOCAL_DIR:-/data0}"
export V4_CONTAINER_RAY_LOCAL_DIR="${V4_CONTAINER_RAY_LOCAL_DIR:-/ray_local}"
export V4_RAY_TEMP_DIR="${V4_RAY_TEMP_DIR:-$V4_CONTAINER_RAY_LOCAL_DIR/ray}"

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

# Fleet (always required): job node pool and capacity.
_FLEET="${V4_FLEET:-h20_16node}"
_load_or_die fleet "$_FLEET" || { return 1 2>/dev/null || exit 1; }

# Control plane (optional for backward compatibility): fixed submit/head entry.
_CONTROL="${V4_CONTROL:-}"
if [[ -n "$_CONTROL" ]]; then
  _load_or_die control "$_CONTROL" || { return 1 2>/dev/null || exit 1; }
fi

# Compatibility aliases. New code should use V4_RAY_HEAD_IP for the Ray control
# plane and V4_MASTER_IP only as the first node in a legacy fleet definition.
: "${V4_RAY_HEAD_IP:=${V4_MASTER_IP:?V4_MASTER_IP or V4_RAY_HEAD_IP required}}"
: "${V4_MASTER_IP:=$V4_RAY_HEAD_IP}"
: "${V4_TRAINING_MASTER_IP:=$V4_RAY_HEAD_IP}"
: "${V4_GRAFANA_HOST:=http://${V4_RAY_HEAD_IP}:${V4_GRAFANA_PORT:-7777}}"
: "${V4_PROMETHEUS_HOST:=http://${V4_RAY_HEAD_IP}:${V4_PROMETHEUS_PORT:-40001}/promql}"

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

unset _SCRIPT_DIR _FLEET _CONTROL
unset -f _load_or_die

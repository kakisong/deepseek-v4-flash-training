#!/usr/bin/env bash
# 分层环境变量加载器。
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

# fleet(始终必选):本次 job 的物理机池与容量。
_FLEET="${V4_FLEET:-h20_16node}"
_load_or_die fleet "$_FLEET" || { return 1 2>/dev/null || exit 1; }

# 控制面(为向后兼容设为可选):固定的提交/head 入口。
_CONTROL="${V4_CONTROL:-}"
if [[ -n "$_CONTROL" ]]; then
  _load_or_die control "$_CONTROL" || { return 1 2>/dev/null || exit 1; }
fi

# 兼容别名。新代码应使用 V4_RAY_HEAD_IP 表示 Ray 控制面,
# V4_MASTER_IP 仅用作旧版 fleet 定义中的第一台节点。
: "${V4_RAY_HEAD_IP:=${V4_MASTER_IP:?V4_MASTER_IP or V4_RAY_HEAD_IP required}}"
: "${V4_MASTER_IP:=$V4_RAY_HEAD_IP}"
: "${V4_TRAINING_MASTER_IP:=$V4_RAY_HEAD_IP}"
: "${V4_GRAFANA_HOST:=http://${V4_RAY_HEAD_IP}:${V4_GRAFANA_PORT:-7777}}"
: "${V4_PROMETHEUS_HOST:=http://${V4_RAY_HEAD_IP}:${V4_PROMETHEUS_PORT:-40001}/promql}"

# Ray 容量默认值。多数 fleet 的 Ray head 跑在 GPU 节点上,因此
# 期望的 Ray 资源与训练 actor 容量一致。head 为纯 CPU 的
# fleet 可显式覆盖这些值。
: "${V4_RAY_HEAD_NUM_GPUS:=$V4_NUM_GPUS_PER_NODE}"
: "${V4_EXPECTED_RAY_NODES:=$V4_NUM_NODES}"
: "${V4_EXPECTED_GPUS:=$((V4_NUM_NODES * V4_NUM_GPUS_PER_NODE))}"

# 项目路径由 fleet 设置的 V4_WORK 派生。
# shellcheck disable=SC1091
source "$_SCRIPT_DIR/base.env"

# 硬件默认值(由 fleet 中设置的 V4_GPU_MODEL 选定)。
_load_or_die hw "$V4_GPU_MODEL" || { return 1 2>/dev/null || exit 1; }

# scale + workload 仅在调用方设置时才加载(run.sh 会设置,infra 脚本不需要)。
if [[ -n "${V4_SCALE:-}" ]]; then
  _load_or_die scale "$V4_SCALE" || { return 1 2>/dev/null || exit 1; }
fi
if [[ -n "${V4_WORKLOAD:-}" ]]; then
  _load_or_die workload "$V4_WORKLOAD" || { return 1 2>/dev/null || exit 1; }
fi

unset _SCRIPT_DIR _FLEET _CONTROL
unset -f _load_or_die

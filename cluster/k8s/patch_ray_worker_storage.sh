#!/usr/bin/env bash
# 在集群 / DaemonSet(重)建之后,重新应用 ray-worker 的存储修复。
#
# H200 k8s ray-gpu-worker DaemonSet 自带 `ray start --temp-dir=/ray_local/ray`
# 且没有 object-store 上限。/ray_local 是块很小(约 100 GB)的共享本地 NVMe;在不足 42
# 节点的运行中,ray object-spill 会把它写满,worker 在 step 1 之前就被打挂。本补丁:
#   1. 把 ray temp-dir(日志 + object spill)挪到 286 GB 容器 overlay 上的
#      /ray_spill_local(本地、socket 可用、不受 k8s ephemeral-storage 限制),并
#   2. 把 object store 上限设为 230 GB(可装进 256 GB 的 /dev/shm,减少 spill)。
# 幂等:可安全重跑。会滚动重启 DaemonSet,让全部 42 个 worker 生效。
#
# 注意:deep_ep 现已固化进镜像(docker/Dockerfile.sft-only)。本脚本
# 不再添加旧的启动行 `pip install deep_ep ...`;请确保 DaemonSet 使用的
# 镜像是包含 deep_ep 的构建(重新构建 Dockerfile.sft-only -> push -> 设置
# DaemonSet/fleet 镜像)。参见 docs/h200_bottleneck_analysis.md 与
# h200-k8s-16node-runbook 记忆。
#
# 用法: cluster/k8s/patch_ray_worker_storage.sh [daemonset-name] [namespace]
set -euo pipefail

DS="${1:-ray-gpu-worker-h200-k8s-42node}"
NS="${2:-ray-system}"
SPILL_DIR="/ray_spill_local/ray"
OBJ_STORE_BYTES="230000000000"

command -v kubectl >/dev/null || { echo "[err] kubectl not found" >&2; exit 1; }
command -v jq >/dev/null || { echo "[err] jq not found" >&2; exit 1; }

echo "[info] patching DaemonSet $NS/$DS : temp-dir -> $SPILL_DIR, object-store -> $OBJ_STORE_BYTES"
TMP="$(mktemp)"; trap 'rm -f "$TMP"' EXIT
kubectl get ds -n "$NS" "$DS" -o json > "$TMP"

jq --arg obj "$OBJ_STORE_BYTES" '
  .spec.template.spec.containers[0].args[0] |= (
      # 1. 把 temp-dir 及其 mkdir 从小容量的 /ray_local 挪到大容量的 overlay
      gsub("/ray_local/ray"; "/ray_spill_local/ray")
      # 2. 在 --block 前插入 --object-store-memory(仅当尚未存在时)
      | if test("--object-store-memory") then .
        else gsub("  --block"; "  --object-store-memory=" + $obj + " \\\n  --block") end
    )
  | .spec.updateStrategy.rollingUpdate.maxUnavailable = "100%"
  | del(.status) | del(.metadata.resourceVersion)
' "$TMP" | kubectl apply -f - >/dev/null

echo "[info] applied; waiting for rollout..."
kubectl rollout status ds/"$DS" -n "$NS" --timeout=300s

echo "[info] verify (one worker's raylet args):"
POD="$(kubectl get pods -n "$NS" -l 2>/dev/null | true; kubectl get pods -n "$NS" | grep "$DS" | grep Running | head -1 | awk '{print $1}')"
[ -n "$POD" ] && kubectl exec -n "$NS" "$POD" -- bash -lc \
  'ps aux 2>/dev/null | grep -oE "temp.?dir=[^ ]+|object[_-]store[_-]memory=[0-9]+" | sort -u' || true
echo "[done] ray-worker storage fixes applied to $NS/$DS"

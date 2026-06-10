#!/usr/bin/env bash
# 在 Kubernetes 上拉起(或拆除)生产环境 H200 42 节点 Ray 集群。
#
# 本脚本是 cluster/bring_up_cluster.sh(即 H20 / 裸金属 H200 fleet 使用的 ssh/docker 路径)
# 在 k8s 侧的对应物。它直接 apply 本目录中保存的普通 Deployment + Service + DaemonSet——
# 这个集群里没有 KubeRay operator,所以"拉起容器"
# == kubectl apply 这些 manifests 并等待对应的 pods 就绪。
#
# 顺序至关重要:必须先有 head,因为每个 worker pod 在执行 `ray start` 前会阻塞最多 300 s,
# 等待位于 10.3.234.60:6379 的 head GCS(该逻辑写死在 ray-gpu-worker.yaml 中)。
# 因此:apply head -> 等待 head Ready -> apply workers -> 等待 DaemonSet 滚动完成 -> 在
# head pod 内 exec `ray status` 校验 Ray 容量(43 节点 / 336 GPU)。
#
# worker manifest 已经是打完补丁后的状态(overlay temp-dir、230 GB object store、
# deep_ep wheel)——全新 apply 之后无需再补跑 cluster/k8s/patch_ray_worker_storage.sh。
#
# 用法:
#   kuberay/h200-k8s-42node/bring_up.sh                 # apply head+workers,等待并校验
#   kuberay/h200-k8s-42node/bring_up.sh --dry-run       # 仅对 manifests 做服务端校验
#   kuberay/h200-k8s-42node/bring_up.sh --head-only     # 仅 head Deployment + Service
#   kuberay/h200-k8s-42node/bring_up.sh --no-verify     # 跳过 ray-status 容量门禁
#   kuberay/h200-k8s-42node/bring_up.sh --delete        # 拆除整个集群
#
# 可用环境变量覆盖:V4_K8S_NAMESPACE(默认 ray-system)、V4_EXPECTED_GPUS(336)、
# V4_EXPECTED_RAY_NODES(43)、HEAD_TIMEOUT(300s)、WORKER_TIMEOUT(600s)。
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
NS="${V4_K8S_NAMESPACE:-ray-system}"
HEAD_DEPLOY="ray-head-gpu-node-64"
HEAD_CONTAINER="ray-head"
WORKER_DS="ray-gpu-worker-h200-k8s-42node"
EXPECTED_GPUS="${V4_EXPECTED_GPUS:-336}"
EXPECTED_NODES="${V4_EXPECTED_RAY_NODES:-43}"
HEAD_TIMEOUT="${HEAD_TIMEOUT:-300s}"
WORKER_TIMEOUT="${WORKER_TIMEOUT:-600s}"
DEEPEP_WHEEL="/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/deps/wheels/deep_ep-1.2.1+9af0e0d-cp312-cp312-linux_x86_64.whl"

HEAD_YAML="$HERE/ray-head.yaml"
WORKER_YAML="$HERE/ray-gpu-worker.yaml"

MODE=apply
HEAD_ONLY=0
DRY_RUN=0
VERIFY=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete)    MODE=delete; shift ;;
    --head-only) HEAD_ONLY=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --no-verify) VERIFY=0; shift ;;
    -h|--help)   sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'; exit 0 ;;
    *) echo "[err] unknown arg: $1 (see --help)" >&2; exit 1 ;;
  esac
done

command -v kubectl >/dev/null || { echo "[err] kubectl not found" >&2; exit 1; }
[[ -f "$HEAD_YAML" && -f "$WORKER_YAML" ]] || { echo "[err] manifests missing next to $0" >&2; exit 1; }

# ---------- 拆除 --------------------------------------------------------------
if [[ "$MODE" == delete ]]; then
  echo "=== Deleting Ray cluster in namespace $NS (workers first, then head) ==="
  kubectl delete -f "$WORKER_YAML" --ignore-not-found
  kubectl delete -f "$HEAD_YAML" --ignore-not-found
  echo "[done] torn down"
  exit 0
fi

# ---------- dry-run -----------------------------------------------------------
if (( DRY_RUN == 1 )); then
  echo "=== Server-side validating manifests (no changes applied) ==="
  kubectl apply -f "$HEAD_YAML" --dry-run=server
  (( HEAD_ONLY == 0 )) && kubectl apply -f "$WORKER_YAML" --dry-run=server
  echo "[ok] manifests validate"
  exit 0
fi

# ---------- 软性预检 ----------------------------------------------------------
if [[ ! -f "$DEEPEP_WHEEL" ]]; then
  echo "[warn] deep_ep wheel not found on fsx: $DEEPEP_WHEEL"
  echo "[warn]   workers start anyway (pip line is '|| true'), but PRESET_MOE_DEEPEP=1 will fail."
fi

# ---------- head --------------------------------------------------------------
echo "=== Phase 1: applying head Deployment + Service ==="
kubectl apply -f "$HEAD_YAML"
echo "=== waiting for head rollout (timeout $HEAD_TIMEOUT) ==="
kubectl rollout status "deploy/$HEAD_DEPLOY" -n "$NS" --timeout="$HEAD_TIMEOUT"

if (( HEAD_ONLY == 1 )); then
  echo "[done] head only; skipping workers"
  exit 0
fi

# ---------- workers -----------------------------------------------------------
echo
echo "=== Phase 2: applying worker DaemonSet (42 H200 nodes) ==="
kubectl apply -f "$WORKER_YAML"
echo "=== waiting for DaemonSet rollout (timeout $WORKER_TIMEOUT) ==="
kubectl rollout status "ds/$WORKER_DS" -n "$NS" --timeout="$WORKER_TIMEOUT"

# ---------- 校验 ---------------------------------------------------------------
if (( VERIFY == 0 )); then
  echo "[done] applied (verification skipped)"
  exit 0
fi

echo
echo "=== Phase 3: verifying Ray capacity from inside the head pod ==="
# worker 连接 head GCS 的回退等待最长可达 300 s;开始统计前先稍等片刻。
sleep 10
kubectl exec -n "$NS" "deploy/$HEAD_DEPLOY" -c "$HEAD_CONTAINER" -- \
  env EXPECTED_GPUS="$EXPECTED_GPUS" EXPECTED_NODES="$EXPECTED_NODES" \
      RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 python3 - <<'PY'
import os
import sys
import time

import ray

want_gpus = int(os.environ["EXPECTED_GPUS"])
want_nodes = int(os.environ["EXPECTED_NODES"])
ray.init(address="auto", logging_level="ERROR")
# 留出短暂的稳定窗口,以防最后几个 worker 仍在加入。
deadline = time.time() + 120
while True:
    alive = sum(1 for n in ray.nodes() if n.get("Alive"))
    gpus = int(ray.cluster_resources().get("GPU", 0))
    if (alive >= want_nodes and gpus >= want_gpus) or time.time() > deadline:
        break
    time.sleep(5)
print(f"[info] ray capacity: alive_nodes={alive}/{want_nodes} gpus={gpus}/{want_gpus}")
sys.exit(0 if alive >= want_nodes and gpus >= want_gpus else 1)
PY

echo
echo "[done] Ray cluster up: head + $WORKER_DS"
echo "Dashboard (in head pod / via Caddy): http://10.3.234.60:8201"
echo "Submit training:  V4_CONTROL=h200_main V4_FLEET=h200_k8s_42node ... bash run.sh"

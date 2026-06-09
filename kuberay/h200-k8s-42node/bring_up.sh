#!/usr/bin/env bash
# Bring up (or tear down) the production H200 42-node Ray cluster on Kubernetes.
#
# This is the k8s analogue of cluster/bring_up_cluster.sh (which is the ssh/docker path for the
# H20 / bare-metal H200 fleets). It applies the plain Deployment + Service + DaemonSet captured
# in this directory — there is NO KubeRay operator in this cluster, so "bring up the containers"
# == kubectl apply these manifests and wait for their pods.
#
# Order is load-bearing: the head must exist first, because every worker pod blocks up to 300 s
# waiting for the head GCS at 10.3.234.60:6379 before `ray start` (baked into ray-gpu-worker.yaml).
# So: apply head -> wait head Ready -> apply workers -> wait DaemonSet rolled out -> verify Ray
# capacity (43 nodes / 336 GPU) by exec'ing `ray status` inside the head pod.
#
# The worker manifest is already the post-patch state (overlay temp-dir, 230 GB object store,
# deep_ep wheel) — a fresh apply needs no follow-up cluster/k8s/patch_ray_worker_storage.sh.
#
# Usage:
#   kuberay/h200-k8s-42node/bring_up.sh                 # apply head+workers, wait, verify
#   kuberay/h200-k8s-42node/bring_up.sh --dry-run       # server-side validate manifests only
#   kuberay/h200-k8s-42node/bring_up.sh --head-only     # head Deployment + Service only
#   kuberay/h200-k8s-42node/bring_up.sh --no-verify     # skip the ray-status capacity gate
#   kuberay/h200-k8s-42node/bring_up.sh --delete        # tear the whole cluster down
#
# Env overrides: V4_K8S_NAMESPACE (default ray-system), V4_EXPECTED_GPUS (336),
# V4_EXPECTED_RAY_NODES (43), HEAD_TIMEOUT (300s), WORKER_TIMEOUT (600s).
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

# ---------- teardown ----------------------------------------------------------
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

# ---------- soft preflight ----------------------------------------------------
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

# ---------- verify ------------------------------------------------------------
if (( VERIFY == 0 )); then
  echo "[done] applied (verification skipped)"
  exit 0
fi

echo
echo "=== Phase 3: verifying Ray capacity from inside the head pod ==="
# Workers dial the head GCS with up to a 300 s backoff; give them a beat before counting.
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
# Allow a short settle window in case the last few workers are still joining.
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

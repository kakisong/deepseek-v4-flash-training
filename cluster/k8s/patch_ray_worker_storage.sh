#!/usr/bin/env bash
# Re-apply the ray-worker STORAGE fixes after the cluster / DaemonSet is (re)created.
#
# The H200 k8s ray-gpu-worker DaemonSet ships with `ray start --temp-dir=/ray_local/ray`
# and no object-store cap. /ray_local is a small (~100 GB), shared local NVMe; on sub-42
# node runs ray object-spill fills it and kills the workers before step 1. This patch:
#   1. moves ray temp-dir (logs + object spill) to /ray_spill_local on the 286 GB
#      container overlay (local, sockets OK, no k8s ephemeral-storage limit), and
#   2. caps the object store at 230 GB (fits the 256 GB /dev/shm, fewer spills).
# Idempotent: safe to re-run. Rolls the DaemonSet so all 42 workers pick it up.
#
# NOTE: deep_ep is now baked into the image (docker/Dockerfile.sft-only). This script
# does NOT add the old startup `pip install deep_ep ...` line; ensure the DaemonSet's
# image is a build that includes deep_ep (rebuild Dockerfile.sft-only -> push -> set
# the DaemonSet/fleet image). See docs/H200_BOTTLENECK_ANALYSIS.md and the
# h200-k8s-16node-runbook memory.
#
# Usage: cluster/k8s/patch_ray_worker_storage.sh [daemonset-name] [namespace]
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
      # 1. temp-dir + its mkdir from the small /ray_local to the big overlay
      gsub("/ray_local/ray"; "/ray_spill_local/ray")
      # 2. add --object-store-memory before --block, only if not already present
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

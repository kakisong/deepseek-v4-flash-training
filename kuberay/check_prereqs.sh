#!/usr/bin/env bash
# Check whether the current shell is connected to a Kubernetes cluster capable
# of running the draft KubeRay manifests.

set -euo pipefail

NAMESPACE="${KUBERAY_NAMESPACE:-v4-train}"
GPU_NODE_SELECTOR="${KUBERAY_GPU_NODE_SELECTOR:-v4.echo/fleet=h20-16node}"

command -v kubectl >/dev/null || {
  echo "[err] kubectl not found" >&2
  exit 1
}

echo "=== Kubernetes context ==="
kubectl config current-context

echo
echo "=== KubeRay CRDs ==="
kubectl get crd rayclusters.ray.io rayjobs.ray.io >/dev/null
echo "[ok] RayCluster and RayJob CRDs are installed"

echo
echo "=== Namespace ==="
kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || {
  echo "[warn] namespace $NAMESPACE does not exist yet"
}

echo
echo "=== GPU nodes ==="
kubectl get nodes -l "$GPU_NODE_SELECTOR" \
  -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.nvidia\\.com/gpu,CPU:.status.allocatable.cpu,MEM:.status.allocatable.memory

echo
echo "=== Required host paths are expected on every selected node ==="
echo "hostPath /data_train -> pod /data_train"
echo "hostPath /data0      -> pod /ray_local"
echo "hostPath /dev/infiniband -> pod /dev/infiniband"
echo
echo "[ok] static prerequisite checks completed"

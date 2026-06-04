#!/usr/bin/env bash
# Start missing worker containers and join them to the already-running Ray head.
#
# Default behavior is non-destructive:
#   - existing running containers are reused
#   - existing worker raylets already pointing at the current head are left alone
# Use --restart-ray to force ray stop/start inside worker containers.

set -euo pipefail

RESTART_RAY=0
VERIFY=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --restart-ray) RESTART_RAY=1; shift ;;
    --no-verify) VERIFY=0; shift ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
      exit 0 ;;
    *) echo "[err] unknown arg: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

DOCKER_MOUNT_FLAGS=""
for m in "${V4_DOCKER_MOUNTS[@]}"; do
  DOCKER_MOUNT_FLAGS+=" -v $m"
done

echo "=== Verifying Ray head/job server ==="
ssh $SSH_OPTS root@"$V4_RAY_HEAD_IP" "docker exec $V4_CONTAINER ray status >/dev/null"
echo "[ok] Ray head is reachable at $V4_RAY_HEAD_IP:$V4_RAY_PORT / dashboard $V4_DASHBOARD_PORT"
echo

check_node_data0() {
  local IP="$1"
  ssh $SSH_OPTS root@"$IP" "V4_HOST_RAY_LOCAL_DIR='${V4_HOST_RAY_LOCAL_DIR:-/data0}' V4_DATA0_MAX_USE_PCT=${V4_DATA0_MAX_USE_PCT:-95} bash -s" <<'EOF'
set -euo pipefail
host_dir="${V4_HOST_RAY_LOCAL_DIR:-/data0}"
root_src="$(findmnt -n -T / -o SOURCE)"
data_src="$(findmnt -n -T "$host_dir" -o SOURCE 2>/dev/null || true)"
if [[ -z "$data_src" ]]; then
  echo "[err] $host_dir is missing"
  exit 1
fi
if [[ "$data_src" == "$root_src" ]]; then
  echo "[err] $host_dir resolves to root filesystem ($data_src); mount a local data disk before starting Ray"
  exit 1
fi
use_pct="$(df -P "$host_dir" | awk 'NR==2 {gsub(/%/, "", $5); print $5}')"
if [[ -n "$use_pct" && "$use_pct" -ge "$V4_DATA0_MAX_USE_PCT" ]]; then
  echo "[err] $host_dir is ${use_pct}% full; Ray temp/object spill needs local disk headroom"
  exit 1
fi
echo "[ok] $host_dir source=$data_src use=${use_pct}%"
EOF
}

ensure_worker() {
  local IP="$1"
  ssh $SSH_OPTS root@"$IP" "WORKER_IP='$IP' RESTART_RAY='$RESTART_RAY' bash -s" <<EOF
set -euo pipefail

if docker ps -a --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  current_image=\$(docker inspect -f '{{.Config.Image}}' $V4_CONTAINER 2>/dev/null || true)
  current_workdir=\$(docker inspect -f '{{.Config.WorkingDir}}' $V4_CONTAINER 2>/dev/null || true)
  if [[ "\$current_image" != "$V4_IMAGE" || "\$current_workdir" != "$V4_MILES_REPO" ]]; then
    echo "[\$WORKER_IP] replacing container $V4_CONTAINER image=\$current_image workdir=\$current_workdir"
    docker rm -f $V4_CONTAINER >/dev/null
  fi
fi

if docker ps --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  echo "[\$WORKER_IP] reusing running container $V4_CONTAINER"
elif docker ps -a --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  docker start $V4_CONTAINER >/dev/null
  echo "[\$WORKER_IP] started existing container $V4_CONTAINER"
else
  docker run -d --name $V4_CONTAINER \\
      --gpus all \\
      --network host \\
      --shm-size=200g \\
      --ulimit memlock=-1 \\
      --ulimit stack=67108864 \\
      --ipc=host \\
      --privileged \\
      $DOCKER_MOUNT_FLAGS \\
      -e PYTHONPATH=$V4_RUNTIME_PYTHONPATH \\
      -e HF_HOME=$V4_HF_HOME \\
      -e HF_DATASETS_CACHE=$V4_HF_HOME/datasets \\
      -e TRANSFORMERS_CACHE=$V4_HF_HOME/transformers \\
      -e HUGGINGFACE_HUB_CACHE=$V4_HF_HOME/hub \\
      -e CUDA_DEVICE_MAX_CONNECTIONS=1 \\
      -e MASTER_ADDR=$V4_TRAINING_MASTER_IP \\
      -e NCCL_IB_DISABLE=0 \\
      -e TZ=${V4_TZ:-Asia/Shanghai} \\
      -w $V4_MILES_REPO \\
      $V4_IMAGE \\
      sleep infinity >/dev/null
  echo "[\$WORKER_IP] created container $V4_CONTAINER"
fi

docker exec $V4_CONTAINER bash -lc 'pip install -e . --quiet --no-deps --no-build-isolation >/tmp/miles-pip-install.log 2>&1 || { tail -20 /tmp/miles-pip-install.log; exit 1; }'
docker exec $V4_CONTAINER python -c 'import torch, fast_hadamard_transform, tile_kernels; import megatron.core.dist_checkpointing.core as c; from megatron.core.transformer.transformer_config import TransformerConfig; assert c.CONFIG_FNAME == "metadata.json"; assert "dsv4_hc_mult" in TransformerConfig.__dataclass_fields__'

raylet_matches_head=0
if docker exec $V4_CONTAINER bash -lc "ps -eo args | grep '[r]aylet/raylet' | grep -q -- '--gcs-address=$V4_RAY_HEAD_IP:$V4_RAY_PORT'"; then
  raylet_matches_head=1
fi

if [[ "\$RESTART_RAY" == "1" || "\$raylet_matches_head" != "1" ]]; then
  docker exec $V4_CONTAINER bash -lc 'ray stop --force 2>/dev/null || true; mkdir -p $V4_RAY_TEMP_DIR'
  docker exec $V4_CONTAINER ray start \\
      --address=$V4_RAY_HEAD_IP:$V4_RAY_PORT \\
      --node-ip-address=\$WORKER_IP \\
      --num-gpus=$V4_NUM_GPUS_PER_NODE \\
      --temp-dir=$V4_RAY_TEMP_DIR \\
      --disable-usage-stats >/tmp/ray-worker-start.log 2>&1 || { cat /tmp/ray-worker-start.log; exit 1; }
  echo "[\$WORKER_IP] ray worker joined $V4_RAY_HEAD_IP:$V4_RAY_PORT"
else
  echo "[\$WORKER_IP] ray worker already joined current head"
fi
EOF
}

echo "=== Preflight: validating worker local Ray disks ($V4_HOST_RAY_LOCAL_DIR -> $V4_CONTAINER_RAY_LOCAL_DIR) ==="
DATA0_ERR=0
for IP in $V4_WORKER_IPS; do
  if ! out="$(check_node_data0 "$IP" 2>&1)"; then
    echo "[$IP] $out"
    DATA0_ERR=1
  else
    echo "[$IP] $out"
  fi
done
if (( DATA0_ERR != 0 )); then
  echo "[err] fix $V4_HOST_RAY_LOCAL_DIR mount/usage before starting worker containers" >&2
  exit 1
fi
echo

echo "=== Ensuring worker containers + Ray workers (${V4_NUM_NODES}-node fleet) ==="
for IP in $V4_WORKER_IPS; do
  ( ensure_worker "$IP" 2>&1 | tail -4 ) &
done
wait
echo

if (( VERIFY == 1 )); then
  echo "=== Verifying Ray capacity ==="
  EXPECTED_NODES="$V4_EXPECTED_RAY_NODES"
  EXPECTED_GPUS="$V4_EXPECTED_GPUS"
  for attempt in $(seq 1 60); do
    if ssh $SSH_OPTS root@"$V4_RAY_HEAD_IP" "docker exec -i -e EXPECTED_NODES=$EXPECTED_NODES -e EXPECTED_GPUS=$EXPECTED_GPUS -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 $V4_CONTAINER python3 - <<'PY'
import os
import sys
import ray

expected_nodes = int(os.environ['EXPECTED_NODES'])
expected_gpus = int(os.environ['EXPECTED_GPUS'])
ray.init(address='auto', logging_level='ERROR')
alive_nodes = sum(1 for n in ray.nodes() if n.get('Alive'))
gpus = int(ray.cluster_resources().get('GPU', 0))
print(f'alive_nodes={alive_nodes} gpus={gpus} expected_nodes={expected_nodes} expected_gpus={expected_gpus}')
sys.exit(0 if alive_nodes >= expected_nodes and gpus >= expected_gpus else 1)
PY
"; then
      break
    fi
    if [[ "$attempt" == "60" ]]; then
      echo "[err] Ray capacity did not reach expected nodes/GPUs" >&2
      exit 1
    fi
    sleep 2
  done
fi

MONITORING_SYNC="$V4_WORK/monitoring/sync_ray_sd.sh"
if [[ -x "$MONITORING_SYNC" ]]; then
  echo
  echo "=== Syncing ray prometheus service discovery ==="
  "$MONITORING_SYNC"
  echo "[ok] synced $V4_WORK/monitoring/sd/ray.json"
else
  echo
  echo "[warn] monitoring sync script not found or not executable: $MONITORING_SYNC"
fi

echo
echo "=== done ==="

#!/usr/bin/env bash
# Prepare a persistent Ray head/job-server on the master node only.
#
# This is the control-plane path for on-demand training workers:
#   1. keep the master container + Ray dashboard/job server alive
#   2. submit scripts can later start worker containers and join them to this head
#
# It intentionally does not start worker containers.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

DOCKER_MOUNT_FLAGS=""
for m in "${V4_DOCKER_MOUNTS[@]}"; do
  DOCKER_MOUNT_FLAGS+=" -v $m"
done

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

echo "=== Preflight: validating Ray head local Ray disk ($V4_HOST_RAY_LOCAL_DIR -> $V4_CONTAINER_RAY_LOCAL_DIR) ==="
check_node_data0 "$V4_RAY_HEAD_IP"
echo

echo "=== Phase 1: ensuring master container $V4_CONTAINER ==="
ssh $SSH_OPTS root@"$V4_RAY_HEAD_IP" bash <<EOF
set -euo pipefail
if docker ps -a --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  current_image=\$(docker inspect -f '{{.Config.Image}}' $V4_CONTAINER 2>/dev/null || true)
  current_workdir=\$(docker inspect -f '{{.Config.WorkingDir}}' $V4_CONTAINER 2>/dev/null || true)
  if [[ "\$current_image" != "$V4_IMAGE" || "\$current_workdir" != "$V4_MILES_REPO" ]]; then
    echo "[head] replacing container $V4_CONTAINER image=\$current_image workdir=\$current_workdir"
    docker rm -f $V4_CONTAINER >/dev/null
  fi
fi

if docker ps --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  echo "[head] reusing running container $V4_CONTAINER"
elif docker ps -a --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  docker start $V4_CONTAINER >/dev/null
  echo "[head] started existing container $V4_CONTAINER"
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
  echo "[head] created container $V4_CONTAINER"
fi

docker exec $V4_CONTAINER bash -lc 'pip install -e . --quiet --no-deps --no-build-isolation >/tmp/miles-pip-install.log 2>&1 || { tail -20 /tmp/miles-pip-install.log; exit 1; }'
docker exec $V4_CONTAINER python -c 'import torch, fast_hadamard_transform, tile_kernels; import megatron.core.dist_checkpointing.core as c; from megatron.core.transformer.transformer_config import TransformerConfig; assert c.CONFIG_FNAME == "metadata.json"; assert "dsv4_hc_mult" in TransformerConfig.__dataclass_fields__; print("torch=" + torch.__version__ + " fht=ok megatron=dsv4 tile_kernels=ok")'
EOF

RAY_HEAD_SCRIPT=$V4_OUT/.ray_head.sh
cat > "$RAY_HEAD_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export TZ='${V4_TZ:-Asia/Shanghai}'

export RAY_GRAFANA_HOST='${V4_GRAFANA_HOST:-}'
export RAY_GRAFANA_IFRAME_HOST='${V4_GRAFANA_IFRAME_HOST:-}'
export RAY_PROMETHEUS_HOST='${V4_PROMETHEUS_HOST:-}'
export RAY_PROMETHEUS_NAME=Prometheus

_REDIS_HOST='${V4_REDIS_HOST:-}'
_REDIS_PORT='${V4_REDIS_PORT:-6379}'
_REDIS_PASSWORD='${V4_REDIS_PASSWORD:-}'
_CLUSTER_NS='${V4_CLUSTER_NAME:-default}'
REDIS_ARGS=()
if [ -n "\$_REDIS_HOST" ]; then
    export RAY_REDIS_ADDRESS="\$_REDIS_HOST:\$_REDIS_PORT"
    export RAY_external_storage_namespace="\$_CLUSTER_NS"
    [ -n "\$_REDIS_PASSWORD" ] && REDIS_ARGS+=(--redis-password="\$_REDIS_PASSWORD")
    echo "[ray-head] GCS external Redis: \$RAY_REDIS_ADDRESS (ns=\$_CLUSTER_NS)"
fi

ray stop --force 2>/dev/null || true
mkdir -p $V4_RAY_TEMP_DIR
ray start --head \\
    --node-ip-address=$V4_RAY_HEAD_IP \\
    --port=$V4_RAY_PORT \\
    --num-gpus=$V4_NUM_GPUS_PER_NODE \\
    --temp-dir=$V4_RAY_TEMP_DIR \\
    --dashboard-host=0.0.0.0 \\
    --dashboard-port=$V4_DASHBOARD_PORT \\
    --disable-usage-stats "\${REDIS_ARGS[@]}"
EOF
chmod +x "$RAY_HEAD_SCRIPT"

echo
echo "=== Phase 2: starting Ray head ==="
ssh $SSH_OPTS root@"$V4_RAY_HEAD_IP" "docker exec $V4_CONTAINER bash $RAY_HEAD_SCRIPT" 2>&1 | tail -12

echo
echo "=== Phase 3: verifying Ray head ==="
ssh $SSH_OPTS root@"$V4_RAY_HEAD_IP" "docker exec $V4_CONTAINER ray status" 2>&1 | head -30

MONITORING_SYNC="$V4_WORK/monitoring/sync_ray_sd.sh"
if [[ -x "$MONITORING_SYNC" ]]; then
  echo
  echo "=== Phase 4: syncing ray prometheus service discovery ==="
  if "$MONITORING_SYNC"; then
    echo "[ok] synced $V4_WORK/monitoring/sd/ray.json"
  else
    echo "[warn] Ray service discovery is not ready yet; check_infra.sh will validate it later"
  fi
else
  echo
  echo "[warn] monitoring sync script not found or not executable: $MONITORING_SYNC"
fi

echo
echo "=== done ==="
echo "Ray head dashboard: http://$V4_RAY_HEAD_IP:$V4_DASHBOARD_PORT"
echo "Submit endpoint:    http://$V4_RAY_HEAD_IP:$V4_DASHBOARD_PORT"

#!/usr/bin/env bash
# Bring up miles containers + ray cluster on V4_NUM_NODES nodes.
# After completion:
#   - V4_NUM_NODES containers named $V4_CONTAINER are running
#   - master is the ray head, workers have joined
#   - dashboard is reachable at http://$V4_MASTER_IP:$V4_DASHBOARD_PORT
#
# Idempotent: if a container already exists, remove and recreate (to guarantee config consistency).
# Failure: exit immediately on any node failure.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

if [[ -z "${V4_IMAGE:-}" ]]; then
  echo "[err] env.sh failed to load" >&2; exit 1
fi

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Flatten V4_DOCKER_MOUNTS array into docker CLI flags (`-v src:dst -v src:dst ...`).
DOCKER_MOUNT_FLAGS=""
for m in "${V4_DOCKER_MOUNTS[@]}"; do
  DOCKER_MOUNT_FLAGS+=" -v $m"
done

check_node_data0() {
  local IP="$1"
  ssh $SSH_OPTS root@"$IP" "V4_DATA0_MAX_USE_PCT=${V4_DATA0_MAX_USE_PCT:-95} bash -s" <<'EOF'
set -euo pipefail
root_src="$(findmnt -n -T / -o SOURCE)"
data_src="$(findmnt -n -T /data0 -o SOURCE 2>/dev/null || true)"
if [[ -z "$data_src" ]]; then
  echo "[err] /data0 is missing"
  exit 1
fi
if [[ "$data_src" == "$root_src" ]]; then
  echo "[err] /data0 resolves to root filesystem ($data_src); mount a local data disk before starting Ray"
  exit 1
fi
use_pct="$(df -P /data0 | awk 'NR==2 {gsub(/%/, "", $5); print $5}')"
if [[ -n "$use_pct" && "$use_pct" -ge "$V4_DATA0_MAX_USE_PCT" ]]; then
  echo "[err] /data0 is ${use_pct}% full; Ray temp/object spill needs local disk headroom"
  exit 1
fi
echo "[ok] /data0 source=$data_src use=${use_pct}%"
EOF
}

echo "=== Preflight: validating /data0 local disks ==="
DATA0_ERR=0
for IP in $V4_ALL_IPS; do
  if ! out="$(check_node_data0 "$IP" 2>&1)"; then
    echo "[$IP] $out"
    DATA0_ERR=1
  else
    echo "[$IP] $out"
  fi
done
if (( DATA0_ERR != 0 )); then
  echo "[err] fix /data0 mount/usage before starting containers" >&2
  exit 1
fi
echo

start_node_container() {
  local IP="$1"
  echo "[$(date +%H:%M:%S)] [$IP] starting container $V4_CONTAINER"
  ssh -o BatchMode=yes -o ConnectTimeout=10 \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR \
      root@$IP bash <<EOF
set -e
# If already running, remove and recreate (to guarantee config consistency).
if docker ps -a --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  docker rm -f $V4_CONTAINER >/dev/null
fi
docker run -d --name $V4_CONTAINER \
    --gpus all \
    --network host \
    --shm-size=200g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --ipc=host \
    --privileged \
    $DOCKER_MOUNT_FLAGS \
    -e PYTHONPATH=$V4_RUNTIME_PYTHONPATH \
    -e HF_HOME=$V4_HF_HOME \
    -e HF_DATASETS_CACHE=$V4_HF_HOME/datasets \
    -e TRANSFORMERS_CACHE=$V4_HF_HOME/transformers \
    -e HUGGINGFACE_HUB_CACHE=$V4_HF_HOME/hub \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e MASTER_ADDR=$V4_MASTER_IP \
    -e NCCL_IB_DISABLE=0 \
    -e TZ=${V4_TZ:-Asia/Shanghai} \
    -w $V4_MILES_REPO \
    $V4_IMAGE \
    sleep infinity >/dev/null
docker exec $V4_CONTAINER bash -lc 'pip install -e . --quiet --no-deps 2>&1 | tail -1' >/dev/null
# Runtime framework deps are image-baked. Validate them inside every container
# so training does not silently fall back to CFS source checkouts.
docker exec $V4_CONTAINER python -c 'import torch, fast_hadamard_transform, tile_kernels; import megatron.core.dist_checkpointing.core as c; from megatron.core.transformer.transformer_config import TransformerConfig; assert c.CONFIG_FNAME == "metadata.json"; assert "dsv4_hc_mult" in TransformerConfig.__dataclass_fields__; print("torch=" + torch.__version__ + " fht=ok megatron=dsv4 tile_kernels=ok")'
echo "[$IP] container ready"
EOF
}

echo "=== Phase 1: starting containers on $V4_NUM_NODES nodes (parallel) ==="
# pre-flight: make sure miles fork is on CFS (used as the workdir + pip install
# target inside containers). Default V4_MILES_REPO_URL points at our public fork
# (kakisong/miles); override env var to swap (e.g. internal mirror).
: "${V4_MILES_REPO_URL:=https://github.com/kakisong/miles.git}"
if [[ ! -d "$V4_MILES_REPO/.git" ]]; then
  echo "[info] cloning miles fork → $V4_MILES_REPO"
  git clone "$V4_MILES_REPO_URL" "$V4_MILES_REPO" 2>&1 | tail -3
fi
for IP in $V4_ALL_IPS; do
  ( start_node_container "$IP" 2>&1 | tail -3 ) &
done
wait
echo

# Materialize the in-container commands into temp scripts on CFS (shared with all workers).
RAY_HEAD_SCRIPT=$V4_OUT/.ray_head.sh
RAY_WORKER_SCRIPT=$V4_OUT/.ray_worker.sh

cat > "$RAY_HEAD_SCRIPT" <<EOF
#!/usr/bin/env bash
set -e
# Timezone — Ray dashboard / log timestamps read TZ env; default to Asia/Shanghai.
export TZ='${V4_TZ:-Asia/Shanghai}'

# Env vars required for embedding Grafana into the Ray dashboard (only read at ray start).
# HOST is for Ray head internal health checks (curl from inside container, internal network);
# IFRAME_HOST is for the browser (public network via Caddy).
# Any unset field is fine — dashboard just loses that integration.
export RAY_GRAFANA_HOST='${V4_GRAFANA_HOST:-}'
export RAY_GRAFANA_IFRAME_HOST='${V4_GRAFANA_IFRAME_HOST:-}'
export RAY_PROMETHEUS_HOST='${V4_PROMETHEUS_HOST:-}'
export RAY_PROMETHEUS_NAME=Prometheus

# External Redis for GCS (head fault tolerance, optional) — skipped when V4_REDIS_HOST is empty.
# Workers only talk to GCS; they never connect to Redis directly.
_REDIS_HOST='${V4_REDIS_HOST:-}'
_REDIS_PORT='${V4_REDIS_PORT:-6379}'
_REDIS_PASSWORD='${V4_REDIS_PASSWORD:-}'
_CLUSTER_NS='${V4_CLUSTER_NAME:-default}'
REDIS_ARGS=()
if [ -n "\$_REDIS_HOST" ]; then
    export RAY_REDIS_ADDRESS="\$_REDIS_HOST:\$_REDIS_PORT"
    # namespace prefixes GCS keys so multiple Ray clusters can share one Redis without collisions.
    export RAY_external_storage_namespace="\$_CLUSTER_NS"
    [ -n "\$_REDIS_PASSWORD" ] && REDIS_ARGS+=(--redis-password="\$_REDIS_PASSWORD")
    echo "[ray-head] GCS external Redis: \$RAY_REDIS_ADDRESS (ns=\$_CLUSTER_NS)"
fi

ray stop --force 2>/dev/null || true
# Put ray temp dir (logs + spill) on the local NVMe data disk instead of overlayfs.
# Reason: 10.0.8.7 has a 492G root vs 5.8T /data0; spill on / can fill overlay.
mkdir -p /data0/ray
ray start --head \\
    --node-ip-address=$V4_MASTER_IP \\
    --port=$V4_RAY_PORT \\
    --num-gpus=$V4_NUM_GPUS_PER_NODE \\
    --temp-dir=/data0/ray \\
    --dashboard-host=0.0.0.0 \\
    --dashboard-port=$V4_DASHBOARD_PORT \\
    --disable-usage-stats "\${REDIS_ARGS[@]}"
EOF
chmod +x "$RAY_HEAD_SCRIPT"

cat > "$RAY_WORKER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
# First argument is the worker's own IP.
set -e
export TZ='__TZ__'
WORKER_IP="$1"
ray stop --force 2>/dev/null || true
mkdir -p /data0/ray
ray start --address=__MASTER_IP__:__RAY_PORT__ \
    --node-ip-address=$WORKER_IP \
    --num-gpus=__NUM_GPUS__ \
    --temp-dir=/data0/ray \
    --disable-usage-stats
EOF
sed -i "s|__MASTER_IP__|$V4_MASTER_IP|g; s|__RAY_PORT__|$V4_RAY_PORT|g; s|__NUM_GPUS__|$V4_NUM_GPUS_PER_NODE|g; s|__TZ__|${V4_TZ:-Asia/Shanghai}|g" "$RAY_WORKER_SCRIPT"
chmod +x "$RAY_WORKER_SCRIPT"

echo "=== Phase 2: starting ray head on master ==="
ssh $SSH_OPTS root@$V4_MASTER_IP "docker exec $V4_CONTAINER bash $RAY_HEAD_SCRIPT" 2>&1 | tail -10

echo
echo "=== Phase 3: $((V4_NUM_NODES - 1)) worker join ray ==="
for IP in $V4_WORKER_IPS; do
  (
    ssh $SSH_OPTS root@$IP "docker exec $V4_CONTAINER bash $RAY_WORKER_SCRIPT $IP" 2>&1 \
        | grep -E "Ray runtime|connected|failed|usage stats" \
        | head -2 | sed "s/^/[$IP] /"
  ) &
done
wait

echo
echo "=== Phase 4: verifying ray cluster ==="
sleep 3
ssh $SSH_OPTS root@$V4_MASTER_IP "docker exec $V4_CONTAINER ray status" 2>&1 | head -25

MONITORING_SYNC="$V4_WORK/monitoring/sync_ray_sd.sh"
if [[ -x "$MONITORING_SYNC" ]]; then
  echo
  echo "=== Phase 5: syncing ray prometheus service discovery ==="
  "$MONITORING_SYNC"
  echo "[ok] synced $V4_WORK/monitoring/sd/ray.json"
else
  echo
  echo "[warn] monitoring sync script not found or not executable: $MONITORING_SYNC"
fi

echo
echo "=== done ==="
echo "Dashboard: http://$V4_MASTER_IP:$V4_DASHBOARD_PORT"
echo "Master container: docker exec -it $V4_CONTAINER bash"

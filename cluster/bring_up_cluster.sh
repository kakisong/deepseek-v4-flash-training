#!/usr/bin/env bash
# 在 V4_NUM_NODES 台节点上拉起 miles 容器 + ray 集群。
# 完成后:
#   - V4_NUM_NODES 个名为 $V4_CONTAINER 的容器在运行
#   - ray head 已在 $V4_RAY_HEAD_IP 启动,workers 已加入
#   - dashboard 可通过 http://$V4_RAY_HEAD_IP:$V4_DASHBOARD_PORT 访问
#
# 幂等:容器已存在则删除重建(保证配置一致)。
# 失败:任一节点失败立即退出。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

if [[ -z "${V4_IMAGE:-}" ]]; then
  echo "[err] env.sh failed to load" >&2; exit 1
fi

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# 把 V4_DOCKER_MOUNTS 数组展开成 docker CLI 参数(`-v src:dst -v src:dst ...`)。
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

echo "=== Preflight: validating local Ray disks ($V4_HOST_RAY_LOCAL_DIR -> $V4_CONTAINER_RAY_LOCAL_DIR) ==="
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
  echo "[err] fix $V4_HOST_RAY_LOCAL_DIR mount/usage before starting containers" >&2
  exit 1
fi
echo

start_node_container() {
  local IP="$1"
  local docker_gpu_flags="--gpus all"
  if [[ "$IP" == "$V4_RAY_HEAD_IP" && "${V4_RAY_HEAD_NUM_GPUS:-$V4_NUM_GPUS_PER_NODE}" -eq 0 ]]; then
    docker_gpu_flags=""
  fi
  echo "[$(date +%H:%M:%S)] [$IP] starting container $V4_CONTAINER"
  ssh -o BatchMode=yes -o ConnectTimeout=10 \
      -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR \
      root@$IP bash <<EOF
set -e
# 若已在运行则删除重建(保证配置一致)。
if docker ps -a --format '{{.Names}}' | grep -qx '$V4_CONTAINER'; then
  docker rm -f $V4_CONTAINER >/dev/null
fi
docker run -d --name $V4_CONTAINER \
    $docker_gpu_flags \
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
    -e MASTER_ADDR=$V4_TRAINING_MASTER_IP \
    -e NCCL_IB_DISABLE=0 \
    -e TZ=${V4_TZ:-Asia/Shanghai} \
    -w $V4_MILES_REPO \
    $V4_IMAGE \
    sleep infinity >/dev/null
docker exec $V4_CONTAINER bash -lc 'pip install -e . --quiet --no-deps --no-build-isolation 2>&1 | tail -1' >/dev/null
# 运行时框架依赖已固化在镜像里。在每个容器内逐一校验,
# 避免训练悄悄回退到 CFS 上的源码 checkout。
docker exec $V4_CONTAINER python -c 'import torch, fast_hadamard_transform, tile_kernels; import megatron.core.dist_checkpointing.core as c; from megatron.core.transformer.transformer_config import TransformerConfig; assert c.CONFIG_FNAME == "metadata.json"; assert "dsv4_hc_mult" in TransformerConfig.__dataclass_fields__; print("torch=" + torch.__version__ + " fht=ok megatron=dsv4 tile_kernels=ok")'
echo "[$IP] container ready"
EOF
}

echo "=== Phase 1: starting containers on $(wc -w <<< "$V4_ALL_IPS") Ray nodes (parallel) ==="
# 预检:确保 miles fork 已在 CFS 上(容器内用作 workdir + pip install
# 目标)。V4_MILES_REPO_URL 默认指向我们的公开 fork
# (kakisong/miles);要换源(如内部镜像)可覆盖该环境变量。
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

# 把容器内要执行的命令落成 CFS 上的临时脚本(所有 worker 共享)。
RAY_HEAD_SCRIPT=$V4_OUT/.ray_head.sh
RAY_WORKER_SCRIPT=$V4_OUT/.ray_worker.sh

cat > "$RAY_HEAD_SCRIPT" <<EOF
#!/usr/bin/env bash
set -e
# 时区 — Ray dashboard / 日志时间戳读 TZ 环境变量;默认 Asia/Shanghai。
export TZ='${V4_TZ:-Asia/Shanghai}'

# 把 Grafana 嵌入 Ray dashboard 所需的环境变量(只在 ray start 时读取)。
# HOST 给 Ray head 内部健康检查用(容器内 curl,走内网);
# IFRAME_HOST 给浏览器用(经 Caddy 走公网)。
# 任一字段不设也没关系 — dashboard 只是少了对应的集成。
export RAY_GRAFANA_HOST='${V4_GRAFANA_HOST:-}'
export RAY_GRAFANA_IFRAME_HOST='${V4_GRAFANA_IFRAME_HOST:-}'
export RAY_PROMETHEUS_HOST='${V4_PROMETHEUS_HOST:-}'
export RAY_PROMETHEUS_NAME=Prometheus

# GCS 用的外部 Redis(head 容错用,可选)— V4_REDIS_HOST 为空时跳过。
# worker 只跟 GCS 通信,从不直连 Redis。
_REDIS_HOST='${V4_REDIS_HOST:-}'
_REDIS_PORT='${V4_REDIS_PORT:-6379}'
_REDIS_PASSWORD='${V4_REDIS_PASSWORD:-}'
_CLUSTER_NS='${V4_CLUSTER_NAME:-default}'
REDIS_ARGS=()
if [ -n "\$_REDIS_HOST" ]; then
    export RAY_REDIS_ADDRESS="\$_REDIS_HOST:\$_REDIS_PORT"
    # namespace 会作为 GCS key 的前缀,让多个 Ray 集群共用一个 Redis 而互不冲突。
    export RAY_external_storage_namespace="\$_CLUSTER_NS"
    [ -n "\$_REDIS_PASSWORD" ] && REDIS_ARGS+=(--redis-password="\$_REDIS_PASSWORD")
    echo "[ray-head] GCS external Redis: \$RAY_REDIS_ADDRESS (ns=\$_CLUSTER_NS)"
fi

ray stop --force 2>/dev/null || true
# 把 ray 临时目录(日志 + spill)放到本地 NVMe 数据盘而不是 overlayfs。
# 原因:10.0.8.7 根分区只有 492G,本地数据盘有 5.8T;spill 写到 / 会把 overlay 撑满。
mkdir -p $V4_RAY_TEMP_DIR
ray start --head \\
    --node-ip-address=$V4_RAY_HEAD_IP \\
    --port=$V4_RAY_PORT \\
    --num-gpus=$V4_RAY_HEAD_NUM_GPUS \\
    --temp-dir=$V4_RAY_TEMP_DIR \\
    --dashboard-host=0.0.0.0 \\
    --dashboard-port=$V4_DASHBOARD_PORT \\
    --disable-usage-stats "\${REDIS_ARGS[@]}"
EOF
chmod +x "$RAY_HEAD_SCRIPT"

cat > "$RAY_WORKER_SCRIPT" <<'EOF'
#!/usr/bin/env bash
# 第一个参数是 worker 自己的 IP。
set -e
export TZ='__TZ__'
WORKER_IP="$1"
ray stop --force 2>/dev/null || true
mkdir -p __RAY_TEMP_DIR__
ray start --address=__MASTER_IP__:__RAY_PORT__ \
    --node-ip-address=$WORKER_IP \
    --num-gpus=__NUM_GPUS__ \
    --temp-dir=__RAY_TEMP_DIR__ \
    --disable-usage-stats
EOF
sed -i "s|__MASTER_IP__|$V4_RAY_HEAD_IP|g; s|__RAY_PORT__|$V4_RAY_PORT|g; s|__NUM_GPUS__|$V4_NUM_GPUS_PER_NODE|g; s|__RAY_TEMP_DIR__|$V4_RAY_TEMP_DIR|g; s|__TZ__|${V4_TZ:-Asia/Shanghai}|g" "$RAY_WORKER_SCRIPT"
chmod +x "$RAY_WORKER_SCRIPT"

echo "=== Phase 2: starting ray head on master ==="
ssh $SSH_OPTS root@$V4_RAY_HEAD_IP "docker exec $V4_CONTAINER bash $RAY_HEAD_SCRIPT" 2>&1 | tail -10

echo
echo "=== Phase 3: $(wc -w <<< "$V4_WORKER_IPS") workers join ray ==="
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
ssh $SSH_OPTS root@$V4_RAY_HEAD_IP "docker exec $V4_CONTAINER ray status" 2>&1 | head -25

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
echo "Dashboard: http://$V4_RAY_HEAD_IP:$V4_DASHBOARD_PORT"
echo "Master container: docker exec -it $V4_CONTAINER bash"

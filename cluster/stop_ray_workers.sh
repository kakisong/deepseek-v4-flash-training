#!/usr/bin/env bash
# 停掉 worker 的 Ray 运行时并删除 worker 容器,保留 master 上的
# Ray head/job-server 容器。执行后 fleet 回到只剩 head 的
# 空 Ray 控制面状态。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

echo "=== stop worker ray + remove worker containers (master is untouched) ==="
for IP in $V4_WORKER_IPS; do
  (
    ssh $SSH_OPTS root@"$IP" "
      docker exec $V4_CONTAINER ray stop --force 2>/dev/null || true
      docker rm -f $V4_CONTAINER 2>/dev/null || true
      echo '[$IP] worker cleaned'
    " 2>&1 | tail -2
  ) &
done
wait

MONITORING_SYNC="$V4_WORK/monitoring/sync_ray_sd.sh"
if [[ -x "$MONITORING_SYNC" ]]; then
  "$MONITORING_SYNC" || true
fi

echo "=== done ==="

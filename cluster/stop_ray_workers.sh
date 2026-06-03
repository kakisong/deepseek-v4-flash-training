#!/usr/bin/env bash
# Stop worker Ray runtimes and remove worker containers, leaving the master
# Ray head/job-server container up. This returns the fleet to a head-only
# empty Ray control plane.

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

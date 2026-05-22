#!/usr/bin/env bash
# Shut down the ray cluster and remove miles containers across all 8 nodes.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=== stop ray + remove containers (parallel across 8 nodes) ==="
for IP in $V4_ALL_IPS; do
  (
    ssh -o BatchMode=yes -o ConnectTimeout=5 root@$IP "
      docker exec $V4_CONTAINER ray stop --force 2>/dev/null || true
      docker rm -f $V4_CONTAINER 2>/dev/null || true
      echo '[$IP] cleaned'
    " 2>&1 | tail -2
  ) &
done
wait
echo "=== done ==="

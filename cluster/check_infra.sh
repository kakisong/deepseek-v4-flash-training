#!/usr/bin/env bash
# 在训练开跑前校验 Ray 控制面、全量 worker 容量与 monitoring 是否就绪。
# Caddy 是公网 web 入口,属于可选项。

set -euo pipefail

REQUIRE_WORKERS=1
CHECK_CADDY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --head-only) REQUIRE_WORKERS=0; shift ;;
    --with-caddy) CHECK_CADDY=1; shift ;;
    -h|--help)
      sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//; /^set -euo/d'
      exit 0 ;;
    *) echo "[err] unknown arg: $1" >&2; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

echo "=== Ray dashboard/job server ==="
curl --noproxy '*' -fsS "http://$V4_RAY_HEAD_IP:$V4_DASHBOARD_PORT/api/version" >/dev/null
echo "[ok] Ray dashboard API: http://$V4_RAY_HEAD_IP:$V4_DASHBOARD_PORT"

echo
echo "=== Ray capacity ==="
EXPECTED_NODES="$V4_EXPECTED_RAY_NODES"
EXPECTED_GPUS="$V4_EXPECTED_GPUS"
if (( REQUIRE_WORKERS == 0 )); then
  EXPECTED_NODES=1
  EXPECTED_GPUS="$V4_RAY_HEAD_NUM_GPUS"
fi
docker exec -i -e EXPECTED_NODES="$EXPECTED_NODES" -e EXPECTED_GPUS="$EXPECTED_GPUS" -e RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 "$V4_CONTAINER" python3 - <<'PY'
import os
import sys

import ray

expected_nodes = int(os.environ["EXPECTED_NODES"])
expected_gpus = int(os.environ["EXPECTED_GPUS"])
ray.init(address="auto", logging_level="ERROR")
alive_nodes = sum(1 for n in ray.nodes() if n.get("Alive"))
gpus = int(ray.cluster_resources().get("GPU", 0))
print(f"alive_nodes={alive_nodes}/{expected_nodes} gpus={gpus}/{expected_gpus}")
sys.exit(0 if alive_nodes >= expected_nodes and gpus >= expected_gpus else 1)
PY

echo
echo "=== Monitoring containers ==="
docker ps --format '{{.Names}}' | grep -qx prometheus
docker ps --format '{{.Names}}' | grep -qx grafana
echo "[ok] prometheus and grafana containers are running"

echo
echo "=== Monitoring endpoints ==="
curl --noproxy '*' -fsS "http://127.0.0.1:$V4_PROMETHEUS_PORT/promql/-/ready" >/dev/null
curl --noproxy '*' -fsS "http://127.0.0.1:$V4_GRAFANA_PORT/api/health" >/dev/null
echo "[ok] Prometheus and Grafana local endpoints are ready"

MONITORING_SYNC="$V4_WORK/monitoring/sync_ray_sd.sh"
if [[ -x "$MONITORING_SYNC" ]]; then
  "$MONITORING_SYNC"
  echo "[ok] synced Ray Prometheus service discovery"
fi

if (( CHECK_CADDY == 1 )); then
  echo
  echo "=== Caddy routes ==="
  docker ps --format '{{.Names}}' | grep -qx caddy
  curl --noproxy '*' -fsS "http://127.0.0.1:${V4_CADDY_PORT:-8200}/ray/" >/dev/null
  curl --noproxy '*' -fsS "http://127.0.0.1:${V4_CADDY_PORT:-8200}/grafana/api/health" >/dev/null
  curl --noproxy '*' -fsS "http://127.0.0.1:${V4_CADDY_PORT:-8200}/promql/-/ready" >/dev/null
  echo "[ok] Caddy routes are ready"
else
  echo
  echo "[info] Caddy check skipped; pass --with-caddy to validate public web routes"
fi

echo
echo "=== done ==="

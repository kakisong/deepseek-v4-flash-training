#!/usr/bin/env bash
# Sync Ray's Prometheus service-discovery file from the Ray head container to
# the host-side monitoring directory consumed by Prometheus.

set -euo pipefail

: "${V4_CONTAINER:=miles-v4-sft}"
: "${V4_WORK:=/data_train/kaynzhang/v4-sft}"
: "${V4_RAY_TEMP_DIR:=/ray_local/ray}"

SRC="$V4_RAY_TEMP_DIR/prom_metrics_service_discovery.json"
LEGACY_SRC=/data0/ray/prom_metrics_service_discovery.json
TMP_SRC=/tmp/ray/prom_metrics_service_discovery.json
DST="${V4_MONITORING_RAY_SD_FILE:-$V4_WORK/monitoring/sd/ray.json}"
TMP="${DST}.tmp"

mkdir -p "$(dirname "$DST")"

if docker exec "$V4_CONTAINER" test -s "$SRC"; then
  docker exec "$V4_CONTAINER" cat "$SRC" > "$TMP"
elif docker exec "$V4_CONTAINER" test -s "$LEGACY_SRC"; then
  docker exec "$V4_CONTAINER" cat "$LEGACY_SRC" > "$TMP"
elif docker exec "$V4_CONTAINER" test -s "$TMP_SRC"; then
  docker exec "$V4_CONTAINER" cat "$TMP_SRC" > "$TMP"
else
  echo "ray service discovery file not found or empty" >&2
  exit 1
fi

python3 -m json.tool "$TMP" >/dev/null
mv -f "$TMP" "$DST"

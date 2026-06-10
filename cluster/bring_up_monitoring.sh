#!/usr/bin/env bash
# 在 master 节点上拉起 Prometheus + Grafana。
#
# Ray 在 master 容器内的 /ray_local/ray 下写本地日志/spill/服务发现文件,
# 底层由宿主机的 /data0 本地盘承载。
# Prometheus/Grafana 是宿主机层面的长生命周期服务,
# 应在提交训练之前准备就绪。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

: "${V4_PROMETHEUS_PORT:?V4_PROMETHEUS_PORT missing from fleet env}"
: "${V4_GRAFANA_PORT:?V4_GRAFANA_PORT missing from fleet env}"

MON_DIR="$V4_WORK/monitoring"
MONITORING_SYNC_SRC="$SCRIPT_DIR/monitoring/sync_ray_sd.sh"
PROM_CONFIG="$MON_DIR/prometheus.yml"
GRAFANA_DASHBOARD_PROVIDERS="$MON_DIR/grafana-dashboard-providers.yml"
GRAFANA_DATASOURCES="$MON_DIR/grafana-datasources.yml"
GRAFANA_DASHBOARDS="$MON_DIR/dashboards"
PROM_DATA="$MON_DIR/prom-data"
PROM_SD="$MON_DIR/sd"
GRAFANA_DATA="$MON_DIR/grafana-data"

[[ -f "$PROM_CONFIG" ]] || { echo "[err] missing $PROM_CONFIG" >&2; exit 1; }
[[ -f "$GRAFANA_DASHBOARD_PROVIDERS" ]] || { echo "[err] missing $GRAFANA_DASHBOARD_PROVIDERS" >&2; exit 1; }
[[ -f "$GRAFANA_DATASOURCES" ]] || { echo "[err] missing $GRAFANA_DATASOURCES" >&2; exit 1; }
[[ -d "$GRAFANA_DASHBOARDS" ]] || { echo "[err] missing $GRAFANA_DASHBOARDS" >&2; exit 1; }

mkdir -p "$PROM_DATA" "$PROM_SD" "$GRAFANA_DATA"
if [[ -f "$MONITORING_SYNC_SRC" ]]; then
  install -m 0755 "$MONITORING_SYNC_SRC" "$MON_DIR/sync_ray_sd.sh"
  echo "[info] sync_ray_sd.sh -> $MON_DIR/sync_ray_sd.sh"
fi

echo "=== Sync Ray service discovery if Ray head is already up ==="
if [[ -x "$MON_DIR/sync_ray_sd.sh" ]]; then
  "$MON_DIR/sync_ray_sd.sh" || echo "[warn] Ray service discovery is not ready yet; run sync_ray_sd.sh after Ray starts"
else
  echo "[warn] missing executable $MON_DIR/sync_ray_sd.sh"
fi

echo
echo "=== Start Prometheus ==="
if docker ps -a --format '{{.Names}}' | grep -qx prometheus; then
  docker rm -f prometheus >/dev/null
fi
docker run -d --name prometheus \
    --restart unless-stopped \
    -p "$V4_PROMETHEUS_PORT":9090 \
    -v "$PROM_CONFIG":/etc/prometheus/prometheus.yml:ro \
    -v "$PROM_SD":/etc/prometheus/sd:ro \
    -v "$PROM_DATA":/prometheus \
    -e TZ="${V4_TZ:-Asia/Shanghai}" \
    prom/prometheus:v2.55.0 \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.console.libraries=/usr/share/prometheus/console_libraries \
    --web.console.templates=/usr/share/prometheus/consoles \
    --web.listen-address=:9090 \
    --web.enable-lifecycle \
    --web.external-url="${V4_PROMETHEUS_EXTERNAL_URL:-https://kaynzhang.woa.com/promql}" \
    --web.route-prefix=/promql \
    >/dev/null

echo "=== Start Grafana ==="
if docker ps -a --format '{{.Names}}' | grep -qx grafana; then
  docker rm -f grafana >/dev/null
fi
docker run -d --name grafana \
    --restart unless-stopped \
    -p "$V4_GRAFANA_PORT":3000 \
    -v "$GRAFANA_DATA":/var/lib/grafana \
    -v "$GRAFANA_DASHBOARDS":/etc/grafana/dashboards:ro \
    -v "$GRAFANA_DASHBOARD_PROVIDERS":/etc/grafana/provisioning/dashboards/providers.yml:ro \
    -v "$GRAFANA_DATASOURCES":/etc/grafana/provisioning/datasources/datasources.yml:ro \
    -e GF_SECURITY_ADMIN_USER=admin \
    -e GF_SECURITY_ADMIN_PASSWORD=admin \
    -e GF_AUTH_ANONYMOUS_ENABLED=true \
    -e GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer \
    -e GF_AUTH_DISABLE_LOGIN_FORM=false \
    -e GF_USERS_ALLOW_SIGN_UP=false \
    -e GF_SECURITY_ALLOW_EMBEDDING=true \
    -e GF_SERVER_SERVE_FROM_SUB_PATH=true \
    -e GF_SERVER_ROOT_URL="${V4_GRAFANA_ROOT_URL:-https://kaynzhang.woa.com/grafana/}" \
    -e GF_DATE_FORMATS_DEFAULT_TIMEZONE="${V4_TZ:-Asia/Shanghai}" \
    -e TZ="${V4_TZ:-Asia/Shanghai}" \
    grafana/grafana:11.3.0 \
    >/dev/null

sleep 2
curl --noproxy '*' -fsS "http://127.0.0.1:$V4_PROMETHEUS_PORT/promql/-/ready" >/dev/null
curl --noproxy '*' -fsS "http://127.0.0.1:$V4_GRAFANA_PORT/api/health" >/dev/null

echo
echo "=== done ==="
echo "Prometheus: http://$V4_RAY_HEAD_IP:$V4_PROMETHEUS_PORT/promql"
echo "Grafana:    http://$V4_RAY_HEAD_IP:$V4_GRAFANA_PORT/grafana"

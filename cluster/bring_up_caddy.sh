#!/usr/bin/env bash
# 在 master 上拉一个 caddy 容器作为统一 web 入口。
# 监听 V4_CADDY_PORT,内部转发到 Ray dashboard / Grafana / Prometheus。
#
# 用法:
#   bash examples/deepseek_v4_sft/cluster/bring_up_caddy.sh
#
# 幂等:容器已存在则删旧启新。
# 跑完后:
#   - 本地 curl http://localhost:$V4_CADDY_PORT/ray/ 能拿到 Ray dashboard
#   - WOA 前置跳板机转发目标改成 $V4_MASTER_IP:$V4_CADDY_PORT 后,
#     https://kaynzhang.woa.com/ray|/grafana|/promql 三条都通

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

: "${V4_CADDY_PORT:?cluster env 缺 V4_CADDY_PORT}"

CADDY_RUNTIME=$V4_WORK/caddy
mkdir -p "$CADDY_RUNTIME"/{data,config,log,static}

# workspace 的 Caddyfile 同步到 CFS,容器只读这份
cp "$SCRIPT_DIR/caddy/Caddyfile" "$CADDY_RUNTIME/Caddyfile"
echo "[info] Caddyfile -> $CADDY_RUNTIME/Caddyfile"

# 静态文档同步到 caddy/static —— /docs/* 路径 serve 这里。
# 后续加页面:在 DOCS_FILES 加一行(相对 repo docs/ 的文件名)。
DOCS_DIR="$SCRIPT_DIR/../docs"
DOCS_FILES=(
  "WRITEUP.html"
  "v4_training.html"
  "QA_FIT_REPORT.html"
  "THROUGHPUT_REPORT.html"
  "EVAL_PLAN.html"
  # 例:加新页面只需在这里追加一行
  # "RUNBOOK.html"
  # "subdir/foo.html"
)

for rel in "${DOCS_FILES[@]}"; do
  src="$DOCS_DIR/$rel"
  dst="$CADDY_RUNTIME/static/$rel"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
    echo "[info] docs sync: $rel -> $dst"
  else
    echo "[warn] $src 不存在,/docs/$rel 将 404"
  fi
done

if docker ps -a --format '{{.Names}}' | grep -qx caddy; then
  docker rm -f caddy >/dev/null
fi

docker run -d --name caddy \
    --restart unless-stopped \
    --network host \
    -v "$CADDY_RUNTIME/Caddyfile":/etc/caddy/Caddyfile:ro \
    -v "$CADDY_RUNTIME/data":/data \
    -v "$CADDY_RUNTIME/config":/config \
    -v "$CADDY_RUNTIME/log":/var/log/caddy \
    -v "$CADDY_RUNTIME/static":/srv/docs:ro \
    caddy:2.8-alpine \
    >/dev/null

sleep 1
if ! docker ps --format '{{.Names}}' | grep -qx caddy; then
  echo "[err] caddy 启动后没看到容器,docker logs caddy:" >&2
  docker logs caddy 2>&1 | tail -20 >&2
  exit 1
fi

echo "[ok] caddy 容器跑起来,listen :$V4_CADDY_PORT (host network)"
echo
echo "本地自测(若机器装了 squid proxy,加 --noproxy '*'):"
echo "  curl -sI http://localhost:$V4_CADDY_PORT/ray/"
echo "  curl -sI http://localhost:$V4_CADDY_PORT/grafana/"
echo "  curl -sI http://localhost:$V4_CADDY_PORT/promql/"
echo "  curl -sI http://localhost:$V4_CADDY_PORT/docs/             # 列目录"
echo "  curl -sI http://localhost:$V4_CADDY_PORT/docs/WRITEUP.html # 项目实录"
echo
echo "公网入口(经 WOA 跳板机 + Caddy):"
echo "  $V4_PUBLIC_URL/ray       — Ray dashboard"
echo "  $V4_PUBLIC_URL/grafana   — Grafana"
echo "  $V4_PUBLIC_URL/promql    — Prometheus"
echo "  $V4_PUBLIC_URL/docs/     — 文档目录(browse)"
echo "  $V4_PUBLIC_URL/docs/WRITEUP.html — 项目实录"
echo
echo "下一步:WOA 后台把前置跳板机转发目标改成 $V4_MASTER_IP:$V4_CADDY_PORT"

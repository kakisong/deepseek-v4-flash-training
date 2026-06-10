#!/usr/bin/env bash
# 跨 8 节点分布式执行 HF BF16 → Megatron torch_dist 转换。
# 配置:TP=1 PP=8 EP=4(PR #1045 默认的 8 节点配置)。

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "$SCRIPT_DIR/env.sh"

SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

[[ -d "$V4_BF16_DIR" ]] || { echo "[err] BF16 dir missing: $V4_BF16_DIR" >&2; exit 1; }
[[ -f "$V4_BF16_DIR/model.safetensors.index.json" ]] || { echo "[err] BF16 not finished casting" >&2; exit 1; }

ssh $SSH_OPTS root@$V4_RAY_HEAD_IP "docker exec $V4_CONTAINER ray status" 2>&1 | grep -qiE "active|node_" || {
  echo "[err] ray cluster not healthy. cluster_up.sh first." >&2
  exit 1
}

if [[ -f "$V4_TORCH_DIST/latest_checkpointed_iteration.txt" ]]; then
  echo "[skip] $V4_TORCH_DIST already exists"
  exit 0
fi

mkdir -p "$V4_TORCH_DIST"
echo "[info] target: $V4_TORCH_DIST"
echo "[info] using TP=1 PP=8 EP=4 across 8 nodes"

# 把转换用的 python 调用落成脚本文件,在容器内执行。
# Monkey-patch exec_command_all_ray_node,在每个节点的命令前注入 NCCL/分布式调试环境变量,
# 这样下次 init_process_group 卡死时(我们曾为此损失过 91 分钟)
# 就能准确看到 NCCL 握手卡在哪一步。
CONV_PY=$V4_OUT/.convert_v4.py
cat > "$CONV_PY" <<EOF
import miles.utils.misc as _misc
import miles.utils.external_utils.command_utils as _cu

_DEBUG_ENV = (
    # 转换阶段禁用 IB — 宿主机的 `ibv_reg_mr_iova2` 在我们的 IB 驱动下
    # 返回 Invalid argument。TP=ETP=1,scatter group_size=1(没有真正的跨 rank 流量),
    # 回退到 socket 只多花初始化时间,不损失性能。
    "NCCL_IB_DISABLE=1 "
    "NCCL_NET_GDR_LEVEL=0 "
    "NCCL_DEBUG=WARN "
    "NCCL_SOCKET_IFNAME=eth0 "
    "GLOO_SOCKET_IFNAME=eth0 "
    "TORCH_NCCL_BLOCKING_WAIT=1 "
    "TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
)

_orig = _misc.exec_command_all_ray_node

def _patched(cmd, *a, **kw):
    cmd = cmd.replace("PYTHONPATH=", _DEBUG_ENV + "PYTHONPATH=", 1)
    return _orig(cmd, *a, **kw)

_misc.exec_command_all_ray_node = _patched
_cu.exec_command_all_ray_node = _patched

_cu.convert_checkpoint(
    model_name="DeepSeek-V4-Flash-FP8",
    megatron_model_type="deepseek-v4-flash",
    num_gpus_per_node=$V4_NUM_GPUS_PER_NODE,
    multinode=True,
    num_nodes=$V4_NUM_NODES,
    extra_args=(
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 8 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--context-parallel-size 1 "
        "--decoder-first-pipeline-num-layers 7 "
        "--decoder-last-pipeline-num-layers 6 "
    ),
    dir_dst="$V4_MODELS",
    hf_checkpoint="$V4_BF16_DIR",
    # 把 mbridge_debug(带 MILES-DEBUG scatter 打印的 CFS 影子版 mbridge)放到最前面,
    # 让所有 worker 节点先于 pip 安装的 mbridge 导入打过补丁的 bridge.py。
    megatron_path="$V4_WORK/mbridge_debug:$V4_MEGATRON",
)
EOF

CONV_SH=$V4_OUT/.convert_v4.sh
# 使用带调试打印的 CFS 影子版 mbridge(针对 scatter shape 不匹配加了 MILES-DEBUG),
# 让全部 8 个节点先于 pip 安装版导入打过补丁的 mbridge。
MBRIDGE_DEBUG=$V4_WORK/mbridge_debug
cat > "$CONV_SH" <<EOF
#!/usr/bin/env bash
set -e
cd $V4_MILES
PYTHONPATH=$MBRIDGE_DEBUG:$V4_MEGATRON python $CONV_PY
EOF
chmod +x "$CONV_SH"

echo "[info] launching: $CONV_SH"
ssh $SSH_OPTS root@$V4_RAY_HEAD_IP "docker exec $V4_CONTAINER bash $CONV_SH" 2>&1 | tee "$V4_OUT/.convert_v4.log"
echo
echo "[done] check $V4_TORCH_DIST"
ls "$V4_TORCH_DIST" 2>/dev/null | head

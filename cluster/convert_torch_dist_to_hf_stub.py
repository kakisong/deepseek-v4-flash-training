"""miles tools/convert_torch_dist_to_hf.py 的驱动脚本:一次 CPU 加载完成剥离 optimizer + 导出 bf16 HF。

为什么需要这个 wrapper(而不是直接调用 miles 工具):
  * miles/.../backends/megatron_utils/sglang.py 在模块加载时硬性 import sglang
    (MultiprocessingSerializer、patch_torch.monkey_patch、FlattenedTensorBucket),尽管
    bf16 导出根本不会调用 quantizer。训练镜像有意不带 sglang,
    这个 import 会直接崩溃。我们预先往 sys.modules 塞一个最小 sglang stub,
    只满足那些硬性依赖的符号。
  * convert_torch_dist_to_hf 的 EmptyStateDictLoadPlanner 在加载时会跳过 key 中含 "optimizer" 或
    "_state" 的条目 -> optimizer 状态(约 3478GB)和所有 _extra_state 顺带被丢弃。
    因此对完整 iter_XXXXXXX checkpoint 做这一次 CPU 加载,就能直接得到去掉 optimizer 的
    bf16 HF 权重;无需单独的 torch_dist 剥离 optimizer 步骤。

单进程,无 GPU / 无分布式。会在 CPU 内存中聚合约 569GB 模型权重 -> 请在
空闲内存 >=600GB 的节点上运行(空闲的 GPU 节点约有 1.6TB)。参数由
cluster/convert_torch_dist_to_hf.sh 设置的环境变量驱动。
"""
import os
import sys
import types

# --- 参数(环境变量驱动;由 .sh 启动器填入,默认值可复现 iter_1164 -> kaynzhang_128k)---
INPUT = os.environ["CONV_INPUT_DIR"]            # 完整 torch_dist ckpt 目录(iter_XXXXXXX,权重+optimizer)
OUTPUT = os.environ["CONV_OUTPUT_DIR"]          # 目标 HF bf16 目录
MILES = os.environ.get("V4_MILES", "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles")
HF_TEMPLATE = os.environ.get(
    "CONV_HF_TEMPLATE",
    "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-bf16-unpacked",
)
VOCAB = os.environ.get("CONV_VOCAB_SIZE", "129280")
LOG = os.environ.get("CONV_LOG", OUTPUT.rstrip("/") + ".convert.log")

# 我们通过 `ray job submit --no-wait` 提交,而 ray 的 job-logs API 在部分节点上不可靠
# (port -1)-> 把整个运行重定向到 fsx 日志文件,保证进度随时可查。
os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
_log = open(LOG, "w", buffering=1)
sys.stdout = sys.stderr = _log


def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# 最小 sglang stub:只提供 sglang.py 无条件 import 硬性依赖的那几个符号
# (MultiprocessingSerializer;patch_torch.monkey_patch;tensor_bucket.FlattenedTensorBucket)。
# fp8_utils 的 import 仍不满足 -> 其 try/except 会把它们置为 None,bf16 路径从不触碰。
_mod("sglang")
_mod("sglang.srt")
_u = _mod("sglang.srt.utils")
_u.MultiprocessingSerializer = type("MultiprocessingSerializer", (), {})
_pt = _mod("sglang.srt.utils.patch_torch")
_pt.monkey_patch_torch_reductions = lambda *a, **k: None
_mod("sglang.srt.weight_sync")
_tb = _mod("sglang.srt.weight_sync.tensor_bucket")
_tb.FlattenedTensorBucket = type("FlattenedTensorBucket", (), {})
print("[stub] sglang stub installed", flush=True)

CONVERT = os.path.join(MILES, "tools", "convert_torch_dist_to_hf.py")
sys.argv = [
    CONVERT,
    "--model-name", "deepseekv4",
    "--input-dir", INPUT,
    "--output-dir", OUTPUT,
    "--origin-hf-dir", HF_TEMPLATE,
    "--vocab-size", VOCAB,
    "--force",
]
print(f"[convert] argv={sys.argv[1:]}", flush=True)

import runpy

runpy.run_path(CONVERT, run_name="__main__")
print("[convert] DONE", flush=True)

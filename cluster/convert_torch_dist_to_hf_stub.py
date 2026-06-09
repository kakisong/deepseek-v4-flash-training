"""Driver for miles tools/convert_torch_dist_to_hf.py: strip optimizer + emit bf16 HF in one CPU pass.

Why this wrapper exists (vs. calling the miles tool directly):
  * miles/.../backends/megatron_utils/sglang.py hard-imports sglang at module-load time
    (MultiprocessingSerializer, patch_torch.monkey_patch, FlattenedTensorBucket) even though the
    bf16 export never calls the quantizer. The training image deliberately ships WITHOUT sglang,
    so that import would crash. We pre-populate sys.modules with a minimal sglang stub that
    satisfies only those hard-required symbols.
  * convert_torch_dist_to_hf's EmptyStateDictLoadPlanner skips any key containing "optimizer" or
    "_state" at load time -> the optimizer state (~3478GB) AND every _extra_state are dropped for
    free. So this single CPU pass over a FULL iter_XXXXXXX checkpoint yields de-optimized bf16 HF
    weights directly; no separate torch_dist optimizer-strip step is needed.

Single process, no GPU / no distributed. Aggregates ~569GB of model weights in CPU RAM -> run on a
node with >=600GB free (a free GPU node has ~1.6TB). Driven by env vars set by
cluster/convert_torch_dist_to_hf.sh.
"""
import os
import sys
import types

# --- params (env-driven; the .sh launcher fills these, defaults reproduce iter_1164 -> kaynzhang_128k) ---
INPUT = os.environ["CONV_INPUT_DIR"]            # full torch_dist ckpt dir (iter_XXXXXXX, weights+optim)
OUTPUT = os.environ["CONV_OUTPUT_DIR"]          # destination HF bf16 dir
MILES = os.environ.get("V4_MILES", "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles")
HF_TEMPLATE = os.environ.get(
    "CONV_HF_TEMPLATE",
    "/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-bf16-unpacked",
)
VOCAB = os.environ.get("CONV_VOCAB_SIZE", "129280")
LOG = os.environ.get("CONV_LOG", OUTPUT.rstrip("/") + ".convert.log")

# We submit via `ray job submit --no-wait` and the ray job-logs API is unreliable on some nodes
# (port -1) -> redirect the whole run to an fsx log file so progress is always inspectable.
os.makedirs(os.path.dirname(LOG) or ".", exist_ok=True)
_log = open(LOG, "w", buffering=1)
sys.stdout = sys.stderr = _log


def _mod(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# Minimal sglang stub: only the symbols hard-required by sglang.py's unconditional imports
# (MultiprocessingSerializer; patch_torch.monkey_patch; tensor_bucket.FlattenedTensorBucket). The
# fp8_utils imports stay unsatisfied -> their try/except sets them to None, which bf16 never touches.
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

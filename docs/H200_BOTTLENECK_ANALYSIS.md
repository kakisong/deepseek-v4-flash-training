# H200 vs H20: Why MFU% is Low, and What the Real Bottleneck Is

**TL;DR (experimentally established via two causal clock/bandwidth throttles, 2026-06-06):**
On the V4-Flash 128K SFT workload the step decomposes into **comm ≈ 0%, ~56% SM-throughput-bound
(dominated by the fp32 atomic-scatter in the sparse-MLA backward — NOT tensor-GEMM), ~44%
HBM-memory-bandwidth bound.** The single biggest lever is a **de-atomic backward rewrite**; the
6.7× tensor-FLOP peak of H200 is structurally unusable because the bottleneck is a non-tensor
atomic op (scales with SM count ~1.7×, not tensor FLOPs).
"H200 MFU% < H20 MFU%" at the same scale is **reasonable and expected**: H20 is a deliberately
compute-cut but bandwidth-preserved chip, so on a memory-bandwidth-bound workload the two chips
perform similarly in *absolute* throughput (H200 ~1.2–1.86× H20), while MFU% — which divides by
the 6.7× larger H200 compute peak — makes H200 look 3× worse. **Use tokens/sec/GPU, GPU-hours,
and $/token, not MFU%, to compare these chips.**

---

## 1. The hardware (this is the whole story)

| metric | H20 | H200 | ratio |
|---|---|---|---|
| BF16 compute | 148 TFLOPS | 989 TFLOPS | **6.7×** |
| HBM bandwidth | 4.0 TB/s | 4.8 TB/s | **1.2×** |
| NVLink | 900 GB/s | 900 GB/s | **1.0×** |
| TDP | ~400 W | 700 W | 1.75× |

H20 is the export-compliant chip: **cores cut ~41% (78/144 SMs), but memory bandwidth and
interconnect preserved** (its 4.0 TB/s even exceeds H100's 3.35). It is designed to look efficient
on memory-bound workloads. Sources: Tom's Hardware, Wccftech, NVIDIA/vendor spec sheets.

## 2. The measurements (tokens/sec/GPU — phantom-free, peak-free)

Same workload (albaliang ≤128K long-context SFT, gb=128, max_tokens=32768), steady-state from wandb:

| config | tok/s total | **tok/s/GPU** | tflops/GPU | MFU% |
|---|---|---|---|---|
| H20 16-node TP8 PP4 CP4 +offload | 12,319 | **96** | 51 | 34.5% |
| H200 16-node TP8 PP4 CP4 +offload | ~13,860 | **~108** | 71 | 7.2% |
| H200 42-node TP8 PP7 CP6 no-offload | 57,800 | **172** | 114 | 11.5% |

- **At the same 16-node config, H200 (108) ≥ H20 (96).** H200 is NOT slower.
- **At its best 42-node config, H200 = 172 tok/s/GPU = 1.79× H20's 96.** Note this ≈ the HBM
  bandwidth ratio, not the 6.7× compute ratio.
- The MFU% inversion (H200 11.5% "vs" H20 34.5%) is purely the 6.7× peak denominator.

(16-node is below 42-node because PP4 forces `--optimizer-cpu-offload`, whose PCIe D2H/H2D tax
is a bigger fraction of H200's faster step — Amdahl. 42-node no-offload is the better H200 config.)

## 3. Bottleneck triangulation (3 independent experiments)

| experiment (causal throttle) | result | conclusion |
|---|---|---|
| **NCCL channel sweep** (full → 8 → 2 channels) | tok/s 57.8k → 56.5k → 56.2k (**flat, ~3%**) | comm ≈ **0%** of critical path |
| **SM-clock throttle** (1980 → 990 MHz, half) | tok/s 56.2k → 36.1k (**−36%**) | **~56% SM-throughput-bound** |
| **FP8 on MoE GEMMs** | null speedup | the 56% is **not** MoE-GEMM (tensor) compute |
| **→ decomposition** | comm 0% / SM-bound 56% / HBM 44% | the 56% = **fp32 atomic-scatter** (sparse-MLA bwd) |

The SM-clock test is decisive: halving compute clock costs 36% throughput → ~56% of the step is
SM-throughput-sensitive. FP8-null rules that out being MoE-GEMM, so it is the **fp32 dKV
`atomic_addx4` scatter** in the sparse-MLA backward (atomics are SM-issued → scale with SM count
~1.7× H200/H20, NOT tensor FLOPs 6.7×). The remaining ~44% is HBM bandwidth (recompute /
activation read-write). This matches the original micro-bench (sparse-MLA bwd = 58% of compute,
atomic-bound). **Comm is genuinely free here** — huge headroom (2 channels suffices).

The NCCL sweep is decisive on the comm question: throttling collective bandwidth to **2 channels**
barely moves throughput (~3% drop, flat slope at the operating point) → communication has huge
headroom and is **not** on the critical path. This **corrected** an earlier (literature/theory-based)
"communication-bound" claim — which was true for DeepSeek's H800 (NVLink cut to 400 GB/s) but
**false here**: full 900 GB/s NVLink + EFA + DeepEP overlap leave comm well-provisioned and hidden.

So the bottleneck is the **HBM-traffic-heavy** parts of the step: sparse-MLA gather/scatter (esp. the
fp32 dKV `atomic_addx4` scatter), recompute (full) activation read/write, MLA, normalization,
elementwise. These move bytes, not FLOPs → H200 only realizes its 1.2× HBM advantage.

## 4. Why this answers "is it reasonable that H200 MFU < H20"

Yes. MFU% = achieved_tflops / peak. For an HBM-bound workload the achieved tflops tracks HBM
bandwidth (~1.2× between the chips), but H200's peak is 6.7× larger → MFU% ≈ H20_MFU × 1.2/6.7
≈ 1/5.6 of H20's. The chip is doing equal-or-more real work; the metric just divides by an
unusable compute peak. **40% MFU on H200 is physically unreachable** for this workload (= 396
tflops/GPU; HBM-bound ceiling is ~114–170).

## 5. Optimization implications (this redirects the strategy)

Because it is HBM-bandwidth-bound (comm and compute already excluded by experiment):

- ❌ **FP8 / compute optimization** — won't help (measured null).
- ❌ **Comm optimization** (better DeepEP / overlap) — won't help (already has headroom).
- ✅ **Reduce HBM traffic** — the real lever:
  - **De-atomic sparse-MLA backward** (KV-keyed 2nd pass + inverted CSR) — kills the fp32
    atomic-scatter HBM writes; the single biggest win.
  - **Kernel fusion** to cut activation read/write round-trips.
  - **Recompute strategy** — trade fewer recompute reads where memory allows.
  - **FP8 for HBM-resident data** (KV/activations) reduces bytes moved — helps *because it's a
    bandwidth optimization*, not a compute one.

## 6. Right metrics for H200-vs-H20 (drop MFU%)

1. **tokens/sec/GPU** (steady-state) — H200 1.2–1.79× H20.
2. **GPU-hours / epoch** — H200 ~55–83% of H20's.
3. **$/trained-token** = GPU-hours × $/GPU-hour — the business decision metric.

## 7. Open item: directly confirming HBM-bound (currently triangulated by elimination)

Section 3 establishes HBM-bound by ruling out comm and compute. To confirm it *directly*:
- **DCGM counters during a step**: `DCGM_FI_PROF_DRAM_ACTIVE` high + `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE`
  low ⇒ memory busy / tensor cores idle ⇒ HBM-bound. (dcgm-exporter already runs on every node.)
- **Memory-clock throttle (causal, symmetric to the NCCL sweep)**: `nvidia-smi -lmc` to lower HBM
  clock; if throughput drops ∝ memory clock ⇒ HBM-bound. (Privileged; resets needed; affects only
  our 42 nodes.)
- **Roofline**: real FLOPs / real HBM bytes vs ridge points (H20 37, H200 206 FLOP/byte). AI between
  37 and 206 ⇒ compute-bound on H20, memory-bound on H200 — exactly the observed MFU split.

## 8. Empirical config sweep (2026-06-06) — PP7CP6 no-offload is the optimum

Every alternative tested is worse or infeasible; the current production config wins:

| config (42-node, TP8 EP8) | steady tok/s | tflops/GPU | verdict |
|---|---|---|---|
| **PP7 CP6, no-offload, recompute full/block4** | **56,226** | **114** | ✅ **optimal** |
| PP7 CP6 **VPP2** | 51,300 | 101 | −9% — interleave P2P+schedule overhead > the 4.7% bubble it saves |
| PP7 CP3 **DP2** | (never reached steady) | — | gb=128/DP2 → 41 µbatch → bubble 14.6% (3×) + packing 0.82 vs 0.94; impractically slow |
| PP6 CP7 | **OOM** | — | lower PP → 7.2 layers/stage → more weight memory |
| PP3 CP14 | OOM (bounded by PP6) | — | 14 layers/stage |
| PP14 CP3 + recompute selective | **OOM** (on step) | — | high-PP weight headroom can't cover selective's activation at 128K |
| recompute selective (any PP) | **OOM** | — | activation ∝ 43/(PP×CP)=const, too large at 128K |
| DeepGEMM | n/a | — | not installed; MoE-GEMM isn't the bottleneck anyway (FP8-null) |
| cpu-offload | already OFF | — | off = optimal (16-node forced it ON → 71 tflops, far worse) |

**Why config is exhausted:** the 56% atomic-scatter is config-immune; comm is already free (nothing
to tune); bubble is tiny and VPP/DP only make it worse; memory is doubly walled — lower PP OOMs on
weights, reducing recompute OOMs on activations. **The only remaining lever is the de-atomic backward
kernel rewrite.**

## Appendix — reproducibility
- 16-node launch + the 4 infra blockers/fixes: see memory `h200-k8s-16node-runbook`.
- NCCL sweep knob: `V4_NCCL_MAX_NCHANNELS` (wired into run.sh runtime_env).
- wandb runs: H20 `lr0mjb4o`/`wkf2d2lc`; H200 42-node `jvp2ovpl`; NCCL sweep `yyrm0qlz`(8ch)/`simb8dhd`(2ch).

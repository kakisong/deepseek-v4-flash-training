# H200 vs H20: Why MFU% is Low, and What the Real Bottleneck Is

**TL;DR (causal throttles for comm/compute + DIRECT kernel profiling for the bwd, 2026-06-06):**
On the V4-Flash 128K SFT workload, **comm ≈ 0%** (NCCL sweep flat) and the step is **never
compute-bound** (SM ~16%). The dominant cost is the **sparse-MLA backward**, which is **one kernel**
(bwd-main = 99% of the backward, 9× the forward) that is **memory-hierarchy-bound on its fp32 dKV
`atomic_addx4` scatter**: directly measured as **L2-reduction-throughput bound (~60% L2, 5–33% DRAM,
16% SM)**. A causal store-ablation removing the atomic recovers **31.5% (fp32) – 43.4% (bf16) of
bwd-main**. The single biggest lever is a **de-atomic backward rewrite** (§7). H200's 6.7× tensor-FLOP
peak is structurally unusable because the bottleneck is a non-tensor atomic-reduction op
(L2/SM-issued, scales ~1.7× not 6.7×). **Verified directly, not by elimination — see §7.**
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

Because bwd-main is L2-atomic-reduction bound on the dKV scatter (comm/compute excluded by
experiment, §3; resource confirmed by ncu, §7):

- ❌ **FP8 / compute optimization** — won't help (measured null; SM only 16%).
- ❌ **Comm optimization** (better DeepEP / overlap) — won't help (already has headroom).
- ❌ **bf16 dKV reduction** — won't help at L2-resident sizes: we are atomic-**op-count** bound, and
  bf16x2 needs 2× the atomic ops for the same values (only helps bytes in the large-S DRAM-spill regime).
- ⚠️ **De-atomic via KV-centric inversion** — the obvious idea, but **backfires here** (see §9): this is
  MQA (1 KV head shared by 64 query heads), so KV-centric would gather Q/dO with a **64× head
  penalty** — far more traffic than the atomic it removes. The §7 ablation's "31–43% removable" is an
  *idealized floor* (it kept the efficient query-centric GEMMs and a free write); it is **not** directly
  achievable. Real de-atomic ceiling ≈ 6–15% via a window-dense hybrid / query-block grouping (hard).
- ✅ **Available now (validated, low-risk, ~6%)**: `block_H=64` (NH=1, halves the dKV op_red count) +
  direct `acc_dq`→global store (frees the 64 KB `dQ_shared` so smem fits). Measured 12.90→12.11 ms
  (1.06×) at S=4096; gradients bit-exact for dq, rel 1.8e-6 for dkv vs production. See §9.

## 6. Right metrics for H200-vs-H20 (drop MFU%)

1. **tokens/sec/GPU** (steady-state) — H200 1.2–1.79× H20.
2. **GPU-hours / epoch** — H200 ~55–83% of H20's.
3. **$/trained-token** = GPU-hours × $/GPU-hour — the business decision metric.

## 7. DIRECT kernel measurement (2026-06-06) — closes the open item, refines §3

Section 3 triangulated the bottleneck by *elimination*. We then measured the sparse-MLA
backward kernel **directly** on an idle H200 (single GPU, tilelang JIT, CUDA-event timing +
nsight-compute + a causal store-mode ablation). Tools: `tools/v4_bwd_profile.py`,
`tools/v4_bwd_ablate.py`. Shapes: B=1, H=64, D=512 (kv_lora), topk=640 (window128+compress512).

**(a) The backward is one kernel.** At S=4096: fwd 1.34 ms; bwd = preprocess 0.13 + **bwd-main
12.43** + postprocess 0.01 ms. **bwd-main is 98.9% of the backward and 9.3× the forward.**
Everything below is about bwd-main.

**(b) Causal ablation — swap ONLY the dKV store, keep every GEMM identical** (S=4096, topk=640):

| bwd-main store variant | time | interpretation |
|---|---|---|
| `atomic` (production: fp32 `atomic_addx4` gather-scatter) | 12.89 ms | — |
| `coalesced` (non-atomic contiguous fp32 store, same bytes) | 8.84 ms | **atomic scatter = 4.06 ms = 31.5% of bwd-main** |
| `coalesced16` (non-atomic bf16 store) | 7.29 ms | **de-atomic+bf16 removes 5.60 ms = 43.4%** |
| `nostore` (GEMM/compute floor) | 3.26 ms | pure compute = 25% |

The `local` (window-heavy, max-contention) index distribution gives the **same** atomic premium
(4.16 ms) as uniform `rand` (4.06 ms) → the atomic cost is **structural (RMW + uncoalesced gather +
L2 reduction throughput), NOT lock contention.** De-atomic wins regardless of the index pattern.

**(c) nsight-compute — what resource is actually saturated.** `atomic_addx4` lowers to a global
**reduction** (`lts__t_sectors_op_red`, not `op_atom`):

| metric | S=2048 (dKV 4MB, L2-resident) | S=32768 (dKV 64MB > 60MB L2) |
|---|---|---|
| L2 throughput | **59.6%** (top) | **62.7%** (top) |
| DRAM throughput | 5.1% | **33.4%** |
| Compute (SM) throughput | 15.6% | 15.9% |
| L2 reduction sectors | 251.7 M = 8.05 GB = 20% of all L2 traffic | (scales) |
| top warp stall | long-scoreboard 3.26 inst (global/L2 latency) | — |

**Conclusion (supersedes §3's "44% HBM-bound"):** bwd-main is **memory-hierarchy-bound on the fp32
dKV atomic-reduction scatter — L2-reduction-throughput bound at all sizes (L2 ~60%), spilling to
DRAM (33%) only once dKV exceeds the 60 MB L2 (large S).** It is **never compute-bound** (SM pinned
~16%). The earlier SM-clock throttle showed 56% sensitivity because lowering the SM clock also
throttles the *atomic-reduction issue rate* (reductions are SM-issued) — consistent, not contradictory.
The de-atomic rewrite is confirmed as the single lever, with a **measured ceiling of 31.5% (fp32) –
43.4% (bf16) of bwd-main**, i.e. ~31–43% of the entire backward.

(Memory-clock throttle `-lmc` is unsupported on these H200s — deferred-only — but ncu's direct
`gpu__dram_throughput`/`lts__throughput` counters make it unnecessary: they show L2, not DRAM, is the wall.)

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

**Why config is exhausted:** the atomic-scatter is config-immune; comm is already free (nothing
to tune); bubble is tiny and VPP/DP only make it worse; memory is doubly walled — lower PP OOMs on
weights, reducing recompute OOMs on activations. The remaining levers are all inside the bwd kernel (§9).

## 9. Can the atomic actually be removed? (2026-06-06) — MQA makes it hard

The §7 ablation said removing the atomic recovers 31–43% of bwd-main. We then tried to *realize* it and
found that number is an **idealized floor, not achievable**, because this is **MQA** (one shared KV head
for all H=64 query heads). Traffic counting (per query, S=4096, topk=640, D=512):

| layout | gather traffic / query | dKV write | net traffic/query |
|---|---|---|---|
| **query-centric (current)** | KV gather **0.65 MB** (topk·D·2; no head dim — MQA) | atomic-scatter 5.2 MB (topk·D·NH·8, fp32 RMW) | ~6 MB |
| KV-centric (naive de-atomic) | Q+dO gather **~84 MB** (topk·H·D·2·2 — the ×64 head dim!) | ~1 KB, no atomic | **~84 MB → ~14× WORSE** |

So inverting to a KV-centric loop to kill the atomic **backfires**: it removes a 5 MB atomic write but
adds an **84 MB head-amplified Q/dO gather** (128× the current 0.65 MB KV gather). The atomic is the
*correct* price of the traffic-minimal MQA layout.

**What was tried (all measured on an idle H200, S=4096):**

| lever | result | why |
|---|---|---|
| `block_H=64` → NH=1 (halve dKV op_red count) | 12.61 ms, 1.02× | forces num_stages=1; pipeline loss ≈ cancels atomic saving |
| NH=1 + **direct `acc_dq`→global** (free 64 KB `dQ_shared`) | **12.11 ms, 1.06×** ✅ | keeps NH=1 saving without the smem wall; **dq bit-exact, dkv rel 1.8e-6 vs prod** |
| NH=1 + ns=2 (need split_store=4 to fit smem) | 12.99 ms, 0.99× | the extra shared-staging passes cancel the deeper pipe |
| NH=4 (block_H=16) | 17.5 ms, 0.74× | more head-blocks = more redundant scatters → directly worse (confirms op-count is the wall) |
| bf16 dKV reduction (bf16x2) | not faster | L2-op-count bound, not byte bound; bf16x2 = 2× ops |

**Verdict.** The production query-centric kernel is **near-optimal**. The one free, validated win is
**NH=1 + direct-dq ≈ 6%** of bwd-main (apply by setting `block_H=64` and storing `acc_dq` straight to
`dQ` global, dropping `dQ_shared`; re-add the `dAttnSink` block; re-run the NaN repro suite). Beyond that,
the only paths >10% are a **window-dense + compress-sparse hybrid** (de-atomic only the 128 contiguous
window keys via a sliding-window backward — no inverse index, no H penalty; ~6–10% since window is 20%
of keys) or **query-block grouping** (a block owns G adjacent queries and pre-reduces their shared
window keys in shared memory before scattering; ~10–15%) — both multi-day kernel projects with MQA-limited
upside, **not** the 31–43% the idealized ablation implied. The inverse-index load is nearly uniform
(`tools/v4_invidx_dist.py`: n_j mean 640, p99 694, BQ=64 padding waste 1.05×), so load-balance is *not*
the blocker — the MQA head-amplification is.

## 10. Per-rank sequence length drives MFU (2026-06-08) — the PP3 CP-scan

A previously under-weighted lever surfaced while bringing up the 256K continued-SFT run:
**MFU rises strongly with per-rank sequence length** (= seq / CP), because longer per-rank
sequences make attention more compute-dense, give fatter GEMMs, and better amortize the
sparse-MLA atomic-scatter overhead (§7).

**Two measured anchor points (both PP7/CP6, the production layout):**
| run | per-rank tokens (max_tokens) | recompute | MFU (actor_train_tflops/989) |
|---|---|---|---|
| 128K SFT (kaynzhang_077_134k) | 32,768 | block-4 | **11.6%** (115 tflops) |
| 256K SFT (kaynzhang_077_256k) | 43,691 | full/uniform-1 | **17.6%** (174 tflops) |

Per-rank 32K→44K gave **+52% relative MFU** — and the 256K point did so *despite* heavier
(full/uniform) recompute, so the per-rank-length effect is even stronger than the raw numbers.
This means the parallelism goal is **not "max CP to kill the PP bubble"** but **"keep per-rank
sequence long (high MFU) while managing the bubble"** — high CP shortens per-rank seq and erodes MFU.

**The scan (256K data, PP3, DP1, init from iter_1164, full/uniform-1 recompute):** vary CP to vary
per-rank seq, measure MFU + peak memory.
| scale | CP | per-rank tokens | nodes (DP1) | result |
|---|---|---|---|---|
| tp8pp3cp4ep8  | 4  | 65,604 | 12 | **OOM** — peak 99.5% (reserved 130GB), DeepEP illegal-access at the memory wall. PP3's 14-layer/stage weights + CP4's high activation exceed 140GB. |
| (baseline PP7) | 6 | 43,691 | 42 | fits (64% / 89GB), **MFU 17.6%** |
| tp8pp3cp7ep8  | 7  | 37,488 | 21 | fits but **ragged edge — ~99.5% in-step** (127.8GB "before-clear"/91%, nvidia-smi ~143GB in-step), **MFU 20.0%** (197.7 tflops, step-1 steady) |
| tp8pp3cp14ep8 | 14 | 18,744 | 42 | fits comfortably (**84% / 117GB**), **MFU 15.0%** (~148 tflops, steps 1-5 steady) |
| tp8pp3cp7ep8 (DP2) | 7 | 37,488 | 42 | fits but **ragged ~99.5% in-step** (92% before-clear), **MFU 18.9%** (~187 tflops, ~224s/step) |

**FINAL verdict (the production comparison, all 42-node 256K configs):**
| config | MFU | actor_train/step | memory | scaling 21→42 |
|---|---|---|---|---|
| PP7/CP6 (current baseline) | 17.6% | ~243s | 64% (safe) | — |
| **PP3/CP7 DP2** | **18.9%** | **~224s** | ~99.5% (ragged) | **1.91×** (CP7-DP1 428s → 224s, near-ideal) |
| PP3/CP14 DP1 | 15.0% | ~293s | 84% (safe) | 1.46× (poor — short seq kills MFU) |

- **PP3/CP7-DP2 is the only PP3 config that beats the PP7/CP6 baseline** (18.9% vs 17.6% → ~8% faster
  wall-clock, ~4h on a 52h 2-epoch run). It keeps per-rank seq long (37.5K → high MFU) and scales
  **1.91×** from 21→42 nodes (vs CP14's 1.46×), confirming: **keep per-rank seq long, don't spend GPUs on CP.**
- **The DP2 cost is real but small**: CP7-DP1 20.0% → CP7-DP2 18.9% (−1.1pp, the doubled bubble + 284B
  DP all-reduce), as predicted.
- **PP3/CP14-DP1 is a dead end**: 15.0% < baseline. Using all 42 nodes via high CP (the only DP1 way)
  shortens per-rank seq below the MFU knee.
- **The catch for production**: CP7-DP2 runs at ~99.5% in-step memory — a multi-day run risks OOM on
  dynamic-batch variance. Mitigation: trim `max-tokens-per-gpu` 37488→~36000 (in-step ~95%) to keep most
  of the +8% with margin. **Decision = +8% speed (CP7-DP2, ragged) vs safe (PP7/CP6 baseline).**

**Two effects, both real, and they STACK (CP7 result clarifies §10):**
1. **Per-rank-seq → MFU** (clean within fixed PP7): 32K→11.6%, 44K→17.6%.
2. **PP3 bubble reduction → MFU**: PP3 CP7 hits **20.0% > PP7 CP6's 17.6%** *despite a shorter
   per-rank seq* (37.5K < 43.7K) — i.e. PP3's smaller bubble (~8% vs PP7's ~12-16%) more than pays back
   the per-rank-seq loss. So PP3 is a genuine win when it fits.

**Memory is the binding constraint for PP3** (14-15 layers/stage weights, NOT shardable by CP/DP):
- CP4 (64K/rank): OOM. CP6 (43.7K) under PP3 would also be over. **CP7 (37.5K) is the practical PP3
  floor — already ~99.5% in-step.** CP14 (18.7K) is the comfortable PP3 config.
- So the viable PP3 band at 256K is **CP7–CP14 (per-rank 18.7K–37.5K)**; CP7 is fastest-but-ragged, CP14
  is safe-but-shorter-seq.

**Clean isolation of the per-rank-seq effect at fixed PP3** (the decisive result): CP7 (37.5K/rank) =
**20.0%** vs CP14 (18.7K/rank) = **15.0%** — same PP3, per-rank seq halved → **MFU −24% relative**. The
per-rank-sequence effect is strong enough that it **dominates PP3's bubble win**: CP14, despite PP3's
small bubble, lands *below* even the PP7/CP6 baseline (17.6%), because its 18.7K/rank sequences are too
short to be compute-dense. So "use all 42 nodes with PP3 at DP1" (which forces CP14) is **MFU-negative**.
The production question is therefore **CP7-DP2 (42 nodes, keeps 37.5K/rank, +DP2 bubble/comm, ragged 99.5%
mem) vs CP14-DP1 (42 nodes, 18.7K/rank, safe 84% mem, 15.0%)** — see the CP7-DP2 row. Reproduce: `--workload sft_pp3_scan_smoke --scale tp8pp3cp<N>ep8
--fleet h200_k8s_<12|21|42>node --max-tokens-per-gpu <seq/N>` (seq 262416; one workload, CLI overrides).

## Appendix — reproducibility
- 16-node launch + the 4 infra blockers/fixes: see memory `h200-k8s-16node-runbook`.
- NCCL sweep knob: `V4_NCCL_MAX_NCHANNELS` (wired into run.sh runtime_env).
- wandb runs: H20 `lr0mjb4o`/`wkf2d2lc`; H200 42-node `jvp2ovpl`; NCCL sweep `yyrm0qlz`(8ch)/`simb8dhd`(2ch).
- **bwd-kernel microbench tools** (run on an idle H200 pod, `PYTHONPATH=<fsx miles>`, `CUDA_VISIBLE_DEVICES=0`):
  - `tools/v4_bwd_profile.py [S topk dist]` — per-kernel CUDA-event breakdown (fwd/preprocess/bwd-main/postprocess) + atomic HBM model.
  - `tools/v4_bwd_ablate.py [S topk dist]` — causal store-mode ablation (atomic / coalesced-fp32 / coalesced-bf16 / nostore); also parametrizes block_H / num_stages / direct_dq / split_store.
  - `tools/v4_bwd_blockh.py [S topk]` — block_H/NH/ns/direct_dq/split sweep + correctness check vs the production kernel.
  - `tools/v4_invidx_dist.py [S topk]` — inverse-index load distribution (queries-per-key) for de-atomic feasibility.
  - ncu: `ncu -k regex:sparse_mqa_bwd_kernel -c 1 -s 20 --section SpeedOfLight --section MemoryWorkloadAnalysis ... python3 tools/v4_bwd_profile.py 2048 640 rand`.

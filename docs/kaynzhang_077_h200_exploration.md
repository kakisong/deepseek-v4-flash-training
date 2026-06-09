# kaynzhang_077 H200 Training Exploration

Date: 2026-06-04

## Goals

- Maximize training token throughput, using `actor_train_tok_per_s`, `step_time`, and TFLOPs/MFU as the primary decision metrics.
- Increase H200 GRAM utilization only when it improves training token throughput or lets us reduce recompute without OOM.
- Keep the 42-node / 336-GPU pool fully occupied for production runs.
- Prefer measurements from one-step smoke jobs before changing long-run configs.

## Cluster

- Ray dashboard: `http://10.3.234.60:8201`
- Prometheus: `http://10.3.234.60:40001/promql`
- Grafana: `http://10.3.234.60:7777/grafana`
- GPU pool: 42 H200 nodes, 336 GPUs.
- CPU nodes: Ray head plus one CPU worker. They should not be used by training actors.
- Known infra risk: `10.3.22.244` has `/ray_local` over 95% full and Ray warns object spilling may fail.

## Model And Data

- Model checkpoint: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/models/DeepSeek-V4-Flash-FP8_torch_dist`
- SFT data: `$V4_DATA/albaliang_077_le134k.jsonl`
- Sequence length: `134136`
- Tool key: `tools`

## Metrics To Record

- Ray job id, output directory, W&B run URL.
- Scale: `TP`, `PP`, `CP`, `EP`, `ETP`, `DP`, pipeline layout.
- Memory knobs: CPU offload, recompute mode, `max_tokens_per_gpu`, `global_batch_size`.
- Communication knobs: CP degree, TP locality, PP depth, DP degree, DeepEP, EP overlap. These are diagnostic metrics, not optimization goals unless they move token throughput.
- Runtime metrics: `actor_train_time`, `step_time`, tokens/GPU/s, effective tokens/GPU/s, TFLOPs/MFU.
- Prometheus metrics: max and average `DCGM_FI_DEV_FB_USED`, average `DCGM_FI_DEV_GPU_UTIL`, node network throughput if queried.
- Failure signals: CUDA OOM, Ray pending demands, dead actors, local disk warnings, checkpoint save failures.

## Runs

### R1: no-offload CP6 baseline, succeeded

- Submitted: 2026-06-04 19:31 CST
- Job id: `raysubmit_nM7fY8TYCnciGCTC`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOff-H200-20260604-193125`
- Workload: `sft_kaynzhang_077_134k_smoke_no_offload`
- Scale: `tp8pp7cp6ep8`
- GPUs: 336
- Parallelism: `TP=8`, `PP=7`, `CP=6`, `EP=8`, `ETP=1`, effective `DP=1`
- Layout: `Et*6|t*6|t*6|t*6|t*6|t*6|t*7L`
- CPU offload: disabled
- DeepEP: enabled
- Router dtype: `fp32`
- Attention: `tilelang`
- Recompute: full, uniform, 1 layer
- `max_tokens_per_gpu`: `32768`
- `global_batch_size`: `128`
- `rollout_batch_size`: `128`
- Debug mode: train-only one-step smoke
- Dump details: disabled
- Checkpoint optimizer save: disabled via `--no-save-optim`
- W&B: enabled, run URL `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/dp3jmf0n`

Observed so far:

- Ray state: `RUNNING`, no pending demands, no recent Ray failures.
- Checkpoint load completed at 19:36:34-19:36:36 CST.
- Model and optimizer initialized at about 34.1GB GPU memory per sampled rank.
- Rollout collected 128 samples at 19:37:23 CST.
- Rollout metrics:
  - response length mean `4719.26`
  - response length max `29050`
  - rollout time `43.39s`
  - rollout tokens/GPU/s `185.62`
  - rollout effective tokens/GPU/s `41.43`
- Training step started at 19:37:25 CST.
- Training step ended at 19:49:49 CST.
- Train metrics, logged at 19:49:59 CST:
  - `perf/update_weights_time`: `0.000033s`
  - `perf/data_preprocess_time`: `0.846s`
  - `perf/train_wait_time`: `45.917s`
  - `perf/actor_train_time`: `752.889s`
  - `perf/train_time`: `754.174s`
  - `perf/actor_train_tflops`: `19.411`
  - `perf/actor_train_tok_per_s`: `9493.726`
  - `perf/step_time`: `800.091s`
  - `perf/wait_time_ratio`: `0.0574`
- Checkpoint save started at 19:49:59 CST; first save completed at 19:50:24 CST, then a second terminal save started.
- Second checkpoint save completed at 19:50:47 CST.
- Ray job status: succeeded at 19:51:24 CST.
- Shutdown warning: W&B teardown raised `ConnectionResetError: Connection lost` in `teardown_atexit`; Ray job still succeeded and metrics were emitted.
- Smoke checkpoint size: `531G`; removed after collecting metrics.
- Prometheus during training, 19:46 CST:
  - max `DCGM_FI_DEV_FB_USED`: `63962 MiB`
  - avg `DCGM_FI_DEV_FB_USED`: `59624 MiB`
  - avg `DCGM_FI_DEV_GPU_UTIL`: about `99.4%`
- Prometheus rolling window, 19:48 CST:
  - 10-minute max `DCGM_FI_DEV_FB_USED`: `63976 MiB`
  - 10-minute avg `DCGM_FI_DEV_FB_USED`: `58748 MiB`
  - 5-minute avg `DCGM_FI_DEV_GPU_UTIL`: `99.63%`
- Ray network speed, 19:50 CST:
  - avg node send: `3.97e9 B/s`
  - avg node receive: `3.95e9 B/s`
  - max node send: `9.64e9 B/s`
  - max node receive: `9.65e9 B/s`
  - hottest nodes included `10.3.28.82`, `10.3.77.237`, `10.3.75.76`, `10.3.94.204`, `10.3.22.244`

Interim interpretation:

- Disabling CPU offload fits comfortably with full recompute at `32768` tokens/GPU.
- Current GRAM utilization is still low for H200, roughly 60GB average and 64GB max during the observed training window.
- The job is compute-active rather than Ray-pending: GPU util is near 99% across the pool.
- Full recompute likely makes the step expensive; next probes should reduce recompute and/or raise local tokens.
- The measured `19.4 TFLOPs/GPU` is far below the H200 BF16 peak, so the next probes should focus on reducing recompute cost and improving training token throughput, not just raw GPU occupancy.

## Candidate Next Probes

### P1: CP6, higher GRAM, less recompute

- Workload: `sft_kaynzhang_077_134k_smoke_no_offload_mem`
- Scale: `tp8pp7cp6ep8`
- Submitted: 2026-06-04 19:52 CST
- Job id: `raysubmit_nfVtsSDzmTcdCgLH`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffMem-H200-20260604-195257`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/n8sve6pk`
- Status: failed
- Goal: increase GRAM utilization and reduce full recompute overhead while keeping the same communication topology.
- Changes from R1:
  - `max_tokens_per_gpu=49152`
  - `recompute_granularity=selective`
  - CPU offload still disabled
- Launch verification:
  - `log_probs_max_tokens_per_gpu=49152`
  - `max_tokens_per_gpu=49152`
  - `recompute_granularity=selective`
  - `recompute_method=None`
  - no optimizer CPU offload flags in entrypoint
- Failure:
  - Failed at actor training start, shortly after rollout and data preprocess.
  - Error: `RuntimeError: shape '[1, 49152, 1, 32]' is invalid for input of size 524288`
  - Stack points to `miles_plugins/models/deepseek_v4/ops/rope.py:64`, `freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))`.
  - This is not OOM; it looks like a RoPE/dynamic-batch shape assumption triggered by `max_tokens_per_gpu=49152`.
  - Output dir stayed small, about `3.5M`; no checkpoint cleanup needed.
- Compare against R1:
  - peak and average GRAM
  - actor train time
  - tokens/GPU/s and TFLOPs
  - any OOM or checkpoint behavior

### P1b: CP6, selective recompute at known-good token cap, failed with OOM

- Workload: `sft_kaynzhang_077_134k_smoke_no_offload_selective`
- Scale: `tp8pp7cp6ep8`
- Submitted: 2026-06-04 20:01 CST
- Job id: `raysubmit_v8gihpHAHy7H2gAA`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffSel-H200-20260604-200141`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/i1gp1dnn`
- Status: failed at about 2026-06-04 20:09 CST
- Goal: isolate selective recompute at the R1 known-good token cap.
- Changes from R1:
  - `recompute_granularity=selective`
  - `max_tokens_per_gpu=32768`
  - CPU offload still disabled
- Reason: P1's `49152` token cap failed in RoPE shape handling before it could measure memory or throughput.
- Launch verification:
  - `log_probs_max_tokens_per_gpu=32768`
  - `max_tokens_per_gpu=32768`
  - `recompute_granularity=selective`
  - `recompute_method=None`
- Observed:
  - Checkpoint load completed at about 20:06 CST.
  - Model and optimizer initialized at about 34.1GB GPU memory per sampled rank, same as R1.
  - Rollout collected 128 samples at 20:07:31 CST.
  - Rollout metrics:
    - response length mean `4719.26`
    - response length max `29050`
    - rollout time `43.36s`
    - rollout tokens/GPU/s `185.75`
    - rollout effective tokens/GPU/s `41.46`
  - `train_wait` elapsed `46.7s`.
  - `data_preprocess` elapsed `0.8s`.
  - Actor training started at 20:07:32 CST and failed after `49.4s`.
- Failure:
  - Error: `torch.OutOfMemoryError: CUDA out of memory`
  - Allocation attempted: `6.00 GiB`
  - Failing GPU state from log: total `139.80 GiB`, free `3.49 GiB`, process memory in use `136.28 GiB`, PyTorch allocated `131.49 GiB`, PyTorch reserved but unallocated `401.73 MiB`.
  - Ray also repeated the existing `/ray_local` over 95% full warning on `10.3.22.244`; that is an infra risk, but the direct failure was CUDA OOM.
- Prometheus around failure, 10-minute rolling query at about 20:09 CST:
  - max `DCGM_FI_DEV_FB_USED`: `139175 MiB`
  - avg per-GPU rolling max `DCGM_FI_DEV_FB_USED`: `67006 MiB`
  - max `DCGM_FI_DEV_GPU_UTIL`: `100%`
  - avg per-GPU rolling max `DCGM_FI_DEV_GPU_UTIL`: `69.6%`
- Interpretation:
  - P1b proves the RoPE shape failure in P1 was caused by the `49152` token cap, not by selective recompute alone.
  - At `32768` tokens/GPU, selective recompute can enter training, but activation memory eventually reaches the H200 limit and OOMs.
  - The stable no-offload region is between R1 full recompute's roughly 64GB peak and P1b selective's roughly 139GB peak; the next probe should keep no CPU offload but use a less aggressive recompute reduction or reduce per-rank token pressure.

### RoPE shape patch for packed THD + zigzag CP

- Applied at about 2026-06-04 20:18-20:21 CST.
- Touched files:
  - `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles/miles_plugins/models/deepseek_v4/ops/cp_utils.py`
  - `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles/miles_plugins/models/deepseek_v4/deepseek_v4.py`
  - `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles/miles_plugins/models/deepseek_v4/ops/v4_indexer.py`
  - `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/train/miles/miles_plugins/models/deepseek_v4/ops/compressor.py`
- Root cause:
  - Miles THD CP slicing uses two zigzag chunks per sample on each CP rank.
  - The V4 attention path assumed contiguous CP and used `cp_rank * local_seqlen` to slice RoPE frequencies.
  - With `local_seqlen=49152`, CP rank 5 sliced `[245760:294912]` from a RoPE table of length `262144`, leaving only `16384` rows. That produced the observed `shape '[1, 49152, 1, 32]' is invalid for input of size 524288`.
- Patch behavior:
  - Build explicit per-token positions from `packed_seq_params.cu_seqlens_q`.
  - Reset RoPE positions at packed sample boundaries.
  - Map each sample's two local zigzag chunks to their true global positions inside that sample.
  - Index RoPE frequencies by explicit positions in attention, V4 indexer, and compressor.
  - Add explicit runtime errors if `cu_seqlens` cannot be mapped back to local CP chunks or if any position still exceeds the RoPE table.
- Validation:
  - Syntax checked by `compile()` locally and `compileall` in a Ray worker container.
  - Container shape test passed for the failed shape:
    - `cu_seqlens=[0, 98304, 196608, 294912]`
    - `cp_size=6`, fake `cp_rank=5`, `local_seqlen=49152`
    - generated positions length `49152`, min `40960`, max `57343`
    - indexed RoPE shape `(49152, 32)`
    - `apply_rotary_emb` accepted `x.shape=(1, 49152, 1, 64)` and returned the same shape.
- Remaining semantic caveat:
  - This patch fixes the RoPE shape and per-token RoPE position source.
  - V4 sparse-attention topk/KV ordering still uses the existing CP helper assumptions; the next correctness pass should audit packed zigzag KV gather/topk indexing, especially before trusting new topology changes.

### P1r: P1 after RoPE patch, shape fixed, failed with OOM

- Workload: `sft_kaynzhang_077_134k_smoke_no_offload_mem`
- Scale: `tp8pp7cp6ep8`
- Submitted: 2026-06-04 20:21 CST
- Job id: `raysubmit_4hWMANkpHG14ZcRA`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffMem-H200-20260604-202124`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/86tkdwa8`
- Status: failed at about 2026-06-04 20:28 CST
- Same config as P1:
  - `max_tokens_per_gpu=49152`
  - `recompute_granularity=selective`
  - CPU offload disabled
  - `--no-save-optim`
- Launch/runtime verification:
  - `optimizer_cpu_offload=False`
  - `use_precision_aware_optimizer=False`
  - `recompute_granularity=selective`
  - `max_tokens_per_gpu=49152`
- Observed:
  - All 336 `MegatronTrainRayActor` actors reached `ALIVE`; no Ray pending demands.
  - Checkpoint load completed at about 20:26:35 CST.
  - Model and optimizer initialized at mostly about 34.1GB GPU memory per sampled rank, with one sampled rank at 40.0GB.
  - Rollout collected 128 samples at 20:27:23 CST.
  - Rollout metrics:
    - response length mean `4719.26`
    - response length max `29050`
    - rollout time `42.93s`
    - rollout tokens/GPU/s `187.62`
    - rollout effective tokens/GPU/s `41.88`
  - `train_wait` elapsed `44.6s`.
  - `data_preprocess` elapsed `0.8s`.
  - Actor training started at 20:27:25 CST.
  - No RoPE shape error after entering actor training; the old P1 failure point was passed.
- Failure:
  - Actor training failed after `47.4s`.
  - Error: `torch.OutOfMemoryError: CUDA out of memory`
  - Allocation attempted: `12.94 GiB`
  - Failing GPU state from log: total `139.80 GiB`, free `12.34 GiB`, process memory in use `127.42 GiB`, PyTorch allocated `123.09 GiB`, PyTorch reserved but unallocated `201.55 MiB`.
  - Output dir stayed small, about `3.5M`; no checkpoint cleanup needed.
- Prometheus:
  - Instantaneous query near failure: max `DCGM_FI_DEV_FB_USED` about `124788 MiB`, avg about `46761 MiB`.
  - 8-minute rolling query after failure: max `DCGM_FI_DEV_FB_USED` about `141025 MiB`, avg per-GPU rolling max about `56139 MiB`.
  - 8-minute rolling max GPU util `100%`, avg per-GPU rolling max GPU util about `67.65%`.
- Interpretation:
  - The RoPE shape bug is fixed for this failed configuration.
  - `49152 + selective` is not viable on H200 at CP6 because it reaches the memory ceiling.
  - The usable no-offload search space is below P1r/P1b selective memory, likely via full recompute with partial/block grouping or by reducing per-rank token pressure.

### P2: CP3 / DP2 topology probe, stopped as throughput-negative

- New scale file: `cluster/scale/tp8pp7cp3ep8.env`
- Workload: `sft_kaynzhang_077_134k_smoke_no_offload_cp3_full`
- Submitted: 2026-06-04 20:33 CST
- Job id: `raysubmit_C2JajppPqERxMNJL`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-cp3Full-H200-20260604-203350`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/uvrfqdap`
- Status: stopped manually at 2026-06-04 20:58 CST because the actor training step had already exceeded the CP6 baseline time without completing.
- Goal: test whether CP3/DP2 is a viable throughput topology while still consuming all 336 GPUs.
- Parallelism: `TP=8`, `PP=7`, `CP=3`, `EP=8`, `ETP=1`, effective `DP=2`
- Recompute: full, uniform, 1 layer
- `max_tokens_per_gpu`: `49152`
- CPU offload: disabled
- Observed:
  - All 336 `MegatronTrainRayActor` actors reached `ALIVE`; no Ray pending demands.
  - Checkpoint load completed at about 20:38 CST.
  - Model and optimizer initialized at roughly `34-40GB` GPU memory per sampled rank.
  - Rollout collected 128 samples at 20:39:55 CST.
  - Rollout metrics:
    - response length mean `4719.26`
    - response length max `29050`
    - rollout time `44.97s`
    - rollout tokens/GPU/s `179.12`
    - rollout effective tokens/GPU/s `39.98`
  - `train_wait` elapsed `48.1s`.
  - `data_preprocess` elapsed `0.4s`.
  - Actor training started at 20:39:56 CST and was still running at 20:57:57 CST.
  - At stop time, actor training had run for about `1081s`, already worse than R1's `752.889s`, so the best possible token throughput was already below baseline.
  - Repeated `miles/backends/training_utils/loss.py:927` tensor-copy warnings appeared across ranks during training; this is log noise and potential Python overhead, not the direct stop reason.
- Prometheus:
  - Peak observed max `DCGM_FI_DEV_FB_USED`: about `66124 MiB`
  - Average `DCGM_FI_DEV_FB_USED`: about `60.4GB`
  - Average `DCGM_FI_DEV_GPU_UTIL`: ranged from about `75.8%` to `93.7%` during the later training window.
  - Ray node network speed during the run: avg send/receive about `1.1e9 B/s`, max send/receive about `4.1e9 B/s`.
- Interpretation:
  - CP3/DP2 was stable and did not hit the RoPE shape failure or CUDA OOM in the observed window.
  - It is not a good throughput direction for this workload: at identical batch size it had already exceeded the CP6 baseline actor-train time before completion.
  - Lower communication is not useful by itself here; the next probes should stay on CP6/DP1 and reduce recompute overhead while keeping memory below the OOM boundary.

### P3: CP6 block recompute throughput probe

- Workload: `sft_kaynzhang_077_134k_smoke_no_offload_block4`
- Scale: `tp8pp7cp6ep8`
- Submitted: 2026-06-04 21:01 CST
- Job id: `raysubmit_xtVcwemdkHk8mamq`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffBlock4-H200-20260604-210121`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/kf7sarwk`
- Status: succeeded at 2026-06-04 21:19 CST
- Goal: improve `actor_train_tok_per_s` over R1 by reducing recompute overhead while staying below the selective-recompute OOM boundary.
- Changes from R1:
  - `recompute_granularity=full`
  - `recompute_method=block`
  - `recompute_num_layers=4`
  - `max_tokens_per_gpu=32768`
  - CPU offload still disabled
- Success criterion:
  - Must exceed R1's `actor_train_tok_per_s=9493.726` and/or reduce R1's `actor_train_time=752.889s` without OOM.
- Observed:
  - Checkpoint load completed at about 21:06:32 CST.
  - Model and optimizer initialized at about `34-40GB` GPU memory per sampled rank, same as R1.
  - Rollout collected 128 samples at 21:07:20 CST.
  - Rollout metrics:
    - response length mean `4719.26`
    - response length max `29050`
    - rollout time `43.18s`
    - rollout tokens/GPU/s `186.51`
    - rollout effective tokens/GPU/s `41.63`
  - `train_wait` elapsed `46.0s`.
  - `data_preprocess` elapsed `0.8s`.
  - Actor training ran from 21:07:22 to 21:17:38 CST.
  - Train metrics:
    - `perf/update_weights_time`: `0.000026s`
    - `perf/data_preprocess_time`: `0.848s`
    - `perf/train_wait_time`: `45.638s`
    - `perf/actor_train_time`: `626.223s`
    - `perf/train_time`: `626.227s`
    - `perf/actor_train_tflops`: `23.337`
    - `perf/actor_train_tok_per_s`: `11414.023`
    - `perf/step_time`: `671.864s`
    - `perf/wait_time_ratio`: `0.0679`
  - Checkpoint save completed twice despite `--no-save-optim`; smoke checkpoint size was `531G` and was removed.
  - W&B teardown raised `ConnectionResetError: Connection lost` during atexit, same as R1; Ray job still succeeded.
- Prometheus:
  - Max observed `DCGM_FI_DEV_FB_USED`: `116852 MiB`
  - Average per-GPU rolling max `DCGM_FI_DEV_FB_USED`: `88562 MiB`
  - Max rolling GPU util: `100%`
  - Average rolling GPU util: `100%`
- Comparison vs R1:
  - `actor_train_tok_per_s`: `11414.023` vs `9493.726`, `+20.2%`
  - `actor_train_time`: `626.223s` vs `752.889s`, `-16.8%`
  - `step_time`: `671.864s` vs `800.091s`, `-16.0%`
  - `actor_train_tflops`: `23.337` vs `19.411`, `+20.2%`
  - Peak GPU memory rose from about `64GB` to about `117GB`, still below the P1b/P1r OOM boundary around `139-141GB`.
- Interpretation:
  - `full/block/4` is the first better-than-baseline config for the current objective.
  - The goal should remain token throughput; this result is useful because higher memory utilization reduced recompute overhead and improved tokens/s.
  - There is still some headroom below the observed selective-recompute OOM boundary, so the next throughput probe should try `full/block/3` at the same CP6/DP1 topology.

### P4: CP6 block3 recompute throughput probe

- New workload file: `cluster/workload/sft_kaynzhang_077_134k_smoke_no_offload_block3.env`
- Workload: `sft_kaynzhang_077_134k_smoke_no_offload_block3`
- Scale: `tp8pp7cp6ep8`
- Submitted: 2026-06-04 21:23 CST
- Job id: `raysubmit_5ksA9BaWTSyBayqp`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffBlock3-H200-20260604-212259`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/0z9ndwhf`
- Status: succeeded at 2026-06-04 21:41 CST
- Goal: reduce recompute one notch further than P3 and test whether token throughput improves.
- Changes from P3:
  - `recompute_num_layers=3`
  - `max_tokens_per_gpu=32768`
  - CP6/DP1 topology unchanged
  - CPU offload still disabled
- Observed:
  - Checkpoint load completed at about 21:28:16 CST.
  - Model and optimizer initialized at about `34GB` GPU memory per sampled rank.
  - Rollout collected 128 samples at 21:29:06 CST.
  - Rollout metrics:
    - response length mean `4719.26`
    - response length max `29050`
    - rollout time `43.31s`
    - rollout tokens/GPU/s `185.98`
    - rollout effective tokens/GPU/s `41.51`
  - `train_wait` elapsed `46.9s`.
  - `data_preprocess` elapsed `0.8s`.
  - Actor training ran from 21:29:07 to 21:39:33 CST.
  - Train metrics:
    - `perf/update_weights_time`: `0.000031s`
    - `perf/data_preprocess_time`: `0.839s`
    - `perf/train_wait_time`: `46.259s`
    - `perf/actor_train_time`: `635.051s`
    - `perf/train_time`: `635.080s`
    - `perf/actor_train_tflops`: `23.012`
    - `perf/actor_train_tok_per_s`: `11255.360`
    - `perf/step_time`: `681.339s`
    - `perf/wait_time_ratio`: `0.0679`
  - Checkpoint save completed twice despite `--no-save-optim`; smoke checkpoint size was `531G` and was removed.
  - W&B teardown raised `ConnectionResetError: Connection lost` during atexit; Ray job still succeeded.
- Prometheus:
  - Max observed `DCGM_FI_DEV_FB_USED`: about `140330 MiB`
  - Average per-GPU rolling max `DCGM_FI_DEV_FB_USED`: about `102170 MiB`
  - Average GPU util during training windows: about `99.4-99.6%`
- Comparison:
  - Versus R1, P4 is still much faster: `11255.360` vs `9493.726` actor train tok/s, `+18.6%`.
  - Versus P3, P4 is slightly slower: `11255.360` vs `11414.023` actor train tok/s, `-1.4%`.
  - P4 uses far more memory than P3: peak about `140GB` vs `117GB`, leaving little to no OOM safety margin.
- Interpretation:
  - More memory utilization stopped helping after block4; block3 is on the wrong side of the throughput/memory tradeoff.
  - Current best one-step config is P3: CP6/DP1, no CPU offload, `full/block/4`, `max_tokens_per_gpu=32768`.
  - Next throughput probe should keep block4 and test whether raising `max_tokens_per_gpu` moderately improves dynamic-batch packing without crossing the OOM boundary.

### P5: CP6 block4 with 36864 token cap

- New workload file: `cluster/workload/sft_kaynzhang_077_134k_smoke_no_offload_block4_tok36k.env`
- Workload: `sft_kaynzhang_077_134k_smoke_no_offload_block4_tok36k`
- Scale: `tp8pp7cp6ep8`
- Submitted: 2026-06-04 21:43 CST
- Job id: `raysubmit_LK7ELWtBKR3pEZ8F`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffBlock4Tok36k-H200-20260604-214329`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/l873om32`
- Status: succeeded at 2026-06-04 22:10 CST
- Goal: keep P3's best recompute setting and raise dynamic batch token cap to see if packing/token throughput improves.
- Changes from P3:
  - `max_tokens_per_gpu=36864`
  - `recompute_granularity=full`
  - `recompute_method=block`
  - `recompute_num_layers=4`
  - CP6/DP1 topology unchanged
  - CPU offload still disabled
- Observed:
  - Checkpoint load completed at about 21:48:39 CST.
  - Rollout collected 128 samples at 21:49:28 CST.
  - Rollout metrics:
    - response length mean `4719.26`
    - response length max `29050`
    - rollout time `43.36s`
    - rollout tokens/GPU/s `185.75`
    - rollout effective tokens/GPU/s `41.46`
  - `train_wait` elapsed `46.2s`.
  - `data_preprocess` elapsed `0.8s`.
  - Actor training ran from 21:49:29 to 22:08:36 CST.
  - Train metrics:
    - `perf/update_weights_time`: `0.000027s`
    - `perf/data_preprocess_time`: `0.843s`
    - `perf/train_wait_time`: `45.565s`
    - `perf/actor_train_time`: `1156.597s`
    - `perf/train_time`: `1156.616s`
    - `perf/actor_train_tflops`: `12.635`
    - `perf/actor_train_tok_per_s`: `6179.957`
    - `perf/step_time`: `1202.182s`
    - `perf/wait_time_ratio`: `0.0379`
  - Checkpoint save completed twice despite `--no-save-optim`; smoke checkpoint size was `531G` and was removed.
  - W&B teardown raised `ConnectionResetError: Connection lost` during atexit; Ray job still succeeded.
- Prometheus:
  - Max observed `DCGM_FI_DEV_FB_USED`: about `127212 MiB`
  - Average per-GPU rolling max `DCGM_FI_DEV_FB_USED`: about `95847 MiB`
- Comparison:
  - Versus P3, P5 is much slower: `6179.957` vs `11414.023` actor train tok/s, `-45.9%`.
  - P5 uses slightly more memory than P3 but far less than P4's near-OOM peak; the slowdown is not due to a CUDA OOM boundary.
- Interpretation:
  - Raising `max_tokens_per_gpu` to `36864` is a clear negative result for this workload. It likely changes dynamic batching/packing or scheduling in a way that reduces training throughput.
  - Keep `max_tokens_per_gpu=32768` for the current best config.
  - Current best remains P3: CP6/DP1, no CPU offload, `full/block/4`, `max_tokens_per_gpu=32768`.

### P6: CP6 block4 with VPP=2

- New scale file: `cluster/scale/tp8pp7cp6ep8_vpp2.env`
- Workload: `sft_kaynzhang_077_134k_smoke_no_offload_block4`
- Scale: `tp8pp7cp6ep8_vpp2`
- Submitted: 2026-06-04 22:25 CST
- Job id: `raysubmit_TU5w17JRP4fraGsP`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-smokeNoOffBlock4-H200-20260604-222505`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/wscc6xeb`
- Status: succeeded at 2026-06-04 22:50 CST
- Goal: test whether interleaved pipeline scheduling improves token throughput over P3 without changing TP/PP/CP/EP, batch size, token cap, or CPU offload.
- Changes from P3:
  - `virtual_pipeline_model_parallel_size=2`, derived from the 14-stage layout.
  - `pipeline_model_parallel_layout=Et*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*3|t*4L`
  - Physical layer ownership remains equivalent to P3: `6/6/6/6/6/6/7` layers across the 7 physical PP ranks.
  - `recompute_granularity=full`, `recompute_method=block`, `recompute_num_layers=4` unchanged.
  - `max_tokens_per_gpu=32768` unchanged.
- Observed:
  - VPP was enabled in Megatron logs:
    - `Number of virtual stages per pipeline stage: 2`
    - `virtual_pipeline_model_parallel_size=2`
  - Checkpoint load completed at about 22:30:27 CST.
  - Rollout collected 128 samples at 22:31:22 CST.
  - Rollout metrics:
    - response length mean `4719.26`
    - response length max `29050`
    - rollout time `48.10s`
    - rollout tokens/GPU/s `167.46`
    - rollout effective tokens/GPU/s `37.38`
  - `train_wait` elapsed `52.4s`.
  - `data_preprocess` elapsed `1.0s`.
  - Actor training ran from 22:31:24 to 22:49:41 CST.
  - Train metrics:
    - `perf/update_weights_time`: `0.000037s`
    - `perf/data_preprocess_time`: `0.842s`
    - `perf/train_wait_time`: `51.835s`
    - `perf/actor_train_time`: `1106.421s`
    - `perf/train_time`: `1106.472s`
    - `perf/actor_train_tflops`: `13.208`
    - `perf/actor_train_tok_per_s`: `6460.217`
    - `perf/step_time`: `1158.307s`
    - `perf/wait_time_ratio`: `0.0448`
  - Checkpoint save completed twice despite `--no-save-optim`; smoke checkpoint size was `531G` and was removed from inside the Ray head container because files were owned by `root`.
- Prometheus:
  - Max observed `DCGM_FI_DEV_FB_USED`: about `71458 MiB`
  - Average per-GPU rolling max `DCGM_FI_DEV_FB_USED`: about `66325 MiB`
  - Average GPU util during the main training window reached about `99.6%`.
- Comparison:
  - Versus P3, P6 is much slower: `6460.217` vs `11414.023` actor train tok/s, `-43.4%`.
  - Versus P3, actor train time increased from `626.223s` to `1106.421s`, `+76.7%`.
  - Versus P3, step time increased from `671.864s` to `1158.307s`, `+72.4%`.
  - VPP did reduce memory: peak max GPU memory fell from about `116852 MiB` to `71458 MiB`, but this is not useful for the current token-throughput objective.
- Interpretation:
  - VPP=2 is a clear negative result for the current P3-derived config.
  - The most likely reason is recompute semantics, not communication saturation. In P3, `recompute_num_layers=4` applies within 6/7-layer physical PP stages. In P6, VPP splits each physical stage into 3/4-layer virtual chunks while keeping `recompute_num_layers=4`, so each virtual chunk is close to fully recomputed. That explains both observations: much lower memory and much lower throughput.
  - VPP also doubles the number of pipeline chunks, adding extra pipeline P2P scheduling and kernel-launch overhead. With `micro_batch_size=1`, `global_batch_size=128`, and DP=1, there are enough microbatches to keep the pipeline busy, but the additional overhead does not compensate for the extra recompute work.
  - The layer layout is physically balanced, but virtual stage boundaries are not identical to the P3 recompute boundary. Stage 0 includes embedding and stage 13 includes loss, so the interleaved chunks are not perfectly symmetric for this V4/MoE workload.
  - Current best remains P3: CP6/DP1, VPP off, no CPU offload, `full/block/4`, `max_tokens_per_gpu=32768`.

### Materialized default for current scale

- Date: 2026-06-04 CST
- Current default launch is now:
  - `fleet=h200_k8s_42node`
  - `scale=tp8pp7cp6ep8`
  - `workload=sft_kaynzhang_077_134k_3epoch`
- `run.sh` defaults to this combination when no `--fleet`, `--scale`, or `--workload` is provided.
- The formal 3epoch workload now uses the P3-derived compute settings:
  - CPU offload disabled
  - DeepEP enabled
  - router dtype `fp32`
  - attention implementation `tilelang`
  - `HW_SEQ_LENGTH=134136`
  - `HW_MAX_TOKENS_PER_GPU=32768`
  - `HW_RECOMPUTE_GRANULARITY=full`
  - `HW_RECOMPUTE_METHOD=block`
  - `HW_RECOMPUTE_NUM_LAYERS=4`
- Keep P3 as the default until a later run beats `actor_train_tok_per_s=11414.023` without reducing stability margin.

### Formal 3epoch run status

- Submitted: 2026-06-04 23:08 CST
- Job id: `raysubmit_egpwCDb8TiFuPqYk`
- Output dir: `/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/outputs/stageKaynzhang077-134K-3ep-H200-20260604-230840`
- W&B run URL: `https://wandb.ai/kaynzhang-none/v4-flash-post/runs/3xvn1ho2`
- Config:
  - `fleet=h200_k8s_42node`
  - `scale=tp8pp7cp6ep8`
  - `workload=sft_kaynzhang_077_134k_3epoch`
  - `TP=8 PP=7 CP=6 EP=8 ETP=1`, VPP off
  - no CPU offload, DeepEP enabled, router fp32
  - `full/block/4`, `max_tokens_per_gpu=32768`, `seq_length=134136`
- First rollout:
  - collected 128 samples
  - rollout time `43.494s`
  - rollout tokens/GPU/s `185.177`
  - effective tokens/GPU/s `41.335`
- First training step:
  - `perf/train_wait_time`: `46.853s`
  - `perf/actor_train_time`: `629.796s`
  - `perf/train_time`: `629.800s`
  - `perf/actor_train_tflops`: `23.204`
  - `perf/actor_train_tok_per_s`: `11349.262`
  - `perf/step_time`: `676.653s`
  - `perf/wait_time_ratio`: `0.0692`
- Prometheus during first step:
  - 15-minute max `DCGM_FI_DEV_FB_USED`: about `116994 MiB`
  - average GPU util reached about `99.6%` during the main compute window
- Interpretation:
  - The formal run is healthy after the first step and matches the P3 smoke result closely.
  - First-step token throughput is `-0.6%` vs P3 smoke (`11349.262` vs `11414.023`), within expected run-to-run noise.

## ROOT CAUSE FOUND (2026-06-05): NCCL on TCP, not EFA

The ~2% MFU / 19-23 TFLOPS plateau across R1-P6 was NOT a recompute/topology
problem. It was the network: the miles image
(`radixark/miles:sft-only-v4deps-20260603`) ships **no aws-ofi-nccl plugin**, so
NCCL silently fell back to **TCP sockets over ENA** for ALL inter-node collectives
(CP ring-attention, EP all-to-all, PP P2P). The 16x EFA NICs (~400 GB/s) sat idle.

Evidence on live workers:
- GPU `100% util` but only `~120W` (H200 TDP 700W) = SMs spinning on comm, not computing.
- Training process mapped `libnccl.so.2` only; **no `libnccl-net.so` / `libfabric.so`**.
- ENA `enp7xs0` carried ~11 GB/s; EFA `rdma_*_bytes` counters = 0.

2-node nccl bandwidth (16 ranks), TCP vs EFA:
| collective (512MB) | TCP busBW | EFA busBW | speedup |
|---|---|---|---|
| all_reduce | 5.6 GB/s | 410 GB/s | 73x |
| all_to_all | 1.3 GB/s | 82.8 GB/s | 64x |

### Fix (no image rebuild)
Staged AWS efa-installer's prebuilt `libfabric 2.4` + `aws-ofi-nccl 1.19` + matching
rdma-core (libefa EFA_1.4) + libhwloc15 onto shared fsx at
`/mnt/fsx-cdsn/kaynzhang/deepseek-v4-flash/efa/root`, validated with `fi_info -p efa`.
Wired into `run.sh` via the fleet env (`V4_EFA_ENABLE=1`, `V4_EFA_ROOT`), injecting
into the Ray runtime_env: `LD_LIBRARY_PATH` (efa libs first), `NCCL_NET_PLUGIN`,
`FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`, `FI_EFA_FORK_SAFE=1`,
`NCCL_PROTO=simple`, `NCCL_SOCKET_IFNAME=enp` (hostNetwork pods have per-node NIC
names enp74s0/enp75s0; prefix-match the routable host NIC for OOB bootstrap).

### Result (smoke, full/block/4, max_tokens_per_gpu=32768, CP6/PP7/EP8)
| metric | TCP (P3/formal) | EFA | gain |
|---|---|---|---|
| actor_train_tflops | 23.3 | **71.0** | 3.05x |
| MFU (vs 989) | 2.4% | **7.2%** | 3x |
| actor_train_time | 626s | **206s** | 3.04x |
| tok/s | 11414 | **34724** | 3.04x |

Per-GPU H200 now exceeds the H20 baseline (~53 TFLOPS). Formal 3-epoch run relaunched
with EFA on 2026-06-05.

### Remaining MFU headroom (now that network is not the bottleneck)
EFA lifted MFU 2.4%->7.2%; the new ceiling is compute/wait, not comm. Next levers:
1. `train_wait_time` jumped to ~97s (wait_ratio 0.32) — investigate rollout/data-load
   serialization and the `/ray_local` 95%-full straggler node `10.3.22.244`.
2. Per-GPU power peaks ~360W (not ~700W): tilelang sparse-MLA attention + full/block/4
   recompute is memory-bound. Revisit recompute reduction (now that EFA frees the comm
   budget the earlier OOM tradeoffs may shift) and the long-context straggler (step-13).
3. Re-test topology (CP6 vs lower CP + DP) — earlier "CP3 worse" verdicts were measured
   under the TCP bottleneck and likely flip now.
4. FP8 MoE GEMMs (needs kernel work, per hw/h200.env).

### Durability note
EFA libs live on fsx + are injected via env; if the image is rebuilt or the fsx path
changes this breaks. Long-term: bake aws-ofi-nccl into the training image.

## MFU optimization campaign (post-EFA, 2026-06-05)

Baseline after EFA: 71 TFLOPS, 7.2% MFU, actor_train_time 206s, wait_ratio 0.32.
Goal: MFU >= 40%. All probes are 1-step smokes (full ckpt load ~11min each).

| # | config | TFLOPS | MFU | train_time | note |
|---|--------|-------:|----:|-----------:|------|
| E0 | PP7CP6 full/block4 (EFA baseline) | 71.0 | 7.2% | 206s | reference |
| P1 | E0 + FP8 (TE blockwise MoE GEMM) | 71.2 | 7.2% | 205s | **no gain** — at 134K the MoE GEMM is NOT the bottleneck; attention+recompute dominate. loss 9.84 healthy. |

Finding from P1: FP8-on-MoE-GEMM does nothing here because expert GEMMs aren't on
the critical path at 134K context (max_seq_len_mean ~86K). The levers are recompute
reduction and per-rank attention work (CP), and ultimately the bf16 tilelang
sparse-MLA attention kernel itself. Pivoting to higher-CP + lighter-recompute probes.

### Campaign continued — memory ceiling & recompute findings (2026-06-05)

| # | config | result |
|---|--------|--------|
| P2 | CP7(PP6) + selective(moe_act,layernorm,mla_up_proj), tok20480 | **OOM 138GB** — selective too memory-heavy |
| P3 | CP6 + full/block/**2** + tok22528 | **AssertionError** in MoE router recompute (`topk_routing_with_score_function: input_ids is not None and not requires_grad`) — block/N<3 trips a Megatron recompute bug (block3/4 work) |

Structural findings:
- **Activation memory ≈ 43·tokens/(PP·CP) is ~constant** for PP·CP=42, so CP/PP rebalancing does NOT create headroom (P2 OOM confirms). Higher CP only lowers the per-rank max_tokens floor.
- **Recompute reduction via block/N is exhausted**: block3≈block4 (P4 was -1.4%), and block<3 hits the router-recompute assertion (real code fix needed). selective/none need more memory than block.
- **MFU formula counts attention as DENSE O(seqlen²)** (flops_utils.py:35-46) though the kernel is sparse (~0.5%). So the numerator is dominated by phantom attention (~28× the MoE term at L=86K) — the workload is overhead-bound, not FLOP-bound; real-FLOP floor is ~1s, so high MFU is physically possible if overhead (recompute+memory-bound kernels+bubble) is removed.
- Pipeline bubble is ~16% at tok32768 (38 microbatches), ~11% at tok22528 (53 µb).
- Data ≤128K filter applied (albaliang_077_le128k.jsonl, 49667 samples); seq_length 134136→131208 (divisible by 2·CP for CP∈{2,3,6,7,14}).

Next: P4 = selective + **CPU-offload optimizer** (frees ~25GB to clear the 3GB OOM margin; selective recomputes only moe_act/layernorm/mla_up_proj, NOT the router, so avoids the block-recompute assertion).

### Campaign conclusion — config space is exhausted at the EFA baseline (2026-06-05)

| # | config | TFLOPS | MFU | note |
|---|--------|-------:|----:|------|
| E0 | PP7/CP6 block4 tok32768 (EFA) | 71.0 | 7.2% | config optimum |
| P1 | E0 + FP8 MoE | 71.2 | 7.2% | no gain (not GEMM-bound) |
| P2 | CP7 selective | OOM | — | mem |
| P3 | block2 | crash | — | router recompute bug (hash layers) |
| P4 | selective+offload | crash | — | CheckpointWithoutOutput backward bug |
| P5 | block4 **tok22528** | 14.1 | 1.4% | **4.5x worse** — more microbatches multiply per-µb fixed cost |

Conclusions:
- **max_tokens=32768 is a sharp optimum** (tok22528 = -80%, tok36864 = -45% per old P5). Per-microbatch fixed overhead (V4 indexer/compressor + EP all-to-all) dominates; fewer, bigger microbatches win until memory/packing breaks.
- **Recompute is locked at block4**: block<3 hits the hash-router input_ids assert (needs a Megatron-core patch to thread input_ids into the block-recompute branch); selective moe_act/layernorm hit the CheckpointWithoutOutput backward bug (only core_attn is safe, but it keeps MoE activations → OOM). Even a perfect recompute fix only removes ~18% (recompute is ~18% of the step).
- **Activation memory ≈ constant across PP·CP=42 splits**, so topology rebalancing gives no headroom.
- The step is compute-bound now (377W during compute, vs 118W pre-EFA) but the sparse-MLA/indexer/MoE kernels are memory-bound (~54% of TDP). The MFU numerator counts attention as dense O(n²) while the kernel is sparse → reaching 40% is physically possible but requires KERNEL work, not config.

**Config optimum = E0 (EFA + PP7/CP6/block4/tok32768/le128k).** Path to 40% MFU is engineering, prioritized:
1. Patch Bug 1 (thread input_ids into block-recompute) → enables block/3-2 → small.
2. Reduce per-microbatch fixed cost: optimize the V4 indexer/compressor kernels (the quadratic-ish indexer score+topk runs per C4 layer per microbatch) — likely the biggest real-time sink.
3. Tune the tilelang sparse-MLA kernel for H200 (block sizes, bwd num_stages>0 pipelining; the HW_TILELANG_* knobs are currently dead).
4. (Later, per user) FP8 attention.

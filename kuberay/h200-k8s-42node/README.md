# H200 42-node Ray cluster (production, plain Kubernetes)

The manifests that actually run the current H200 SFT fleet in namespace `ray-system`. This is
**not** KubeRay — the cluster has no `ray.io` operator/CRDs. The Ray cluster is three plain
objects, captured from the live cluster on 2026-06-09 with runtime-only fields stripped:

| File | Object | What it is |
|------|--------|------------|
| `ray-head.yaml` | Deployment `ray-head-gpu-node-64` + Service | 1 CPU-only head pod (num-cpus/gpus=0), pinned to head node `hyperpod-i-0d4e32b7b3a91c397` (10.3.234.60). Hosts GCS + dashboard/job server (:8201). `run.sh` submits training via `kubectl exec` into this Deployment. |
| `ray-gpu-worker.yaml` | DaemonSet `ray-gpu-worker-h200-k8s-42node` | 42 GPU worker pods (8 H200 each = 336 GPU), pinned to the 42-node keep-list. No autoscaler. |
| `bring_up.sh` | — | Apply head→workers in order, wait for rollout, verify 43 nodes / 336 GPU. Also `--delete` / `--dry-run` / `--head-only`. |

`raycluster-h20-16node.yaml` (parent dir) is an unused KubeRay migration draft; these files are
the real thing.

## Bring up from zero

```bash
kuberay/h200-k8s-42node/bring_up.sh --dry-run   # validate manifests (no changes)
kuberay/h200-k8s-42node/bring_up.sh             # apply + wait + verify capacity
```

Head goes up first on purpose: each worker blocks up to 300 s waiting for the head GCS
(`10.3.234.60:6379`) before `ray start`.

## Built-in fixes (worker DaemonSet)

The worker manifest is the **post-patch** state — a fresh `kubectl apply` already includes every
h200-k8s runbook fix, so no follow-up `cluster/k8s/patch_ray_worker_storage.sh` is needed:

1. ray `--temp-dir=/ray_spill_local/ray` on the container overlay (not the small ~100 GB hostPath
   `/ray_local`) so object spill can't fill the node NVMe and OOM-kill workers.
2. `--object-store-memory=230000000000` (fits the 256 Gi `/dev/shm`).
3. `deep_ep` pip-installed at startup from the fsx wheel (stock image lacks it; needed by
   `PRESET_MOE_DEEPEP=1`).

NCCL 600 s dist-ckpt timeout and the 3.7 TB warmup save are run.sh-side (`--distributed-timeout-minutes`
/ `--no-save-optim`), not in these manifests.

## Editing the fleet

The 42 worker instance IDs in `ray-gpu-worker.yaml` (`nodeAffinity`) ARE the fleet; keep them in
sync with `cluster/fleet/h200_k8s_42node.env` `V4_WORKER_IPS`. The image tag
(`radixark/miles:sft-only-v4deps-20260603`) appears in both manifests and the fleet env — bump all
three together.

## Verify against live

```bash
kubectl diff -f ray-head.yaml -f ray-gpu-worker.yaml   # should show no functional change
```

# KubeRay migration draft

This directory is the first migration layer from the current bare-metal
Docker-Ray workflow to KubeRay. It is intentionally additive: the existing
`cluster/` scripts remain the source of truth for the running H20 job until a
Kubernetes GPU/RDMA cluster is available.

## Target topology

```text
Kubernetes namespace v4-train
  RayCluster v4-h20-16node
    head pod
      Ray head / dashboard / job server
    16 worker pods
      each pod uses the training image and requests 8 GPUs
      each pod runs Ray worker and hosts Miles/Megatron actors

Prometheus / Grafana
  should be managed by the Kubernetes monitoring stack, not by host Docker
```

This matches the KubeRay model: `RayCluster` owns the Ray head and worker pods,
with pod configuration under `headGroupSpec` and `workerGroupSpecs`; GPU access
is expressed through Kubernetes `nvidia.com/gpu` resources.

## Files

- `raycluster-h20-16node.yaml` - draft RayCluster for the current 16-node H20 fleet.
- `check_prereqs.sh` - lightweight checks for kubectl, KubeRay CRDs, GPU nodes, and storage paths.

## Required platform prerequisites

1. Kubernetes is installed across the H20 nodes.
2. KubeRay operator CRDs are installed.
3. NVIDIA GPU device plugin or GPU Operator exposes `nvidia.com/gpu`.
4. `/data_train` is available on every node and mounted into pods.
5. Each node has a local disk path mounted as `/ray_local` inside pods.
6. InfiniBand/RDMA devices are exposed to pods, either through hostPath or an RDMA device plugin.
7. The training image is available to the Kubernetes container runtime.

## Node labels

The draft manifest expects labels like:

```bash
kubectl label node <h20-node> v4.echo/gpu-model=h20
kubectl label node <h20-node> v4.echo/fleet=h20-16node
kubectl label node <head-node> v4.echo/ray-head=true
```

Because each worker pod requests 8 GPUs, Kubernetes should schedule at most one
worker pod on an 8-GPU H20 node.

## Next code step

`run.sh` currently does two things in one script:

1. resolve fleet/scale/workload config and generate the in-container launch script;
2. submit that launch script with `ray job submit`.

For KubeRay, those must be separated. `RayJob.spec.entrypoint` should run the
Miles/Megatron training command directly. It should not call the current
`run.sh`, because that would nest a Ray job submit inside another Ray job.

The next refactor should split `run.sh` into:

```text
generate_launch.sh     # config resolution + launch script generation
submit_docker_ray.sh   # current ssh/docker/ray job submit path
submit_kuberay.sh      # render/apply RayJob from the generated training entrypoint
```

Only after that split should we add a real RayJob manifest for the 128K 3 epoch
training workload.

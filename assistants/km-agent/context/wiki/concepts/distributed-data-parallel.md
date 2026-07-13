---
title: "Distributed Data Parallel (DDP)"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/ddp.md]
related: [pytorch, deep-learning-training]
code: [code/deep-learning/ddp/]
tags: [pytorch, distributed-training, multi-gpu, deep-learning, allreduce, nccl, torchrun]
---

# Distributed Data Parallel (DDP)

Distributed Data Parallel (DDP) is a PyTorch framework (`torch.distributed`) for training machine learning models across multiple GPUs or machines while maintaining model consistency. It is a fundamental technique for scaling deep learning training beyond single-device limitations.

## How DDP Works

### Single-Host Training
On a single host with one GPU, training follows the standard pipeline: forward pass → loss computation → backward propagation → gradient optimization.

### Multi-Host / Multi-GPU Training
DDP extends this by:
1. **Replicating** the model and optimizer across all hosts/GPUs.
2. **Partitioning** the dataset so each replica trains on a different subset.
3. **Synchronizing** gradients after each backward pass using the bucketed Ring AllReduce algorithm, ensuring all replicas converge with identical weights.

### Gradient Overlap
DDP overlaps gradient computation with inter-replica communication. Instead of waiting for all gradients, communication begins along the ring while the backward pass continues. This keeps GPUs fully utilized throughout training.

## Architecture Components

- **Process Groups**: Each GPU runs an independent process. A master node coordinates across GPUs and machines.
- **Backend**: NVIDIA NCCL is the primary communication backend for GPU clusters. GLOO is available for CPU or single-GPU testing.
- **`torch.multiprocessing`**: Spawns training functions across processes, assigning a unique `rank` to each.
- **`DistributedSampler`**: Partitions the dataset across processes, ensuring each replica sees a distinct subset.

## Using `torchrun`

`torchrun` is PyTorch's modern launcher for distributed training, offering:
- **Worker recovery**: Automatically restarts workers from snapshots on failure.
- **Environment management**: Sets `LOCAL_RANK`, `RANK`, and other distributed variables.
- **Multi-node support**: Coordinates across machines using rendezvous backends (e.g., `c10d`).

Example:
```sh
torchrun --standalone --nproc_per_node=gpu multi_gpu_torchrun.py 50 10
```

## Multi-Node Considerations

- Use `LOCAL_RANK` for intra-machine GPU indexing and a global `RANK` across machines.
- Select a master node with high-bandwidth networking.
- Configure network security policies (e.g., AWS security groups) to allow inter-node TCP traffic.
- Prefer single-machine multi-GPU setups over cross-node splitting for efficiency.

## Checkpointing

Only one process (typically rank 0) should handle checkpoint saves to avoid contention:
```python
if self.gpu_id == 0 and epoch % self.save_interval:
    save_checkpoint(...)
```

## Code Examples

- [multi_gpu_ddp.py](https://github.com/jbcodeforce/ML-studies/tree/master/code/deep-learning/ddp/multi_gpu_ddp.py) — Basic DDP with `DistributedSampler`.
- [multi_gpu_torchrun.py](https://github.com/jbcodeforce/ML-studies/tree/master/code/deep-learning/ddp/multi_gpu_torchrun.py) — Multi-GPU via `torchrun`.
- [multinode.py](https://github.com/jbcodeforce/ML-studies/tree/master/code/deep-learning/ddp/multinode.py) — Multi-node global rank example.

## Related Resources

- [PyTorch DDP Tutorial Series](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series)
- [nanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT) — Reproduction of GPT-2 at 124M parameters.
- [NCCL Installation Guide](https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html)

## Sources
- [DDP (raw/studies/coding/ddp.md)](../summaries/ddp.md)

## Related
- [Deep Learning Training](deep-learning-training.md) (future article)
- [PyTorch](pytorch.md) (future article)
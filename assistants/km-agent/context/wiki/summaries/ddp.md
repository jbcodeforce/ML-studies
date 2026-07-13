---
title: "DDP (Distributed Data Parallel)"
source: raw/studies/coding/ddp.md
ingested: 2026-06-19
tags: [pytorch, distributed-training, multi-gpu, deep-learning]
type: summary
---

# DDP Summary

Distributed Data Parallel (DDP) is a PyTorch-based approach for training models across distributed compute resources while preserving model integrity. The core idea is to replicate the model across multiple hosts/GPUs, partition the dataset among them, and synchronize gradients during training.

## Key Concepts

- **Data Splitting**: The dataset is divided across hosts, each running the same initial model and optimizer. Different data leads to different gradients per replica.
- **Gradient Synchronization**: Before gradient optimization, DDP aggregates gradients from all replicas using the bucketed [Ring AllReduce algorithm](https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da), ensuring all model replicas maintain identical weights.
- **Overlapped Communication**: DDP overlaps gradient computation with inter-model communication, allowing GPUs to remain fully utilized even during sync phases.
- **Process Groups**: On multi-GPU machines, each GPU runs a separate process coordinated by a master node. Communication uses the NVIDIA NCCL backend (or GLOO for single-GPU testing).
- **torchrun**: A modern launcher providing worker failure recovery via snapshots (model state, optimizer state, epoch). Handles environment setup including `LOCAL_RANK` for GPU identification.
- **Multi-node Training**: Across machines, global ranks coordinate communication. `torchrun` supports multi-node setups with rendezvous backends. Performance favors single-machine multi-GPU over cross-machine splitting due to network overhead.

## Code Samples

- `multi_gpu_ddp.py` — Basic DDP training with `DistributedSampler` and `DistributedDataParallel`.
- `multi_gpu_torchrun.py` — Multi-GPU training on a single machine via `torchrun`.
- `multinode.py` — Multi-machine training demonstrating global rank coordination.

## References

- [PyTorch DDP Tutorial Series](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series)
- [nanoGPT by Andrej Karpathy](https://github.com/karpathy/nanoGPT)
- Ring AllReduce visual intuition (towardsdatascience.com)
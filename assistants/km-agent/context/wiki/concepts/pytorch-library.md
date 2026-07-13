---
title: "PyTorch Library"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/pytorch.md]
related: [distributed-data-parallel, python-ml-development-environment]
code: [code/deep-learning/get_started/, code/computer-vision/, code/classification/]
tags: [pytorch, deep-learning, python, tensors, neural-networks, gpu]
---

# PyTorch Library

PyTorch is the most popular open-source Python library for machine learning and deep learning. It serves as both a low-level math library (similar to NumPy) and a high-level framework for building neural networks, with the ability to compile compute graphs into efficient C++/CUDA code for GPU/TPU execution.

## Environment Setup

PyTorch can be installed via pip or Anaconda/Miniconda. Virtual environments are recommended, with packages `torch`, `torchvision`, and `torchaudio`. Jupyter integration is supported via `ipykernel` registration.

## Core Concepts

### Tensors
Tensors are PyTorch's fundamental data structure — n-dimensional matrices similar to NumPy's ndarrays, but with GPU support. Tensors on CPU share memory with NumPy arrays. Key operations include creation from data or NumPy arrays, device placement, and dtype specification.

### Neural Network Construction
PyTorch uses `torch.nn` for computational graph building blocks and `torch.optim` for optimization algorithms. Models are declared by subclassing `nn.Module` and implementing a `forward()` method, or alternatively via `nn.Sequential` for simpler architectures.

### Loss Functions & Optimizers
- **Binary Cross Entropy** — for binary classification
- **Cross Entropy** — for multi-class classification
- **L1/L2 (MAE/MSE)** — for regression
- **SGD and Adam** — common optimizers

### Training Loop
The standard PyTorch training loop consists of: forward pass → calculate loss → zero gradients → backpropagation (`loss.backward()`) → optimizer step (`optimizer.step()`). Inference uses `model.eval()` with `torch.inference_mode()`.

### GPU Support
PyTorch supports CUDA (Linux/Windows with NVIDIA), MPS (macOS), and CPU. Tensors are moved to devices via `.to(device)` and can be converted back to NumPy via `.cpu().numpy()`.

## Model Improvement Techniques
Adding layers/hidden units, fitting longer, changing activation functions, adjusting learning rate, and using transfer learning are the primary model improvement strategies.

## Datasets & Transfer Learning
PyTorch provides domain libraries (TorchVision, TorchText, TorchAudio) for loading datasets. Data augmentation via `torchvision.transforms` improves generalization. Transfer learning involves freezing pre-trained model features and customizing the classifier head.

## Sources
- [PyTorch Library](../summaries/pytorch.md)

## Related
- [Distributed Data Parallel (DDP)](distributed-data-parallel.md)
- [Python ML Development Environment](python-ml-development-environment.md)
# PyTorch Library

## Summary

PyTorch is the most popular open-source Python library for machine learning and deep learning, combining low-level tensor computation with high-level neural network construction. It compiles compute graphs into efficient C++/CUDA code for GPU/TPU execution.

## Key Topics

- **Tensors**: n-dimensional matrices similar to NumPy arrays, with GPU acceleration and automatic gradient tracking
- **Neural Networks**: Built using `torch.nn` (building blocks) and `torch.optim` (optimizers), declared via `nn.Module` subclassing or `nn.Sequential`
- **Training Loop**: Forward pass → loss calculation → zero gradients → backpropagation → optimizer step
- **Loss Functions**: BCE for binary classification, CrossEntropy for multi-class, L1/L2 for regression
- **GPU Support**: CUDA (NVIDIA), MPS (macOS), CPU fallback
- **Data Handling**: TorchVision, TorchText, TorchAudio libraries; data augmentation via `torchvision.transforms`
- **Transfer Learning**: Freeze pre-trained features, customize classifier head for new tasks
- **Model Evaluation**: Accuracy, precision, recall, F1-score, confusion matrices

## Connections

- Extends concepts from [Distributed Data Parallel](distributed-data-parallel.md) (DDP training)
- Part of the [Python ML Development Environment](python-ml-development-environment.md)
- Code samples available in the studies repo under `code/deep-learning/`, `code/computer-vision/`, and `code/classification/`
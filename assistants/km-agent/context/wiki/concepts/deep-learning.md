---
title: "Deep Learning"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/deep-learning.md]
related: [neural-networks, convolutional-neural-networks, recurrent-neural-networks, transfer-learning, regularization]
code: [code/deep-learning/, code/computer-vision/, code/regression/neuralnetwork/]
tags: [deep-learning, neural-networks, ml, pytorch, tensorflow]
---

# Deep Learning

Deep learning is a machine learning technique that uses neural networks with more than one hidden layer. It is a subset of machine learning inspired by the structure and function of the human brain.

## Core Characteristics

- **Multi-layered architecture**: Unlike traditional ML, deep learning stacks multiple layers of neurons to learn hierarchical representations of data.
- **Automatic feature learning**: Deep networks learn features directly from raw data rather than relying on hand-engineered features.
- **Data-hungry**: Performance improves significantly with more training data.
- **GPU-accelerated**: Modern deep learning relies heavily on GPU computing for efficient training.

## Major Frameworks

The two dominant Python frameworks are:

- **TensorFlow/Keras** (Google) — High-level API with strong production deployment support
- **PyTorch** (Meta) — Dynamic computation graphs, preferred in research

## Learning Paradigms

- **Supervised learning**: Training with labeled data (e.g., images with class labels)
- **Unsupervised learning**: Learning patterns from unlabeled data
- **Self-supervised learning**: Generating supervisory signals from input data relationships
- **Transfer learning**: Reusing pre-trained models on new tasks

## Key Hyperparameters

| Parameter | Typical Values |
|---|---|
| Hidden layers | 1 to unlimited (problem-specific) |
| Neurons per layer | 10 to 512 |
| Hidden activation | ReLU (most common) |
| Output activation | Sigmoid (binary), Softmax (multi-class), Linear (regression) |
| Loss function | Cross entropy (classification), MSE (regression) |
| Optimizer | SGD, Adam |

## Training Considerations

- **Overfitting**: When training loss is much lower than test loss; addressed via regularization, dropout, and data augmentation.
- **GPU vs CPU**: Deep learning benefits from GPU compute; optimize by minimizing data transfer between CPU and GPU.

## See Also

- [Neural Networks](neural-networks.md) for the foundational building blocks
- [Convolutional Neural Networks](convolutional-neural-networks.md) for image processing
- [Recurrent Neural Networks](recurrent-neural-networks.md) for sequential data
- [Transfer Learning](transfer-learning.md) for reusing pre-trained models

## Sources
- [Deep Learning](../summaries/deep-learning.md)

## Related
- [Neural Networks](neural-networks.md)
- [Convolutional Neural Networks](convolutional-neural-networks.md)
- [Recurrent Neural Networks](recurrent-neural-networks.md)
- [Transfer Learning](transfer-learning.md)
- [Regularization](regularization.md)
- [Distributed Data Parallel](distributed-data-parallel.md)
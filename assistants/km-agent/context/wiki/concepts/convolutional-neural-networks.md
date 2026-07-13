---
title: "Convolutional Neural Networks"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/deep-learning.md]
related: [deep-learning, neural-networks, computer-vision]
code: [code/computer-vision/]
tags: [cnn, convolution, image-processing, computer-vision, deep-learning]
---

# Convolutional Neural Networks

Convolutional Neural Networks (CNNs) are neural network architectures designed to process grid-like data, especially images. They capture spatial and temporal dependencies through the application of learnable filters (kernels).

## How CNNs Work

An image consists of three matrices matching the picture dimensions (H×W) for RGB values (R, G, B matrices). CNNs reduce the size of these matrices without losing meaning by using **kernels** — small windows that slide across the image.

## Architecture

A typical CNN structure:

```
Input layer → [Convolutional layer → Activation layer → Pooling layer] → Output layer
```

The layers in brackets can be repeated multiple times. Each layer compresses data from higher-dimensional space to lower-dimensional space.

### Key Layers

- **Conv2d**: Compresses information by applying learnable filters. Parameters include `in_channels`, `out_channels`, `kernel_size`, `stride`, and `padding`.
- **ReLU (Activation)**: Introduces non-linearity, allowing the network to learn complex patterns.
- **MaxPool2d**: Takes the maximum value from a portion of the tensor, reducing spatial dimensions and providing translation invariance.

## Why CNNs for Images?

CNNs allow input size to change without retraining. Instead of treating each pixel as an independent feature (as in a standard neural network), CNNs define a neuron as a unique image pattern (e.g., a 3×3 kernel), learning spatial hierarchies of features — from edges to textures to objects.

## Example: Fashion MNIST Classification

A simple CNN for classifying Fashion MNIST images:

```python
nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
nn.ReLU(),
nn.MaxPool2d(kernel_size=2, stride=2),
```

## Resources

- [CNN Explainer Tool](https://poloclub.github.io/cnn-explainer/) — Interactive visualization
- [MIT Convolutional Neural Network Presentation](https://www.youtube.com/watch?v=iaSUYvmCekI)

## Sources
- [Deep Learning](../summaries/deep-learning.md)

## Related
- [Deep Learning](deep-learning.md)
- [Neural Networks](neural-networks.md)
- [Transfer Learning](transfer-learning.md)
---
title: "Deep Learning"
source: studies:ml/deep-learning.md
created: 2026-07-12
---

# Deep Learning Summary

Deep learning is a machine learning technique using multi-layered neural networks. The document covers the full spectrum of deep learning from fundamentals to advanced topics.

**Key Topics:**

- **Neural Networks**: Biologically-inspired programming approach using layers of connected neurons (input, hidden, output, bias). Modern networks use ReLU activation instead of sigmoid to avoid gradient saturation.
- **Classification Architecture**: Hyperparameters include hidden layer count (1+), neurons per layer (10-512), output shape (1 for binary, N for multi-class), activation functions (ReLU hidden, Softmax/Sigmoid output), and optimizers (SGD, Adam).
- **Recurrent Neural Networks (RNNs)**: Process sequential data with hidden state. LSTMs solve vanishing gradients via forget, input, and output gates. Used for sentiment analysis, time series, language modeling.
- **Convolutional Neural Networks (CNNs)**: Process images via kernels sliding over pixel matrices. Architecture: Conv2d → ReLU → MaxPool2d repeated, then output. Captures spatial dependencies.
- **Transfer Learning**: Fine-tune pre-trained models (e.g., EfficientNet) on custom data by freezing base features and retraining classifier head. Reduces data, cost, and time.
- **Regularization**: L1 (sparse), L2 (weight decay), Dropout (random neuron removal), Batch Normalization (stabilizes training). Prevents overfitting.
- **Data Augmentation**: Artificially diversify training data via transforms (flips, rotations, color jitter). Applied to training data only.
- **Distributed Training**: DDP replicates models across GPUs, splits batches, syncs gradients via Ring AllReduce. Use `torchrun` for multi-GPU.

**Frameworks**: TensorFlow/Keras (Google) and PyTorch (Facebook) are the two dominant Python deep learning frameworks.

**Code examples**: Shallow/deep nets, CNNs (Fashion MNIST, TinyVGG), RNNs (IMDB sentiment), transfer learning, DDP, and a medical image classification project.
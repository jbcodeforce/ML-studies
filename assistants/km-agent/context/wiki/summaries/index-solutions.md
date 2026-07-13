# ML Studies: Classification and Computer Vision Solutions

This document summarizes two practical machine learning study projects from the ML-studies repository.

## Main Projects

**Mammogram Mass Classification**: A binary classification task using Multi-Layer Perceptrons (MLPs) to determine whether mammographic masses are benign or malignant. The study uses the UCI Mammographic Mass dataset and focuses on data quality challenges—including missing values and outlier detection—before training with scikit-learn.

**Food Image Classification**: A computer vision project classifying food images (sushi, pizza, steak) from the Food-101 dataset. The study progresses from a basic neural network (Tiny VGG) to more sophisticated CNNs, and finally demonstrates transfer learning techniques. Executable PyTorch scripts are provided for hands-on experimentation.

## Key Techniques

- **MLP for tabular classification**: Using scikit-learn with numpy data transformations for binary medical classification.
- **CNNs for image classification**: Building neural networks with Conv2d → ReLU → MaxPool2d layer patterns.
- **Transfer learning**: Leveraging pre-trained models to improve classification accuracy on limited datasets.

## Connections

These studies connect to broader concepts in the wiki: [classification](../concepts/classification.md), [convolutional neural networks](../concepts/convolutional-neural-networks.md), [transfer learning](../concepts/transfer-learning.md), and [PyTorch](../concepts/pytorch-library.md). They serve as practical examples of applying ML fundamentals to real-world datasets.
---
title: "ML Studies: Classification and Computer Vision Solutions"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/solutions/index.md]
related: [classification, convolutional-neural-networks, transfer-learning, pytorch-library, scikit-learn]
tags: [classification, computer-vision, ml-studies, medical-imaging, food-classification]
---

# ML Studies: Classification and Computer Vision Solutions

Overview of two practical machine learning study projects covering classification with MLPs and image classification with CNNs.

## Mammogram Mass Classification

A binary classification study that uses a Multi-Layer Perceptron (MLP) to classify mammographic masses as benign or malignant. The dataset is sourced from the University of Irvine's Mammographic Mass dataset. Key challenges include handling missing data and identifying erroneous outliers. The approach involves data quality review, dropping rows with excessive missing values, and transforming data for scikit-learn using numpy.

## Food Image Classification

A computer vision study classifying food images (sushi, pizza, steak) using the Food-101 dataset from PyTorch vision. The study progresses from a basic neural network to comparing against existing CNNs, and finally applying transfer learning. Demonstrations are available via PyTorch scripts using Tiny VGG and transfer learning approaches.

## Sources
- [Solutions Index](../summaries/index-solutions.md)

## Related
- [Classification](classification.md)
- [Convolutional Neural Networks](convolutional-neural-networks.md)
- [Transfer Learning](transfer-learning.md)
- [PyTorch Library](pytorch-library.md)
- [Scikit-learn](scikit-learn.md)
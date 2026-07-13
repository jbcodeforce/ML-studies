---
title: "KNN Classifier"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/classifier.md]
related: [classification, overfitting-underfitting, normalization]
tags: [knn, k-nearest-neighbors, classification, nonparametric, lazy-learning, curse-of-dimensionality]
---

# KNN Classifier

**K-Nearest Neighbors (KNN)** is a **nonparametric**, **instance-based**, **lazy-learning** classifier that predicts class labels by finding the K closest training examples and taking a majority vote.

## How It Works

1. Choose the number **K** and a distance metric
2. Find the K nearest neighbors of the sample to classify
3. Assign the class by majority vote (scikit-learn breaks ties by preferring closer neighbors)

## Nonparametric vs Parametric

- **Parametric models**: Estimate a fixed set of parameters during training; classify new data without needing the training set.
- **Nonparametric models**: No fixed parameter set; the number of parameters grows with the training data (decision trees, random forests, kernel SVM).

KNN belongs to a subcategory called **instance-based learning**, where the training dataset is memorized. It is also a **lazy learning** model — prediction incurs computational cost but training has zero cost.

## Curse of Dimensionality

KNN is very susceptible to overfitting due to the **curse of dimensionality**: as the number of dimensions increases, the feature space becomes increasingly sparse for a fixed-size training set, making distance-based measures unreliable.

## Sources
- [Classifiers](../summaries/classifier.md)

## Related
- [Classification](classification.md)
- [Overfitting Underfitting](overfitting-underfitting.md)
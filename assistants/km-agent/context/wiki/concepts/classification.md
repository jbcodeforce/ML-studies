---
title: "Classification"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/classifier.md]
related: [decision-tree, random-forest, perceptron, adaline, logistic-regression, support-vector-machines, knn-classifier]
tags: [classification, ml, supervised-learning, discrete-output]
---

# Classification

**Classification** is a supervised learning task where the goal is to predict one of a small number of discrete-valued outputs (class labels). The task may be **binary** (two classes) or **multi-class** (multiple classes). The machine learning algorithm learns rules to distinguish between possible classes.

## Three Core Components

Every classifier requires three elements:

1. **Representation**: How the classifier is structured — a rule, decision tree, neural network, etc.
2. **Evaluation**: How to assess quality — error count, recall, accuracy (`correct predictions / coverage`), confusion matrices.
3. **Optimization**: How to search among alternatives — greedy search, gradient descent, etc.

## Evaluation: Confusion Matrix

A **confusion matrix** is a square matrix where rows and columns represent class labels. Each cell counts how often the classifier assigned a given label. The **accuracy** is the sum of correct predictions divided by total predictions.

## Common Algorithms

- **Perceptron** — Single-neuron binary classifier with unit-step activation.
- **Adaline** — Perceptron variant using linear activation for continuous weight updates.
- **Logistic Regression** — Produces class probabilities via sigmoid activation.
- **Support Vector Machines** — Maximizes margin between classes; supports non-linear kernels.
- **Decision Trees** — Hierarchical splits based on information gain.
- **Random Forests** — Ensemble of decision trees for better generalization.
- **K-Nearest Neighbors** — Nonparametric instance-based lazy learning.

## Workflow

Before selecting a classifier, inspect and plot the data. Identify a naive classification rule to form an initial hypothesis, then evaluate with a confusion matrix.

## Sources
- [Classifiers](../summaries/classifier.md)

## Related
- [Decision Tree](decision-tree.md)
- [Random Forest](random-forest.md)
- [Perceptron](perceptron.md)
- [Logistic Regression](logistic-regression.md)
- [Support Vector Machines](support-vector-machines.md)
- [KNN Classifier](knn-classifier.md)
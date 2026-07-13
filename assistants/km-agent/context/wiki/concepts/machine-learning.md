---
title: "Machine Learning"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/index.md]
related: [supervised-learning, unsupervised-learning, reinforcement-learning, cost-function-gradient-descent]
tags: [ml, overview, machine-learning]
---

# Machine Learning

Machine learning is a system that automatically learns programs and functions from data without explicit programming. The goal is to discover a function that predicts **y** from a set of features **X**, continuously measuring prediction performance.

## Core Distinction from Statistics

Statistics applies models of the world (linear regression, logistic regression, Cox model) to data. Machine learning discovers functions from data through algorithms, without requiring a pre-specified model of the world.

## Major Categories

- **Supervised learning** — Learns from labeled training data to predict unseen or future data (classification and regression).
- **Unsupervised learning** — Explores data structure without guidance from known outcome variables (clustering).
- **Reinforcement learning** — Develops an agent that improves through environment interaction, maximizing rewards via trial-and-error or deliberative planning.

## ML System Pipeline

1. **Data preprocessing** — Raw data rarely comes in the optimal form; preprocessing is one of the most crucial steps.
2. **Feature scaling** — Transform features to the same scale (e.g., [0, 1] or standard normal) for optimal algorithm performance.
3. **Dimensionality reduction** — Compress features to lower-dimensional subspace, reducing storage and computation.
4. **Model training** — Fit algorithms to training data.
5. **Cross-validation** — Evaluate using train/test splits, leave-one-out (LOOCV), or k-fold validation.
6. **Model selection** — Compare multiple algorithms using metrics like classification accuracy.
7. **Experiment tracking** — Track metadata and results using tools like TensorBoard, MLFlow, or Weights & Biases.

## Model Representation

- Hypothesis function: `h(x) = θ^T * x` (multivariate linear regression)
- Cost function: Mean squared error `J(θ) = 1/(2m) * Σ(h(x_i) - y_i)²`
- Gradient descent minimizes cost by stepping opposite the gradient, scaled by learning rate

## Sources
- [ML Index](../summaries/index.md)

## Related
- [Supervised Learning](supervised-learning.md)
- [Unsupervised Learning](unsupervised-learning.md)
- [Reinforcement Learning](reinforcement-learning.md)
- [Cost Function and Gradient Descent](cost-function-gradient-descent.md)
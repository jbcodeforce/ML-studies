---
title: "Bias and Variance"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/index.md]
related: [regularization, overfitting-underfitting, performance-metrics]
tags: [ml-fundamentals, model-evaluation]
---

# Bias and Variance

## Bias
**Bias** measures how far off a model's predictions are from the correct values on average. High bias indicates the model is making incorrect assumptions about the data, leading to underfitting. Bias arises from overly simplistic models that fail to capture the underlying patterns.

## Variance
**Variance** measures the consistency of a model's predictions for a given sample instance across multiple training runs. If a model is trained on different subsets of the training data and produces widely varying predictions for the same input, it has high variance. High variance indicates the model is overly sensitive to randomness in the training data, leading to overfitting.

## Bias-Variance Tradeoff
The goal in model development is to find the optimal balance between bias and variance. Reducing bias typically increases variance, and vice versa. Key strategies for managing this tradeoff include:
- **Regularization**: Controls model complexity to prevent overfitting while managing underfitting
- **Model complexity tuning**: Adjusting the number of parameters, layers, or hidden units
- **Cross-validation**: Evaluating performance across multiple data splits

## Sources
- [ML Concepts Overview](../summaries/index.md)

## Related
- [Regularization](regularization.md)
- [Overfitting and Underfitting](overfitting-underfitting.md)
- [Performance Metrics](performance-metrics.md)
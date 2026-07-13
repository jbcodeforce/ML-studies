---
title: "Supervised Learning"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/index.md]
related: [machine-learning, classification, unsupervised-learning]
tags: [supervised-learning, ml, classification, regression]
---

# Supervised Learning

The main goal of supervised learning is to learn a model from labeled training data that allows making predictions about unseen or future data.

## Classification

A **classification** problem predicts one of a small number of discrete-valued outputs. Class labels can be:
- **Binary** — two classes (e.g., yes/no)
- **Multi-class** — more than two classes (e.g., Sunny/Cloudy/Rainy designated as class 0, 1, 2)

The algorithm learns rules to distinguish between possible classes. Examples include topic labeling of documents in a corpus.

## Regression

**Regression** predicts a **continuous value** output. Given predictor (explanatory) variables and a continuous response variable (outcome), the algorithm finds a relationship that allows predicting future outcomes.

Example: Predicting house price from features like square footage and number of bedrooms.

## Model Representation

- Training examples: `m` examples with inputs `X` and outputs `y`
- Individual example: `(x⁽ⁱ⁾, y⁽ⁱ⁾)` for the i-th training example
- Hypothesis function: `h(x) = θ^T * x` (multivariate linear regression)

## Sources
- [ML Index](../summaries/index.md)

## Related
- [Machine Learning](machine-learning.md)
- [Classification](classification.md)
- [Unsupervised Learning](unsupervised-learning.md)
---
title: "Adaline"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/classifier.md]
related: [classification, perceptron, stochastic-gradient-descent]
tags: [adaline, neural-networks, linear-activation, weight-update]
---

# Adaline

**Adaline** (ADAptive LInear NEuron) is a classifier similar to the Perceptron but uses a **linear activation function** (identity function) instead of a unit step function. This enables continuous weight updates based on the actual error.

## Key Differences from Perceptron

| Aspect | Perceptron | Adaline |
|--------|-----------|---------|
| Activation | Unit step | Linear (identity) |
| Weight Update | Based on class error | Based on linear output error |
| Optimization | Discrete | Continuous |

## Feature Standardization

Standardizing features (subtracting mean, dividing by standard deviation) makes Adaline converge more quickly:

```python
X_std[:, 0] = (X[:, 0] - np.mean(X[:, 0])) / np.std(X[:, 0])
X_std[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])
```

## Learning Rate Sensitivity

If the learning rate is too large, the error rate increases with each epoch because the algorithm overshoots the global minimum.

## Stochastic Gradient Descent

For large datasets, **stochastic gradient descent** (SGD) updates weights one sample at a time:

```
wᵢ = η * (yᵢ - φ(zᵢ)) * xᵢ
```

Training data should be shuffled each epoch to prevent cycles.

## Sources
- [Classifiers](../summaries/classifier.md)

## Related
- [Classification](classification.md)
- [Perceptron](perceptron.md)
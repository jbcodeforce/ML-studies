---
title: "Cost Function and Gradient Descent"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/index.md]
related: [machine-learning, supervised-learning, deep-learning]
tags: [cost-function, gradient-descent, optimization, ml, loss-function]
---

# Cost Function and Gradient Descent

A cost function defines the objective to be optimized during the learning process. Weights are updated to minimize the cost function.

## Mean Squared Error Cost Function

The cost function measures sum of squared errors between predictions and target labels:

```
J(θ₀, θ₁, ..., θₙ) = 1/(2m) * Σ(h_θ(xᵢ) - yᵢ)²
```

In Python:
```python
errors = (y - output)
cost = (errors**2).sum() / 2.0
```

## Gradient Descent

Gradient descent minimizes the cost function by iteratively stepping in the opposite direction of the gradient.

### Key Properties
- Works because the cost function is continuous, convex, and differentiable
- Steps downhill until a local or global minimum is reached
- Step size determined by **learning rate (alpha/eta)** and gradient slope

### Weight Update Rule

```
Δwⱼ = -η * δJ/δwⱼ = η * Σ(yᵢ - φ(zᵢ)) * xᵢ,ⱼ
```

Where `η` is the learning rate.

### Convergence Behavior
- At local minimum, tangent slope = 0, so weights stop changing
- Approaching minimum, slope decreases, steps automatically become smaller
- If learning rate is too large, gradient descent can overshoot, fail to converge, or diverge

### Batch Gradient Descent

Weight updates are calculated based on **all samples** in the training set (not incrementally after each sample):

```python
def fit(X, y):
    weights = np.zeros(1 + X.shape[1])
    for _ in range(nbOfIteration):
        output = netInput(X)
        errors = (y - output)
        weights[1:] += eta * X.T.dot(errors)
        weights[0] += eta * errors.sum()
        cost = (errors**2).sum() / 2.0
    return costs
```

### Feature Scaling Requirement

When features have very different units, gradient descent takes much longer to find the minimum. Features should be transformed to the same scale (e.g., [-1, 1] or [0, 1] range).

## Sources
- [ML Index](../summaries/index.md)

## Related
- [Machine Learning](machine-learning.md)
- [Supervised Learning](supervised-learning.md)
- [Deep Learning](deep-learning.md)
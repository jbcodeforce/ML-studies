---
title: "Overfitting and Underfitting"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/index.md]
related: [bias-variance, regularization, performance-metrics]
tags: [ml-fundamentals, model-evaluation, overfitting, underfitting]
---

# Overfitting and Underfitting

## Overfitting
Overfitting occurs when a model performs well on training data but fails to generalize to unseen data. It is characterized by **high variance** — the model is too complex relative to the underlying data, with too many parameters capturing noise rather than signal.

### Methods to Prevent Overfitting

| Method | Description |
| --- | --- |
| **Simplify the model** | Reduce layers or hidden units if the model is too complex |
| **Regularization** | Add penalty terms (L1, L2, Elastic Net) to constrain parameters |
| **Data augmentation** | Artificially increase data variety to improve generalization |
| **Transfer learning** | Leverage pre-trained weights as a foundation for the task |
| **Dropout layers** | Randomly remove connections in neural networks during training |
| **Learning rate decay** | Decrease learning rate as training progresses for finer convergence |
| **Early stopping** | Halt training when validation loss stops improving |

### Decision Boundaries
The decision boundary is the hypothesis that separates the training set. A control factor `C` affects overfitting:
- Decreasing `C` shrinks weight coefficients, potentially leading to overfitting
- Around `C=100`, coefficient values stabilize, producing good decision boundaries

## Underfitting
Underfitting occurs when a model generates poor predictive ability because it hasn't captured the complexity of the training data. It is characterized by **high bias**.

### Methods to Prevent Underfitting

| Method | Description |
| --- | --- |
| **Add model capacity** | Increase hidden layers or units to enable learning complex patterns |
| **Tweak learning rate** | Lower a learning rate that's too high, preventing effective weight updates |
| **Transfer learning** | Use patterns from a pretrained model as a starting point |
| **Train longer** | More epochs may yield better performance |
| **Reduce regularization** | Excessive regularization can constrain the model too much |

## The Tradeoff
Preventing overfitting and underfitting remains an active area of machine learning research. The key is finding the right model complexity for the data at hand.

## Sources
- [ML Concepts Overview](../summaries/index.md)

## Related
- [Bias and Variance](bias-variance.md)
- [Regularization](regularization.md)
- [Performance Metrics](performance-metrics.md)
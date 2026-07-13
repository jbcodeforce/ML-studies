---
title: "Sigmoid Function"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/maths.md]
related: [mathematical-foundations, pytorch-library]
tags: [sigmoid, activation-function, logistic-function, neural-networks, deep-learning]
---

# Sigmoid Function

The sigmoid function is an S-shaped curve that maps any real-valued number to a value between 0 and 1. The most common variant is the logistic function.

## Logistic Sigmoid

**φ(z) = 1 / (1 + e^(−z))**

Properties:
- **Domain**: (−∞, +∞)
- **Range**: (0, 1)
- **Differentiable** everywhere
- At z = 0, φ(z) = 0.5

## Inverse: Logit Function

The logistic sigmoid is invertible. Its inverse, the logit function, maps probabilities back to the real line:

**logit(p) = ln(p / (1 − p))**

Where p is a probability and p/(1−p) is the corresponding **odds**.

## Use in Machine Learning

- **Activation function** in artificial neurons, particularly in the output layer of binary classification models
- Converts raw neuron outputs (logits) into probabilities
- Used as the link function in logistic regression

## Sources
- [Mathematical Foundations](../summaries/maths.md)

## Related
- [Mathematical Foundations](mathematical-foundations.md)
- [PyTorch Library](pytorch-library.md)
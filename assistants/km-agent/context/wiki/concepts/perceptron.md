---
title: "Perceptron"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/classifier.md]
related: [classification, adaline, sigmoid-function]
tags: [perceptron, neural-networks, binary-classification, perceptron-learning-rule]
---

# Perceptron

The **Perceptron** is a single-neuron binary classifier proposed by Frank Rosenblatt, based on a model of human neurons. It automatically learns optimal weight coefficients that are multiplied with input features to decide whether a neuron fires or not.

## How It Works

The problem is reduced to binary classification with outputs of -1 or 1. The **unit step** activation function takes a linear combination of inputs:

```
z = Σ(wᵢ * xᵢ) for i = 1 to n
```

If z exceeds a threshold, the output is 1; otherwise -1.

## Perceptron Learning Rule

Weights are updated using the training set. The delta update rule:

```
Δ(θⱼ) = η * (yᵢ - mean(yᵢ)) * xᵢʲ
```

Where η is the learning rate and yᵢ is the target for sample i. The weight update is proportional to the input value.

## Convergence

The perceptron converges only if:
- The two classes are **linearly separable**
- The learning rate is **sufficiently small**

## Implementation

Python implementations use NumPy for matrix operations:

```python
def netInput(self, X):
    return np.dot(X, self.weights[1:]) + self.weights[0]

def predict(self, X):
    return np.where(self.netInput(X) >= 0.0, 1, -1)
```

Tested on the Iris dataset (setosa vs. versicolor binary classification).

## Sources
- [Classifiers](../summaries/classifier.md)

## Related
- [Classification](classification.md)
- [Adaline](adaline.md)
- [Sigmoid Function](sigmoid-function.md)
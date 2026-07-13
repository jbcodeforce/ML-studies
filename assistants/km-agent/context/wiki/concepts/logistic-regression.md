---
title: "Logistic Regression"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/classifier.md]
related: [classification, sigmoid-function, regularization, perceptron]
tags: [logistic-regression, classification, sigmoid, probability, binary-classification]
---

# Logistic Regression

**Logistic Regression** is a classification approach that performs well on linearly separable datasets. It predicts the probability that a sample belongs to a given class.

## Sigmoid Function

Logistic regression uses the **sigmoid function** as its activation function, mapping a linear combination of features to a probability between 0 and 1:

```
φ(z) = 1 / (1 + e⁻ᶻ)
```

Where z is the net input (W' · x). The output is interpreted as `P(y=1 | x; w)` — the probability of the sample belonging to class 1.

## Log-Odds (Logit)

The mathematical model uses the **log-odds** formulation:

```
logit(P(y=1 | x)) = Σ(θᵢ * xᵢ) = θᵀ · x
```

This transforms probabilities from (0, 1) to the entire real number range, enabling a linear relationship between features and log-odds.

## Cost Function

The cost function for logistic regression includes a **regularization term** to penalize extreme weights:

```
J(w) = C[Σ(-yⁱ log(φ(zⁱ)) - (1-yⁱ)log(1-φ(zⁱ)))] + (1/2)||w||²
```

The parameter `C = 1/λ` controls overfitting — larger C means less regularization.

## Use Cases

- Binary classification (e.g., disease prediction from symptoms)
- Producing class probabilities rather than hard labels
- Online learning via stochastic gradient descent

## Implementation

```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)
lr.predict_proba(X_test_std[0, :])
```

## Sources
- [Classifiers](../summaries/classifier.md)

## Related
- [Classification](classification.md)
- [Sigmoid Function](sigmoid-function.md)
- [Regularization](regularization.md)
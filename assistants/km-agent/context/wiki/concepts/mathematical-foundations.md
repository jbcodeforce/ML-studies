---
title: "Mathematical Foundations"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/maths.md]
related: [bayesian-inference, covariance-correlation, sigmoid-function, normalization]
tags: [mathematics, probability, statistics, machine-learning]
---

# Mathematical Foundations

This document covers essential mathematical concepts underpinning machine learning and data science.

## Main Topics

- **Covariance and Correlation**: Measures of relationship between variables. Covariance captures co-variation, while correlation normalizes it to a standardized range.
- **Probability Theory**: Rules for independent events (P(A ∧ B) = P(A) × P(B)), dependent events via conditional probability (Bayes' rule), disjoint events (P(A ∨ B) = P(A) + P(B)), and non-mutually exclusive events (inclusion-exclusion principle).
- **Bayesian Inference**: The Bayesian approach updates beliefs using prior probabilities, likelihoods, and evidence to compute posteriors. Includes a worked example (the "Steve the librarian" problem) demonstrating the importance of base rates.
- **Data Distributions**: Overview of common distributions — Uniform, Gaussian, Poisson — with associated code exercises.
- **Normalization**: Scaling values to a common range, typically [0, 1], using min-max normalization to enable fair comparison across features.
- **Sigmoid Function**: The logistic function maps real values to (0, 1), serving as a neuron activation function. Its inverse, the logit function, converts probabilities back to the real line.

## Key Data Points

- Bayes' theorem formula: P(H|E) = P(E|H) × P(H) / P(E)
- Min-max normalization: X' = (X − Xmin) / (Xmax − Xmin)
- Logistic sigmoid: φ(z) = 1 / (1 + e^(−z))
- The Steve example shows that even when a description seems to match a librarian, base rates (20:1 farmer-to-librarian ratio) dominate the posterior.

## Connections

These mathematical building blocks feed directly into ML model training (covariance in PCA, sigmoid in neural networks), probabilistic reasoning (Bayesian inference in anomaly detection and classification), and data preprocessing (normalization for feature scaling).

## Sources
- [Mathematical Foundations](../summaries/maths.md)

## Related
- [Bayesian Inference](bayesian-inference.md)
- [Covariance and Correlation](covariance-correlation.md)
- [Sigmoid Function](sigmoid-function.md)
- [Normalization](normalization.md)
- [Anomaly Detection](anomaly-detection.md)
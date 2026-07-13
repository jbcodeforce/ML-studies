---
title: "Mathematical Foundations"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/maths.md]
tags: [mathematics, probability, statistics, machine-learning]
---

# Summary: Mathematical Foundations

This source document covers the core mathematical concepts that underpin machine learning and data science.

## Key Topics

- **Covariance and Correlation**: Formulas for measuring how variables co-vary, with correlation providing a normalized [-1, +1] metric.
- **Probability Rules**: Independent events (P(A∧B) = P(A)×P(B)), conditional probability, disjoint events (P(A∨B) = P(A)+P(B)), and the inclusion-exclusion principle for overlapping events. Includes a worked card-drawing example.
- **Bayesian Inference**: Bayes' theorem (P(H|E) = P(E|H)×P(H)/P(E)) explained through priors, likelihoods, evidence, and posteriors. The "Steve" example demonstrates how base rates can override intuitive priors — a description matching a librarian is outweighed by the 20:1 farmer-to-librarian population ratio, making Steve ~5× more likely to be a farmer.
- **Data Distributions**: Brief mention of Uniform, Gaussian, and Poisson distributions with a linked code exercise.
- **Normalization**: Min-max scaling to [0,1] range for feature preprocessing.
- **Sigmoid Function**: The logistic function φ(z) = 1/(1+e^(-z)) maps reals to (0,1), used as a neural network activation function. Its inverse is the logit.

## Connections

These mathematical building blocks support many other wiki concepts: Bayesian inference underpins anomaly detection and classification; normalization is a key preprocessing step; covariance/correlation inform feature engineering; and the sigmoid function is central to neural network activation.
---
title: "Mutual Information"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/data/features.md]
related: [feature-engineering, feature-selection]
tags: [machine-learning, feature-selection, statistics]
---

# Mutual Information

**Mutual information (MI)** is a univariate feature-selection metric that measures the association between a feature and the target variable.

## Properties

- **Detects any relationship type**: Unlike Pearson correlation, which only captures linear relationships, MI can detect nonlinear associations.
- **Range**: MI ranges from 0.0 upward. A value of 0.0 means the two quantities are independent — neither tells you anything about the other.
- **Univariate only**: MI evaluates each feature in isolation. A feature may be highly informative in combination with others but score low alone.

## Usage

MI is used to rank features by utility, then select a smaller set of the most useful features for initial model development.

### Discrete vs. Continuous Features

Scikit-learn's MI implementation treats discrete and continuous features differently:

- Anything with a float dtype is treated as continuous.
- Categorical features (object or category dtype) should be label-encoded and marked as discrete.

## Limitations

- MI cannot detect **feature interactions** — it is strictly univariate.
- A feature may need transformation before its association with the target is exposed to MI.

## Sources
- [Feature Engineering](../summaries/features.md)

## Related
- [Feature Engineering](feature-engineering.md)
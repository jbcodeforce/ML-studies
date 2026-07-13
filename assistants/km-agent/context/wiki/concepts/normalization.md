---
title: "Normalization"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/maths.md]
related: [mathematical-foundations, feature-scaling]
tags: [normalization, feature-scaling, statistics, machine-learning, preprocessing]
---

# Normalization

Normalization adjusts values measured on different scales to a common scale, typically prior to averaging or comparing features.

## Definitions

- **Statistical normalization**: Creating shifted and scaled versions of statistics so that values can be compared across datasets, eliminating effects of gross influences (e.g., in anomaly time series).
- **Feature scaling (unity-based normalization)**: Bringing all values into the range [0, 1].

## Min-Max Normalization

The standard min-max normalization formula:

**X' = (X − Xmin) / (Xmax − Xmin)**

Where:
- **X** is the original value
- **Xmin** is the minimum value in the dataset
- **Xmax** is the maximum value in the dataset
- **X'** is the normalized value in [0, 1]

## Use Cases

- Bringing heterogeneous features (e.g., age in years, income in dollars) to a common scale before feeding them to ML models
- Ensuring distance-based algorithms (KNN, clustering) are not dominated by features with larger ranges
- Improving gradient descent convergence in neural networks

## Sources
- [Mathematical Foundations](../summaries/maths.md)

## Related
- [Mathematical Foundations](mathematical-foundations.md)
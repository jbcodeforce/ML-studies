---
title: "Covariance and Correlation"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/maths.md]
related: [mathematical-foundations]
tags: [covariance, correlation, statistics, machine-learning]
---

# Covariance and Correlation

Covariance and correlation measure the relationship between two random variables.

## Covariance

Covariance quantifies how two variables change together:

**cov(x, y) = Σ (xᵢ − μₓ)(yᵢ − μᵧ)**

- Positive covariance: variables tend to increase together
- Negative covariance: one tends to increase as the other decreases
- Zero covariance: no linear relationship

## Correlation

Correlation normalizes covariance to a bounded range, making it comparable across different datasets:

**corr(x, y) = cov(x, y) / (√Σ(xᵢ − μₓ)² × √Σ(yᵢ − μᵧ)²)**

Correlation ranges from −1 to +1, where:
- **+1**: perfect positive linear relationship
- **−1**: perfect negative linear relationship
- **0**: no linear relationship

## Sources
- [Mathematical Foundations](../summaries/maths.md)

## Related
- [Mathematical Foundations](mathematical-foundations.md)
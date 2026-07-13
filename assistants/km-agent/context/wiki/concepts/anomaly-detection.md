---
title: "Anomaly Detection"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/anomaly.md]
related: [outlier-detection, gaussian-distribution, unsupervised-learning]
tags: [anomaly-detection, unsupervised, probability, outlier]
---

# Anomaly Detection

Anomaly detection is an unsupervised machine-learning technique that identifies rare or unusual observations by modeling the probability distribution of normal (negative) examples. Unlike supervised classification, it does **not** require positive anomaly labels during training.

## Core Idea

Given a dataset `{x(1), …, x(m)}` of mostly normal examples, fit a model `P(X)` — typically assuming a Gaussian distribution per feature — and then flag any test point where `P(X_test) < ε` (epsilon). Points with very low probability under the fitted distribution are considered anomalies or outliers.

## When to Use

| Anomaly Detection | Supervised Learning |
|---|---|
| Very few positive examples | Abundant labeled positives and negatives |
| Many types of anomalies; future ones may look different | Future positives resemble training positives |
| Fraud detection, data-center monitoring, manufacturing quality checks | Spam filtering, weather prediction, cancer classification |

## Approaches

- **Standard deviation**: On a single numerical feature, flag values beyond 1–3 standard deviations of the mean (covering ~68%, ~95%, ~99.7% of data under normality).
- **Box plots**: A simple visualization that depicts quantiles and reveals outliers graphically.
- **Gaussian modelling**: Fit a Gaussian to each feature; points far from the mean relative to variance receive low probability scores.

## Feature Engineering Tips

- Prefer features that appear bell-curve (approximately Gaussian) in historical data.
- Non-Gaussian features can often be transformed (e.g., log or square root) toward Gaussianity.
- Look for features that take unusually large or small values for anomalous examples.

## Sources
- [Anomaly Detection](../summaries/anomaly.md)

## Related
- [Outlier Detection](outlier-detection.md)
- [Gaussian Distribution](gaussian-distribution.md)
- [Unsupervised Learning](unsupervised-learning.md)
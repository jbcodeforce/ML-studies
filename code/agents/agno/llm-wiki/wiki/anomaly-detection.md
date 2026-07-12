# Anomaly Detection

**Summary**: An overview of anomaly detection objectives, methodology using Gaussian distributions, and key use cases.

**Sources**: anomaly.md

**Last updated**: 2026-04-18

---

Anomaly detection is the process of identifying data points that deviate significantly from the norm. The primary goal is to compute the probability that a test sample $X_{test}$ is anomalous, specifically looking for cases where $P(X_{test}) < \epsilon$ (source: anomaly.md).

### Methodology
The core approach involves fitting a model $P(X)$ using only negative (normal) examples. This is particularly useful when the number of positive (anomalous) examples is extremely small or when anomalies are unpredictable (source: anomaly.md). In this context, a [[gaussian-distribution]] is often used to model the probability of the features (source: anomaly.md).

### Applications
Common use cases include:
* Fraud detection (source: anomaly.md)
* Manufacturing quality testing (source: anomaly.md)
* Data center computer monitoring (source: anomaly.md)

### Key Differences
Unlike [[anomaly-detection-vs-supervised-learning]], which relies on large, labeled datasets of both classes, anomaly detection focuses on the distribution of the "normal" class to find outliers (source: anomaly.md).

## Related pages

- [[gaussian-distribution]]
- [[feature-engineering-for-anomalies]]
- [[anomaly-detection-vs-supervised-learning]]
- [[box-plots]]
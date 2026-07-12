# Gaussian Distribution in Anomaly Detection

**Summary**: A description of using Gaussian distributions to model feature probability for detecting outliers.

**Sources**: anomaly.md

**Last updated**: 2026-04-18

---

In anomaly detection, features are often assumed to follow a [[gaussian-distribution]] to model the probability density of normal examples (source: anomaly.md).

### The 68-95-99.7 Rule
A key property of the normal distribution is the relationship between the mean and the standard deviation:
* Approximately 68% of the data lies within one standard deviation of the mean.
* Approximately 95% of the data lies within two standard deviations.
* Approximately 99.7% of the data lies within three standard deviations (source: anomaly.md).

### Application in Outlier Detection
By fitting a Gaussian model to a dataset of normal examples, we can identify anomalies as points that have a very low probability under the fitted distribution. This is often visually represented via [[box-plots]] or by observing deviations in the standard deviation (source: anomaly.md).

### Feature Transformation
If a feature does not follow a Gaussian distribution, it can be transformed using methods like square root or log transformations to make it more Gaussian-like, which improves the effectiveness of the detection model (source: anomaly.md).

## Related pages

- [[anomaly-detection]]
- [[feature-engineering-for-anomalies]]
- [[box-plots]]
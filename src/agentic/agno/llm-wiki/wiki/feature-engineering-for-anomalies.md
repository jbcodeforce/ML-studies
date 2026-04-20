# Feature Engineering for Anomaly Detection

**Summary**: Techniques for selecting and transforming features to improve the performance of anomaly detection models.

**Sources**: anomaly.md

**Last updated**: 2026-04-18

---

Effective feature engineering is critical for the success of an anomaly detection system.

### Feature Selection
When choosing features, look for those that take on unusually large or small values for the anomalous examples (source: anomaly.md).

### Feature Transformation
Since many anomaly detection models (like those using Gaussian distributions) assume a normal distribution, features that are non-Gaussian can be transformed:
* **Log transformation** (source: anomaly.md)
* **Square root transformation** (source: anomaly.md)

These transformations help align the data with the [[gaussian-distribution]] assumptions.

## Related pages

- [[anomaly-detection]]
- [[gaussian-distribution]]
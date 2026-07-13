# Anomaly Detection Summary

Anomaly detection is an unsupervised ML technique for identifying rare, unusual observations by modeling the probability distribution of normal data. Given a dataset of mostly negative (normal) examples, the goal is to compute `P(X_test) < epsilon` to flag anomalies.

**Key points:**
- Fit a model `P(X)` using only normal examples; no positive anomaly labels are required.
- Typically assumes features follow a Gaussian distribution; outliers are points with very low probability.
- Common applications: fraud detection, manufacturing quality control, data center monitoring, and data quality verification during analysis.
- Best suited when positive examples are few, anomalies vary widely in type, and future anomalies may differ from past ones.
- Contrasted with supervised learning, which is preferred when ample labeled positives and negatives exist (e.g., spam classification, cancer detection).

**Approaches covered:**
- **Standard deviation**: Flag values beyond 1–3 standard deviations of the mean (68%, 95%, 99.7% coverage).
- **Box plots**: Visual method for identifying outliers via quantiles.
- **Feature selection**: Prefer features that are approximately Gaussian and that take unusually large/small values for anomalies. Non-Gaussian features can be transformed (log, square root).

**Related concepts:** Unsupervised learning, Gaussian distribution, outlier detection, supervised learning.
# Anomaly Detection vs. Supervised Learning

**Summary**: A comparison of when to use anomaly detection versus supervised learning based on the availability and nature of positive examples.

**Sources**: anomaly.md

**Last updated**: 2026-04-18

---

Choosing between anomaly detection and supervised learning depends primarily on the characteristics of your dataset, specifically the number and variety of positive examples.

### Use Anomaly Detection When:
* There is a very small number of positive examples ($y=1$) (source: anomaly.md).
* There are many different types of anomalies, making it hard for an algorithm to learn a single pattern (source: anomaly.md).
* Future anomalies may look entirely different from any previously observed anomalies (source: anomaly.md).
* Examples include: Fraud detection, manufacturing testing, and data center monitoring (source: anomaly.md).

### Use Supervised Learning When:
* Both positive and negative examples are plentiful and large (source: anomaly.md).
* There are enough positive examples for the algorithm to learn specific characteristics (source: anomaly.md).
* Future positive examples are likely to be similar to the ones in the training set (source: anomaly.md).
* Examples include: Spam detection, weather prediction, and cancer classification (source: anomaly.md).

## Related pages

- [[anomaly-detection]]
- [[gaussian-distribution]]
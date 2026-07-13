---
title: "Data Leakage"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/kaggle.md]
related: [ml-pipeline, cross-validation, overfitting-underfitting]
tags: [data-leakage, ml, target-leakage, train-test-contamination, pipeline]
---

# Data Leakage

Data leakage occurs when training data contains information about the target that will not be available at prediction time. This causes models to perform well on training and validation sets but fail in production.

## Two Types

### Target Leakage
Predictors include data that won't be available when making predictions. The key consideration is **timing** — whether a feature's value would be known chronologically before the prediction is made, not just whether it correlates with the target.

**Example**: Forecasting monthly shoelace demand using leather consumption. If leather values are reported at month-end (after shoe production), using them as a feature creates leakage.

### Train-Test Contamination
Occurs when training and validation data are not properly separated. Common mistake: fitting a preprocessing step (e.g., imputer for missing values) on the full dataset before calling `train_test_split()`. This leaks information from the validation set into the training set.

**Example**: Predicting surgical infection risk using each surgeon's average infection rate. If the patient's own outcome contributes to their surgeon's rate, that's target leakage. If the rate includes test-set surgeries, that's train-test contamination.

## Prevention
- Use **Pipelines** to bundle preprocessing and modeling, ensuring transforms fit only on training folds.
- Carefully audit feature timing and data lineage.
- Calculate aggregate features (e.g., surgeon statistics) using only historical data prior to each prediction.

## Sources
- [Kaggle and ML Tutorial](../summaries/kaggle.md)

## Related
- [ML Pipeline](ml-pipeline.md)
- [Cross-Validation](cross-validation.md)
- [Overfitting/Underfitting](overfitting-underfitting.md)
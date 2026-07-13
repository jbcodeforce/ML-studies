---
title: "Performance Metrics"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/index.md]
related: [bias-variance, overfitting-underfitting]
tags: [ml-fundamentals, model-evaluation, metrics]
---

# Performance Metrics

## Classification Metrics

### Accuracy
Measures the proportion of correct predictions over the total number of predictions. Best suited for balanced classification problems.

### Precision and Recall
- **Precision**: Ratio of true positive predictions to total predicted positives. Measures the model's ability to correctly identify positive instances.
- **Recall**: Ratio of true positive predictions to total actual positives. Measures the model's ability to find all positive instances.
- Precision and recall are essential for imbalanced classification where the positive class matters most.

### F1 Score
The harmonic mean of precision and recall. Provides a single metric balancing both, useful when both correctly identifying positives and finding all positives matter equally.

### AUC-ROC (Area Under the ROC Curve)
Used for binary classification. Measures the model's ability to distinguish between positive and negative instances across different classification thresholds. Higher values indicate better discrimination.

### Mean Average Precision (MAP)
Used for information retrieval and recommendation systems. Considers average precision at different recall levels to assess ranking or recommendation quality.

## Regression Metrics

### Mean Squared Error (MSE)
Measures the average squared difference between predicted and actual values. Lower values indicate better performance.

### Root Mean Squared Error (RMSE)
The square root of MSE, providing a more interpretable metric in the same units as the target variable.

### R-squared (R²)
Measures the proportion of variance in the target variable explained by the model. Ranges from 0 to 1, with higher values indicating better fit.

## Choosing the Right Metric
- **Balanced classification**: Accuracy
- **Imbalanced classification**: Precision, Recall, F1
- **Binary classification with threshold sensitivity**: AUC-ROC
- **Ranking/recommendation**: MAP
- **Regression tasks**: MSE, RMSE, R²

## Sources
- [ML Concepts Overview](../summaries/index.md)

## Related
- [Bias and Variance](bias-variance.md)
- [Overfitting and Underfitting](overfitting-underfitting.md)
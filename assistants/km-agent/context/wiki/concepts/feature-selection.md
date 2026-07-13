---
title: "Feature Selection"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/skill.md]
related: [overfitting-underfitting, regularization, ml-pipeline, data-scientist-skill-set]
tags: [feature-selection, ml-fundamentals, preprocessing, overfitting, interpretability]
---

# Feature Selection

Feature selection is the process of identifying and selecting the most relevant features for training a machine learning model. It is a critical step that improves model quality while reducing computational cost.

## Benefits

- **Reduce Overfitting**: By removing noisy or irrelevant features, models generalize better to unseen data.
- **Improve Interpretability**: Models with fewer features are easier to understand and explain, as the most important predictors are identified.
- **Reduce Computational Cost**: Fewer features means faster training and inference times, lower memory usage, and reduced infrastructure costs.
- **Improve Accuracy**: Removing noise from the data can lead to more accurate predictions.

## Methods

### Filter Methods
Select features based on statistical properties independent of any machine learning model. Examples include correlation with the target variable and variance. These methods are computationally efficient and work well as a preprocessing step.

### Wrapper Methods
Select features based on their impact on model performance. These methods use a search algorithm to find the subset of features that maximizes performance. They are computationally expensive but often yield better results than filter methods.

### Embedded Methods
Learn feature importance during model training itself. Examples include LASSO (Least Absolute Shrinkage and Selection Operator), which performs feature selection through L1 regularization, and Ridge Regression.

### Ensemble Methods
Combine multiple feature selection approaches for more robust results. Examples include recursive feature elimination and random forests.

## Sources
- [Data Scientist Skill Set](../summaries/skill.md)

## Related
- [Overfitting and Underfitting](overfitting-underfitting.md)
- [Regularization](regularization.md)
- [ML Pipeline](ml-pipeline.md)
- [Data Scientist Skill Set](data-scientist-skill-set.md)
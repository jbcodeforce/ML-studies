---
title: "Data Scientist Skill Set"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/concepts/skill.md]
related: [mathematical-foundations, performance-metrics, regularization, overfitting-underfitting, bias-variance, feature-selection, ml-pipeline]
tags: [data-science, skills, machine-learning, mathematics, statistics, python, visualization, deep-learning]
---

# Data Scientist Skill Set

The data scientist skill set encompasses five core categories: **Mathematics**, **Statistics**, **Python**, **Data Visualization**, and **Machine Learning** (including deep learning). These categories form the foundation for building, evaluating, and deploying effective ML systems.

## Key Topics

- **Supervised vs Unsupervised Learning**: Supervised learning uses labeled training data to build predictive models, while unsupervised learning discovers patterns without knowing the outcome variable upfront.
- **Bias-Variance Tradeoff**: The balance between systematic error (bias) and prediction variability (variance). Model complexity tuning via regularization and cross-validation helps achieve an optimal balance.
- **Regularization**: Techniques to prevent overfitting including L1 (Lasso), L2 (Ridge), and Elastic Net regularization.
- **Overfitting and Underfitting**: Overfitting occurs when a model memorizes training data but fails to generalize; underfitting occurs when a model is too simple to capture underlying patterns.
- **ML Pipeline**: The standard workflow involves: (1) defining the problem, (2) collecting and preparing data, (3) selecting a model, (4) training, (5) evaluating with metrics, (6) optimizing via hyperparameter tuning, and (7) deploying with ongoing monitoring.
- **Classification vs Regression**: Classification predicts categorical labels; regression predicts continuous values.
- **Cross-Validation**: Splitting data into training and validation sets using techniques like k-fold cross-validation to assess performance across different hyperparameter configurations.
- **Feature Selection**: Methods for choosing relevant features to reduce overfitting, improve interpretability, lower computational cost, and increase accuracy. Includes filter methods (statistical properties), wrapper methods (model-based search), embedded methods (LASSO, Ridge), and ensemble methods (recursive feature elimination, random forests).

## Sources
- [Data Scientist Skill Set](../summaries/skill.md)

## Related
- [Mathematical Foundations](mathematical-foundations.md)
- [Performance Metrics](performance-metrics.md)
- [Regularization](regularization.md)
- [Overfitting and Underfitting](overfitting-underfitting.md)
- [Bias and Variance](bias-variance.md)
- [Feature Selection](feature-selection.md)
- [ML Pipeline](ml-pipeline.md)
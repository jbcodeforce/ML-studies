---
title: "Feature Engineering"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/data/features.md]
tags: [machine-learning, data-preprocessing, feature-selection]
---

# Feature Engineering

A concise summary of feature engineering techniques for preparing data for machine learning models.

Feature engineering aims to make data better suited for a given prediction problem. Features are the measurable data points models use to predict outcomes. Key goals include improving model performance, reducing computational needs, and improving interpretability.

## Key Techniques Covered

- **Missing value handling**: Imputation strategies (mean, median), adding missing-value indicator columns, and knowing when to drop columns.
- **Categorical encoding**: Ordinal encoding for ordered categories, one-hot encoding for nominal categories (with cardinality constraints), and handling unseen categories in test sets.
- **Mutual information**: A univariate feature-selection metric that detects any kind of relationship (not just linear), unlike correlation.
- **Feature discovery**: Domain research, studying Kaggle solutions, data visualization for suggesting transformations (powers, logarithms), and aggregating related features into counts.
- **Group transforms**: Aggregating information across rows grouped by a category (e.g., average income by state), using only training data to preserve validation independence.

See the related concept articles for detailed coverage of each technique.

## Related Work

The source references Kaggle's concrete compressive strength example and a car price prediction notebook for practical examples.
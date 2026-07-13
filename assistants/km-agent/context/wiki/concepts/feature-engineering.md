---
title: "Feature Engineering"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/data/features.md]
related: [missing-value-imputation, categorical-encoding, mutual-information, group-transforms, feature-store]
tags: [machine-learning, data-preprocessing]
---

# Feature Engineering

**Feature engineering** is the process of transforming raw data into features (measurable data points) that make a model better suited to the prediction task at hand. For a feature to be useful, it must have a relationship to the target that the model can learn — for example, linear models can only learn linear relationships, so features may need transformation to expose linearity.

## Goals

Feature engineering aims to:

- **Improve predictive performance** — by creating informative features or removing noise.
- **Reduce computational or data needs** — by selecting only the most useful features.
- **Improve interpretability** — by simplifying relationships the model must learn.

## Core Activities

Feature engineering encompasses several activities:

1. **Locating high-potential features** — using metrics like [mutual information](mutual-information.md) to rank feature utility.
2. **Creating new features** — through domain knowledge, aggregation, splitting strings, or combining related features.
3. **Handling missing values** — via imputation strategies and indicator columns. See [Missing Value Imputation](missing-value-imputation.md).
4. **Encoding categorical variables** — converting nominal or ordinal categories into numeric representations. See [Categorical Encoding](categorical-encoding.md).
5. **Analyzing variations and clusters** — using visualization to discover pathologies or opportunities for transformation.

## Principle of Simplicity

The more complicated a feature combination, the more difficult it will be for a model to learn. Simpler transformations — such as log transforms or feature counts — are often more effective than complex interactions.

## Feature Stores

Once features are engineered, they can be managed and served at scale using a [feature store](feature-store.md) — a centralized system that stores, version-controls, and serves features for both training and real-time inference.

## Sources
- [Feature Engineering](../summaries/features.md)

## Related
- [Missing Value Imputation](missing-value-imputation.md)
- [Categorical Encoding](categorical-encoding.md)
- [Mutual Information](mutual-information.md)
- [Group Transforms](group-transforms.md)
- [Feature Store](feature-store.md)
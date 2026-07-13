---
title: "Unsupervised Learning"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/unsupervised.md]
related: [clustering, k-means, feature-engineering]
tags: [ml, unsupervised, clustering, k-means]
---

# Unsupervised Learning

Unsupervised learning algorithms do not use a target variable. Instead, the goal is to discover properties of the data and represent the structure of features in a meaningful way. Unlike supervised learning, there are no labels to predict—the model must find patterns inherent in the data.

## Key Approaches

The primary technique discussed is **clustering**, which groups data points based on their similarity to one another. Clustering can reveal hidden structure in datasets and is widely used for exploratory data analysis, customer segmentation, anomaly detection, and feature engineering.

## Clustering as Feature Engineering

Adding cluster labels as a new categorical feature can help machine learning models decompose complex relationships. The clusters partition complicated feature-space relationships into simpler, more manageable chunks—essentially a "divide and conquer" strategy. The model then learns simpler patterns within each cluster rather than trying to capture the full complexity at once.

## K-Means Clustering

**K-means** is one of the most common clustering algorithms. It works by:
1. Placing `k` centroids in the feature space
2. Assigning each data point to the nearest centroid
3. Reiterating centroid positions and reassigning points until convergence

Key considerations:
- **Scale sensitivity**: K-means uses Euclidean distance, making it sensitive to feature scale. Features should be standardized or normalized when they have different magnitudes.
- **Choosing `k`**: The number of clusters is a hyperparameter that must be specified in advance.
- **When not to scale**: If features are already on comparable scales (e.g., repeated measurements of the same quantity), rescaling is unnecessary.

## Practical Application

In a housing price prediction example, features like lot area and living area were standardized before fitting a K-means model with 10 clusters. The resulting cluster labels and centroid distances were added as new features to the dataset, potentially improving downstream model performance.

## Sources
- [Unsupervised Learning](../summaries/unsupervised.md)

## Related
- [Clustering](clustering.md)
- [K-Means](k-means.md)
- [Feature Engineering](feature-engineering.md)
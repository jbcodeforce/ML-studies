---
title: "Clustering"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/unsupervised.md]
related: [unsupervised-learning, k-means]
tags: [ml, unsupervised, clustering]
---

# Clustering

Clustering is an unsupervised learning technique that groups data points based on how similar they are to one another. The goal is to discover natural groupings within the data without any pre-defined labels.

## Purpose

Clustering serves several purposes:
- **Exploratory data analysis**: Revealing hidden structure in datasets
- **Feature engineering**: Adding cluster labels as categorical features to improve downstream supervised models
- **Data segmentation**: Grouping similar items for targeted analysis

## Cluster Labels as Features

Adding cluster labels to a dataset creates a new categorical feature. The motivating idea is that clusters break up complicated relationships across features into simpler chunks. A downstream model can then learn simpler patterns within each cluster rather than trying to model the full complexity at once—a "divide and conquer" approach.

## Cluster Distances

Beyond cluster labels, the distance of each point to its assigned centroid can also be used as additional features. These centroid-distance features capture how far each observation lies from the center of its cluster, providing continuous signals about cluster membership strength.

## Sources
- [Unsupervised Learning](../summaries/unsupervised.md)

## Related
- [Unsupervised Learning](unsupervised-learning.md)
- [K-Means](k-means.md)
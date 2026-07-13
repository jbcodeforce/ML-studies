---
title: "K-Means Clustering"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/unsupervised.md]
related: [clustering, unsupervised-learning]
tags: [ml, unsupervised, clustering, k-means]
---

# K-Means Clustering

K-means is a popular clustering algorithm that partitions data into `k` clusters by iteratively assigning points to the nearest centroid and updating centroid positions.

## How It Works

1. Initialize `k` centroids in the feature space
2. Assign each data point to the nearest centroid (using Euclidean distance)
3. Recompute centroids as the mean of assigned points
4. Repeat steps 2-3 until convergence

## Scale Sensitivity

K-means is sensitive to feature scale because it measures similarity using Euclidean distance. Features with larger ranges will dominate the distance calculation. **Standardization** (subtracting the mean and dividing by standard deviation) is recommended when features are on different scales.

Exception: If features are already directly comparable (e.g., test results measured at different times), rescaling is not necessary and may even be undesirable.

## Practical Example

```python
from sklearn.cluster import KMeans

features = ["LotArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "GrLivArea"]

# Standardize
X_scaled = (X.loc[:, features] - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)

# Fit KMeans
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
X["Cluster"] = kmeans.fit_predict(X_scaled)

# Create centroid-distance features
X_cd = kmeans.fit_transform(X_scaled)
```

## Sources
- [Unsupervised Learning](../summaries/unsupervised.md)

## Related
- [Clustering](clustering.md)
- [Unsupervised Learning](unsupervised-learning.md)
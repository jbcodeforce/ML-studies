# Unsupervised Learning Summary

Unsupervised learning algorithms discover structure in data without target labels. The primary technique discussed is **clustering**, which groups similar data points together.

**Key points:**
- Clustering creates categorical cluster labels that can be added as features for downstream supervised models
- This is a "divide and conquer" strategy—clusters break complex relationships into simpler chunks
- **K-means** is the main clustering algorithm discussed, using Euclidean distance to assign points to `k` centroids
- Scale normalization is essential for K-means when features have different magnitudes, but not needed for comparable features
- Both cluster labels and centroid distances can be used as engineered features

**Example application:** Housing price prediction, where area-related features were clustered into 10 groups to improve model interpretability.

Connects to concepts of [feature engineering](../concepts/feature-engineering.md), [clustering](../concepts/clustering.md), and [k-means](../concepts/k-means.md).
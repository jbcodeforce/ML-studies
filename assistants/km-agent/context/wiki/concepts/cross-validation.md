---
title: "Cross-Validation"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/kaggle.md]
related: [performance-metrics, ml-pipeline, scikit-learn]
tags: [cross-validation, validation, ml, scikit-learn]
---

# Cross-Validation

Cross-validation evaluates model quality by running the modeling process on different subsets of the data, yielding multiple quality measures. This provides a more robust assessment than a single train/validation split.

## When to Use

- **Small datasets**: Recommended; the computational cost is manageable and reusing data for holdout is valuable.
- **Large datasets**: A single validation set is usually sufficient; extra computation adds little value.

## K-Fold Cross-Validation

With scikit-learn's `cross_val_score`, you specify the number of folds (e.g., `cv=5`):

```python
from sklearn.model_selection import cross_val_score
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
```

Note: sklearn returns *negative* MAE, so multiply by -1 for the actual metric.

## Sources
- [Kaggle and ML Tutorial](../summaries/kaggle.md)

## Related
- [Performance Metrics](performance-metrics.md)
- [ML Pipeline](ml-pipeline.md)
- [Scikit-learn](scikit-learn.md)
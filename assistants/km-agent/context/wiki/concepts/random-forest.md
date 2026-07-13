---
title: "Random Forest"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/kaggle.md]
related: [decision-tree, gradient-boosting, overfitting-underfitting]
tags: [random-forest, ensemble, ml, classification, regression]
---

# Random Forest

Random Forest is an ensemble learning method that builds many decision trees and averages their predictions. It generally achieves much better predictive accuracy than a single decision tree and works well with default parameters.

## How It Works

- Multiple decision trees are trained on bootstrap samples of the data.
- Each tree considers a random subset of features at each split.
- For regression, predictions are averaged; for classification, predictions are voted.

## Key Parameters (scikit-learn)

- `n_estimators`: Number of trees in the forest (e.g., 50, 100, 200).
- `max_depth`: Maximum tree depth.
- `min_samples_split`: Minimum samples required to split a node.
- `criterion`: Split quality metric (e.g., `'mae'` for regression).
- `random_state`: Ensures reproducibility.

## Example Configuration

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
```

## See Also
- [Decision Tree](decision-tree.md)
- [Gradient Boosting](gradient-boosting.md)

## Sources
- [Kaggle and ML Tutorial](../summaries/kaggle.md)

## Related
- [Decision Tree](decision-tree.md)
- [Gradient Boosting](gradient-boosting.md)
- [Overfitting/Underfitting](overfitting-underfitting.md)
- [Scikit-learn](scikit-learn.md)
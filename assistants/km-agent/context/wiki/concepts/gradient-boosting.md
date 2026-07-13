---
title: "Gradient Boosting"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/kaggle.md]
related: [random-forest, decision-tree, xgboost]
tags: [gradient-boosting, ensemble, ml, xgboost]
---

# Gradient Boosting

Gradient boosting is an ensemble method that iteratively adds models to a sequence, where each new model corrects errors made by the combined predictions of previous models. It builds an ensemble by going through cycles, adding one model at a time.

## XGBoost

**XGBoost** (Extreme Gradient Boosting) is a high-performance implementation of gradient boosting optimized for speed and accuracy.

### Key Parameters

- `n_estimators`: Number of models in the ensemble (e.g., 1000).
- `learning_rate`: Multiplier applied to each model's predictions before adding them; smaller values require more trees but can improve accuracy.
- `early_stopping_rounds`: Stops training when validation score stops improving for N rounds.
- `n_jobs`: Number of parallel cores to use.

### Example

```python
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(X_train, y_train,
             early_stopping_rounds=5,
             eval_set=[(X_valid, y_valid)])
```

## See Also
- [Random Forest](random-forest.md)
- [Decision Tree](decision-tree.md)

## Sources
- [Kaggle and ML Tutorial](../summaries/kaggle.md)

## Related
- [Random Forest](random-forest.md)
- [Decision Tree](decision-tree.md)
- [Scikit-learn](scikit-learn.md)
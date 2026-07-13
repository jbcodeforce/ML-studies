---
title: "Missing Value Imputation"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/data/features.md]
related: [feature-engineering, categorical-encoding]
tags: [machine-learning, data-preprocessing, sklearn]
---

# Missing Value Imputation

Missing values are common in real-world datasets and must be handled before training machine learning models.

## Strategies

### Dropping

- **Drop rows** with missing target values is standard practice.
- **Drop columns** with many missing values, especially if the column contributes little to predictions. Avoid dropping columns with only a few missing values.

```python
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
```

### Imputation

**Mean/median imputation** replaces missing values with the column's mean or median. Scikit-learn's `SimpleImputer` handles this via the transformer pattern (`fit` on training data, `transform` on validation/test data).

```python
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer(strategy='median')
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
```

### Missing Indicators

Adding a **boolean indicator column** that flags whether a value was imputed can meaningfully improve results, because the absence of data may itself carry signal.

```python
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
```

## Key Considerations

- Always `fit` imputers on training data only, then `transform` both training and validation sets.
- Imputation preserves the column count, unlike dropping.
- The indicator column approach captures information about data quality that imputation alone discards.

## Sources
- [Feature Engineering](../summaries/features.md)

## Related
- [Feature Engineering](feature-engineering.md)
- [Categorical Encoding](categorical-encoding.md)
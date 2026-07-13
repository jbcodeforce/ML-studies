---
title: "Categorical Encoding"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/data/features.md]
related: [feature-engineering, missing-value-imputation]
tags: [machine-learning, data-preprocessing, sklearn, encoding]
---

# Categorical Encoding

Categorical variables take a limited set of values and must be converted to numeric form before most ML algorithms can process them.

## Ordinal Encoding

**Ordinal encoding** assigns each unique category a distinct integer. This is appropriate for categories with a natural ordering (e.g., small/medium/large). Scikit-learn's `OrdinalEncoder` handles this.

```python
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
```

**Handling unseen categories**: If the test set contains categories not in the training set, either write a custom encoder or drop the problematic columns:

```python
good_label_cols = [col for col in object_cols if
                   set(X_valid[col]).issubset(set(X_train[col]))]
bad_label_cols = list(set(object_cols) - set(good_label_cols))
```

## One-Hot Encoding

**One-hot encoding** creates a new binary column for each category value, indicating presence (1) or absence (0). It makes no assumptions about ordering.

```python
from sklearn.preprocessing import OneHotEncoder
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
```

### Cardinality Considerations

- One-hot encoding can dramatically expand dataset size.
- Apply one-hot encoding only to **low-cardinality columns** (e.g., fewer than 10 unique values).
- For high-cardinality columns, consider dropping or using ordinal encoding.

### Pandas Shortcut

Pandas provides `get_dummies()` for quick one-hot encoding, with alignment to handle mismatched columns between train/valid/test sets:

```python
X_train = pd.get_dummies(X_train)
X_train, X_valid = X_train.align(X_valid, join='left', axis=1)
```

## Sources
- [Feature Engineering](../summaries/features.md)

## Related
- [Feature Engineering](feature-engineering.md)
- [Missing Value Imputation](missing-value-imputation.md)
---
title: "Group Transforms"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/data/features.md]
related: [feature-engineering, feature-creation]
tags: [machine-learning, data-preprocessing, pandas]
---

# Group Transforms

**Group transforms** aggregate information across multiple rows grouped by a categorical feature, creating new features that capture context beyond individual rows.

## Concept

A group transform combines two features: a **categorical grouping feature** and a **numerical feature to aggregate**. Common aggregation functions include `mean`, `max`, `min`, `median`, `var`, `std`, and `count`.

### Examples

- "Average income of a person's state of residence"
- "Proportion of movies released on a weekday, by genre"
- "Number of roadway features per accident"

## Implementation

Pandas `groupby` + `transform` is the standard approach:

```python
customer["AverageIncome"] = (
    customer.groupby("State")
    ["Income"]
    .transform("mean")
)
```

### Preserving Train/Validation Independence

When using train/valid splits, create grouped features **using only training data**, then merge into the validation set. This prevents data leakage:

```python
# Create grouped feature on training set only
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge into validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)
```

## Related Feature Creation Techniques

- **Aggregating binary features**: Summing boolean columns to count types of something present.
- **String splitting**: Extracting category from structured strings (e.g., splitting `"One_Story_1946_and_Newer_All_Styles"` to get `"One_Story"`).

## Sources
- [Feature Engineering](../summaries/features.md)

## Related
- [Feature Engineering](feature-engineering.md)
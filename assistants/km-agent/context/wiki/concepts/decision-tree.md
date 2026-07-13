---
title: "Decision Tree"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/kaggle.md]
related: [random-forest, overfitting-underfitting, scikit-learn]
tags: [decision-tree, ml, classification, regression, overfitting]
---

# Decision Tree

A decision tree is a supervised learning model that splits data into branches based on feature values, creating a tree structure of decision nodes and leaf nodes. Each leaf holds a prediction (class label for classification, numeric value for regression).

## Key Properties

- **Depth control**: Trees are configured via `max_leaf_nodes` or `max_depth`. Deeper trees with more leaves can capture fine-grained patterns but risk **overfitting** (each prediction comes from only a few training records). Shallower trees generalize better but may underfit.
- **Implementation**: Available in scikit-learn as `DecisionTreeClassifier` and `DecisionTreeRegressor`.
- **Base for ensembles**: Single decision trees serve as building blocks for Random Forest and Gradient Boosting methods.

## Parameters
- `max_leaf_nodes`: Limits the number of terminal nodes; controls model complexity.
- `random_state`: Ensures reproducible splits.

## See Also
- [Overfitting/Underfitting](overfitting-underfitting.md)
- [Random Forest](random-forest.md)

## Sources
- [Kaggle and ML Tutorial](../summaries/kaggle.md)

## Related
- [Random Forest](random-forest.md)
- [Overfitting/Underfitting](overfitting-underfitting.md)
- [Scikit-learn](scikit-learn.md)
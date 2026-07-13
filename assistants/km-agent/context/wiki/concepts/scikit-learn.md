---
title: "Scikit-learn Library"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/coding/sklearn.md]
related: [pytorch-library, pandas, distributed-data-parallel]
tags: [scikit-learn, ml, classification, perceptron, pipeline, feature-scaling, python]
---

# Scikit-learn Library

Scikit-learn (sklearn) is a widely-used open-source Python machine learning library providing a comprehensive suite of classifier algorithms, preprocessing utilities, and model evaluation tools. It follows a consistent API design pattern centered around estimators, transformers, and pipelines.

## Five-Step ML Workflow

Scikit-learn supports a structured approach to machine learning:

1. **Feature selection** — Identify and select relevant input features from raw data
2. **Performance metric selection** — Choose appropriate evaluation criteria for the problem
3. **Classifier and optimization algorithm selection** — Match the algorithm to the data and problem constraints
4. **Parameter tuning** — Optimize hyperparameters for best performance
5. **Performance evaluation** — Assess model accuracy on unseen data

## Core Components

### Data Loading
Scikit-learn includes built-in datasets (e.g., `load_iris()`) for experimentation. Data is represented as NumPy arrays: `X` for features with shape `(n_samples, n_features)` and `y` for target values.

### Data Splitting
The `train_test_split` function from `sklearn.model_selection` randomly partitions datasets into training and test subsets (e.g., 70/30 split), enabling evaluation on unseen data.

### Feature Scaling
Many algorithms require scaled input features. `StandardScaler` from `sklearn.preprocessing` standardizes features by computing mean (μ) and standard deviation (δ) on training data, then applying the same transformation to test data:

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
```

### Classifiers
Scikit-learn offers a large variety of classifiers:

- **Perceptron** (`sklearn.linear_model.Perceptron`): A basic linear classifier. Main limitation is that it never converges if classes are not perfectly linearly separable.
- **LogisticRegression**: A probabilistic linear classifier suitable for binary and multiclass classification.

### Pipelines
Transformers and estimators can be combined into a `Pipeline`, unifying preprocessing and modeling into a single object:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)
predictions = pipe.predict(X_test)
```

### Evaluation
The `sklearn.metrics` module provides a wide variety of performance metrics including `accuracy_score`, precision, recall, and more.

## Installation

```sh
pip3 install pandas scikit-learn
```

## Sources
- [Scikit-learn Library Notes](../summaries/sklearn.md)

## Related
- [PyTorch Library](pytorch-library.md)
- [Pandas](pandas.md)
- [Distributed Data Parallel (DDP)](distributed-data-parallel.md)
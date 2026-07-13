---
title: "Support Vector Machines"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/classifier.md]
related: [classification, normalization, regularization]
tags: [svm, support-vector-machines, classification, margin, kernel, rbf]
---

# Support Vector Machines

**Support Vector Machines (SVM)** classify data by finding a decision boundary that **maximizes the margin** — the distance between the boundary and the nearest training samples (support vectors).

## Rationale

Large-margin boundaries tend to have **lower generalization error**, while small-margin models are more prone to overfitting.

## Kernels

- **Linear Kernel**: Creates a straight-line decision boundary in the original feature space.
- **RBF (Radial Basis Function) Kernel**: Projects data into a higher-dimensional space via a mapping function φ() where it becomes linearly separable, enabling non-linear decision boundaries.

### Gamma Parameter

In RBF kernels, **gamma** is a cutoff parameter for the Gaussian sphere:
- Higher gamma → larger influence/reach of training samples → softer decision boundary
- Lower gamma → tighter decision boundary
- Optimizing gamma is important to avoid overfitting

## Implementation

```python
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.3, random_state=0
)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
```

For non-linear boundaries:

```python
svm = SVC(kernel='rbf', C=10.0, random_state=0, gamma=0.10)
```

## Sources
- [Classifiers](../summaries/classifier.md)

## Related
- [Classification](classification.md)
- [Normalization](normalization.md)
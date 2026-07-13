---
title: "Scikit-learn Library"
source: ML-studies
ingested: 2026-07-12
tags: [scikit-learn, ml, classification, perceptron, pipeline, python]
type: summary
---

# Scikit-learn Library Summary

## Main Thesis
Scikit-learn is a comprehensive Python machine learning library providing classifier algorithms and utilities. Choosing the right algorithm requires understanding each method's assumptions, as classifier performance depends heavily on the underlying data.

## Five-Step ML Workflow
The document outlines five main steps in training a machine learning model:
1. **Feature selection** — identifying relevant input features
2. **Performance metric selection** — choosing appropriate evaluation criteria
3. **Classifier and optimization algorithm selection** — matching algorithm to problem
4. **Parameter tuning** — optimizing hyperparameters
5. **Performance evaluation** — assessing model on unseen data

## Key Concepts Covered
- **Data loading**: Built-in datasets like IRIS with `sklearn.datasets`
- **Data splitting**: `train_test_split` for separating training/test sets (e.g., 70/30 split)
- **Feature scaling**: `StandardScaler` for standardizing features using mean and standard deviation
- **Perceptron**: A basic linear classifier (`sklearn.linear_model.Perceptron`) with configurable iterations and learning rate; disadvantage is non-convergence on non-linearly separable data
- **Pipelines**: `make_pipeline` combines transformers and estimators into a single unified object, streamlining preprocessing and modeling
- **Evaluation**: `accuracy_score` and other metrics from `sklearn.metrics`

## Code Example
A complete workflow is demonstrated using the IRIS dataset: load data, select features, split into train/test, scale features, train a Perceptron or LogisticRegression in a pipeline, and evaluate accuracy.

## Connections
- Relates to [PyTorch](wiki/concepts/pytorch-library.md) as another Python ML library
- Connects to [Pandas](wiki/concepts/pandas.md) for data manipulation
- Supports classifier concepts discussed in ML classification studies
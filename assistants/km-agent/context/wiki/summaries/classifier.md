---
title: "Classifiers"
source: ML-studies
ingested: 2026-07-12T21:51:16Z
tags: [ml, classification, supervised-learning]
---

# Classifiers Summary

This document provides an overview of classification as a supervised learning task — predicting one of a small number of discrete-valued outputs. It covers the three major components of any classifier: **representation** (rules, decision trees, neural networks), **evaluation** (confusion matrices, accuracy, recall), and **optimization** (greedy search, gradient descent).

The source walks through multiple classifier algorithms using the Iris dataset (4 features, 3 classes: setosa, versicolor, virginica):

- **Perceptron**: A single-neuron binary classifier based on Frank Rosenblatt's model, using a unit-step activation function. Converges only for linearly separable data with small learning rates.
- **Adaline (Adaptive Linear Neuron)**: Like the perceptron but uses a linear (identity) activation function, enabling continuous weight updates. Benefits from feature standardization and stochastic gradient descent for large datasets.
- **Logistic Regression**: Maps linear combinations of features through a sigmoid function to produce class probabilities. Uses log-odds formulation and includes regularization (L2) to control overfitting via the C parameter.
- **Support Vector Machines (SVM)**: Maximizes the margin between classes. Works well with linear kernels and can handle non-linearly separable data via RBF kernels with gamma tuning.
- **Decision Trees**: Build hierarchical splits based on information gain (Gini impurity, entropy, or classification error). Pruning (max depth) prevents overfitting.
- **Random Forests**: Ensemble of decision trees that reduce overfitting and improve generalization. Parameter tuning involves number of trees and max depth.
- **K-Nearest Neighbors (KNN)**: A nonparametric, instance-based, lazy-learning classifier that memorizes training data and predicts by majority vote among K closest neighbors. Susceptible to the curse of dimensionality.

All Python implementations execute in Docker-based environments. See linked code in the ML-studies repository.
# Machine Learning Overview

Machine learning is a system that automatically learns programs and functions from data without explicit programming. The goal is to discover a function that predicts **y** from features **X**, continuously measuring prediction performance.

## Key Distinctions
- **Statistics** applies models of the world (linear regression, logistic regression, Cox model) to data.
- **Machine learning** discovers functions from data through algorithms.

## Two Primary Categories

### Supervised Learning
Learns from labeled training data to predict unseen or future data:
- **Classification**: Predicting discrete-valued outputs (e.g., weather classes: Sunny/Cloudy/Rainy). Can be binary or multi-class.
- **Regression**: Predicting continuous-valued outcomes (e.g., house prices from features like square footage, bedrooms).

### Unsupervised Learning
Explores data structure without guidance from known outcome variables:
- **Clustering**: Organizes data into meaningful subgroups without prior knowledge of group memberships.

### Reinforcement Learning
Develops an agent that improves through environment interaction, using trial-and-error or deliberative planning to maximize rewards.

## ML System Components
1. **Data preprocessing** — crucial step; raw data rarely comes in the right shape
2. **Model training** — fitting algorithms to training data
3. **Validation** — evaluating using train/test splits, LOOCV, or k-fold cross-validation
4. **Feature scaling** — transforming features to same scale for optimal performance
5. **Dimensionality reduction** — compressing features to lower-dimensional subspace
6. **Model selection** — comparing multiple algorithms using performance metrics like classification accuracy
7. **Experiment tracking** — assessing and tracking ML experiments (TensorBoard, MLFlow, Weights & Biases)

## Model Representation & Cost Functions
- Hypothesis function: `h(x) = θ^T * x` (multivariate linear regression)
- Cost function: Mean squared error `J(θ) = 1/(2m) * Σ(h(x_i) - y_i)²`
- **Gradient descent**: Iteratively minimizes cost by stepping opposite the gradient, scaled by learning rate (alpha/eta)
- Feature scaling essential for efficient gradient descent convergence
- Batch gradient descent computes weight updates from entire training set

## Experiment Tracking Tools
- Python dictionaries (simple metadata tracking)
- [TensorBoard](https://www.tensorflow.org/tensorboard)
- [MLFlow](https://mlflow.org/)
- [Weights & Biases](https://wandb.ai/site/experiment-tracking)

## Sources
- [ML Index](../summaries/index.md)
- [Classifier Notes](../summaries/classifier.md)
- [Unsupervised Notes](../summaries/unsupervised.md)
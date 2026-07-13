---
title: "Regularization"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/deep-learning.md]
related: [deep-learning, bias-variance, data-augmentation]
code: []
tags: [regularization, overfitting, l1, l2, dropout, batch-normalization, deep-learning]
---

# Regularization

Regularization techniques help prevent overfitting by adding constraints to the model during training. Overfitting occurs when training loss is much lower than test loss, meaning the model has memorized the training data rather than learning generalizable patterns.

## L1 Regularization (Lasso)

Adds the absolute value of weights to the loss function, producing sparse models by driving some weights to zero:

```
Loss_L1 = Loss + λ Σᵢ |wᵢ|
```

L1 regularization performs feature selection automatically by eliminating less important features.

## L2 Regularization (Ridge / Weight Decay)

Adds the squared magnitude of weights to the loss function, penalizing large weights without forcing them to zero:

```
Loss_L2 = Loss + λ Σᵢ wᵢ²
```

In PyTorch, L2 regularization is implemented via the `weight_decay` parameter in optimizers:

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

## Dropout

Randomly sets a fraction of input units to zero during training, preventing neurons from co-adapting. Typical dropout rates are 0.2–0.5:

```python
nn.Dropout(p=0.5)  # 50% of neurons dropped
```

Dropout forces remaining connections to learn robust features that compensate for the removed neurons.

## Batch Normalization

Normalizes layer inputs to have zero mean and unit variance. Benefits include:

- Stabilizes training by reducing internal covariate shift
- Allows higher learning rates
- Provides slight regularization effect

```python
nn.BatchNorm2d(num_features)  # For conv layers
nn.BatchNorm1d(num_features)  # For linear layers
```

## Data Augmentation

Artificially increases training set diversity by applying random transformations to input data. This improves generalization without collecting more data:

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Data augmentation is applied **only to training data**, not validation or test sets.

## Sources
- [Deep Learning](../summaries/deep-learning.md)

## Related
- [Deep Learning](deep-learning.md)
- [Bias-Variance](bias-variance.md)
- [Transfer Learning](transfer-learning.md)
---
title: "Transfer Learning"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/deep-learning.md]
related: [deep-learning, convolutional-neural-networks]
code: [code/computer-vision/transfer_learning.py]
tags: [transfer-learning, pre-trained-models, fine-tuning, deep-learning]
---

# Transfer Learning

Transfer learning involves taking an existing pre-trained model and using it on your own data to fine-tune the parameters. It helps achieve better results with less data, at lower cost and time.

## How It Works

The typical transfer learning process:

1. **Load a pre-trained model** with its trained weights (e.g., EfficientNet trained on ImageNet)
2. **Prepare custom data** using the same transforms as the original training data
3. **Freeze base layers** (typically the feature extraction layers) to preserve learned representations
4. **Replace the classifier head** with new layers matching your task's output classes
5. **Fine-tune** the new layers on your custom dataset

### Example: EfficientNet Transfer Learning

```python
# Load pre-trained weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
transformer = weights.transforms()

# Load the model
model = torchvision.models.efficientnet_b0(weights=weights).to(device)

# Freeze the feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace the classifier for custom classes
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True),
    torch.nn.Linear(in_features=1280, out_features=len(classes), bias=True)
).to(device)
```

## Model Architecture

Pre-trained models like EfficientNet have three main parts:

- **Features**: Convolutional layers and activations that learn base visual representations
- **AvgPool**: Averages the feature layer output into a feature vector
- **Classifier**: Maps the feature vector to output class dimensions

## Available Pre-trained Models

- **PyTorch**: [torchvision.models](https://pytorch.org/vision/stable/models.html)
- **Hugging Face**: [huggingface.co/models](https://huggingface.co/models)
- **Timm**: [PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) — large collection of image models
- **Papers with Code**: [paperswithcode.com/sota](https://paperswithcode.com/sota) — state-of-the-art results with implementations

## Benefits

- **Less data needed**: Pre-trained features generalize well to new domains
- **Faster training**: Only fine-tuning a few layers vs. training from scratch
- **Lower cost**: Reduced compute requirements
- **Better accuracy**: Pre-trained models on large datasets (e.g., ImageNet with millions of images) provide strong starting points

## Sources
- [Deep Learning](../summaries/deep-learning.md)

## Related
- [Deep Learning](deep-learning.md)
- [Convolutional Neural Networks](convolutional-neural-networks.md)
- [Regularization](regularization.md)
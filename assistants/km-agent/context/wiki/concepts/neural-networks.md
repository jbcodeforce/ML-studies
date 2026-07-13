---
title: "Neural Networks"
created: 2026-07-12
updated: 2026-07-12
sources: [raw/studies/ml/deep-learning.md]
related: [deep-learning, convolutional-neural-networks, recurrent-neural-networks]
code: [code/regression/neuralnetwork/, code/deep-learning/fundamentals/]
tags: [neural-networks, neurons, activation-functions, deep-learning]
---

# Neural Networks

A neural network is a programming approach based on the biologically-inspired neuron, used to teach a computer from training data instead of programming it with structured code.

## Architecture

The basic structure of a neural network includes:

- **Input layer** (feature vector): Where data is fed into the model. Each input neuron maps to one element in the feature vector.
- **Hidden layers**: Perform computational processing. Each layer receives all outputs from the previous layer. Hidden neurons allow the network to abstract and transform inputs.
- **Output layer**: Generates the final result. Each output neuron calculates one part of the output.
- **Bias neurons**: Work like the y-intercept of a linear equation, introducing a constant 1 as input to shift activation thresholds.

Neurons are also called nodes, units, or summations.

## Neuron Computation

A neuron computes a weighted sum of its inputs passed through an activation function:

```
neuron(j) = σ(Σᵢ ωᵢ × aᵢ - bias)
```

Where:
- `ωᵢ` is the weight of the connection from neuron i to neuron j
- `aᵢ` is the activation (output) of neuron i from the previous layer
- `σ` is the activation function
- `bias` defines when the neuron should fire

## Activation Functions

| Function | Formula | Use Case |
|---|---|---|
| Sigmoid | σ(x) = 1 / (1 + e⁻ˣ) | Binary classification output (legacy) |
| ReLU | max(0, x) | Hidden layers (modern default) |
| Softmax | Normalized exponential | Multi-class classification output |
| Tanh | Hyperbolic tangent | Hidden layers (less common now) |
| Linear | f(x) = x | Regression output |

**Why ReLU over Sigmoid?** ReLU avoids the gradient saturation problem — sigmoid's derivative quickly approaches zero as inputs move away from zero, hindering gradient descent. ReLU maintains non-zero gradients for positive inputs.

## Training

Training is the process that determines optimal weight values. It involves:

1. Forward pass: computing predictions through the network
2. Loss computation: comparing predictions to targets
3. Backward pass: computing gradients via backpropagation
4. Weight update: adjusting weights using an optimizer (SGD, Adam, etc.)

## Types of Neural Networks

- **Feedforward networks**: Standard multi-layer perceptrons (MLPs)
- **Convolutional Neural Networks (CNNs)**: Specialized for image data
- **Recurrent Neural Networks (RNNs)**: Designed for sequential data

## Sources
- [Deep Learning](../summaries/deep-learning.md)

## Related
- [Deep Learning](deep-learning.md)
- [Convolutional Neural Networks](convolutional-neural-networks.md)
- [Recurrent Neural Networks](recurrent-neural-networks.md)
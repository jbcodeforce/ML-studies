# Deep learning

It is a machine learning techniques which uses neural networks. 

## Neural Network

A Neural network processes numerical representation of unstructured data, injected in an 'input layer' (called "feature vector") to generate result as part of the output layer by using a mathematical construct of network of hidden layers (See YouTube video: ["Neural Network the ground up"](https://www.youtube.com/watch?v=aircAruvnKk)). "Deep" learning means many hidden layers.

A classical learning example of neural network usage, is to classify images, like the hand written digits of the NIST dataset.

A neuron holds a function that returns a number between 0 and 1. For example in simple image classification, neuron may hold the grey value of a pixel of a 28x28 pixels image (784 neurons). The number is called **activation**. At the output layer, the number in the neuron represents the percent of one output being the expected response. Neurons are connected and each connection is weighted.

Convolutional neural networks (CNNs) allows input size to change without retraining. The output of the regression neural network is numeric, and the classification output is a class.

The value of the neuron 'j' in the next layer is computed by the classical logistic equation taking into account previous layer neurons (`a`) (from 1 to n (i being the index on the number of input)) and the weight of the connection (`a(i)` to `neuron(j)`):

![](https://latex.codecogs.com/svg.latex?neuron(j)=\sigma (\sum_{i} \omega_{i} * a_{i} - bias)){ width=300 }

To get the activation between 0 and 1, it uses the [sigmoid function](../concepts/maths.md#sigmoid-function), the bias is a number to define when the neuron should be active.

![](./images/basic-math-neuron-net.png)

Modern neural network does not use sigmoid function anymore but the [Rectifier Linear unit function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

![](https://latex.codecogs.com/svg.image?ReLu(a)=max(0,a))

Neural networks input and output can be an image, a series of numbers that could represent text, audio, or a time series...

The simplest architecture is the [perceptron](./classifier/#perceptron), represented by the following diagram:

![](./images/perceptron.png)

There are four types of neurons in a neural network:

1. Input Neurons - We map each input neuron to one element in the feature vector.
1. Hidden Neurons - Hidden neurons allow the neural network to be abstract and process the input into the output. Each layer receives all the output of previous layer.
1. Output Neurons - Each output neuron calculates one part of the output.
1. Bias Neurons - Work similar to the y-intercept of a linear equation. It introduces a 1 as input.

Neurons is also named nodes, units or summations. See [the sigmoid play notebook to understand the effect of bias and weights](https://github.com/jbcodeforce/ML-studies/tree/master/deep-neural-net/sigmoid-play.ipynb) 

Training refers to the process that determines good weight values.

It is possible to use different Activation functions,(or transfer functions), such as hyperbolic tangent, sigmoid/logistic, linear activation function, Rectified Linear Unit (ReLU), Softmax (used for the output of classification neural networks), Linear (used for the output of regression neural networks (or 2-class classification)).

ReLU activation function is popular in deep learning because the gradiant descend function needs to take the derivative of the activation function. With sigmoid function, the derivative quickly saturates to zero as 
 moves from zero, which is not the case for ReLU.

The two most used Python frameworks for deep learning are [TensorFlow/Keras](https://www.tensorflow.org/) (Google) or [PyTorch](../coding/pytorch.md) (Facebook).

## Classification neural network architecture

The general architecture of a classification neural network.

| Hyperparameter | Classification | 
| --- | --- | 
| Input layer shape (in_features) |	Same as number of features |
| Hidden layer(s) | Problem specific, minimum = 1, maximum = unlimited |
| Neurons per hidden layer| Problem specific, generally 10 to 512 |
| Output layer shape (out_features)| for binary 1 class, for multi-class: 1 per class |
| Hidden layer activation |	Usually ReLU but can be many others |
| Output activation | For binary: Sigmoid, for multi-class:	Softmax |
| Loss function	| Binary cross entropy.  For multi-class Cross entropy |
| Optimizer | SGD (stochastic gradient descent), Adam (see torch.optim for more options) |

Below is an example of very simple NN in PyTorch

```python
from torch import nn

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),  # layer 1
    nn.Linear(in_features=5, out_features=1)   # layer 2
).to(device)

model_0
```

Or use a subclass of `nn.Module` as demonstrated in [classifier.ipynb](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/classifier.ipynb) notebook, to search for the circle classes in sklearn circles dataset, or a multi classes classification in [multiclass-classifier.ipynb]

## Learning

Same as previous ML problems, we can use supervised ( picture and corresponding class) and unsupervised learning. For image or voice, the 'self-supervised learning' uses to generate supervisory signals for training data sets by looking at the relationships in the input data.

Transfer learning is used to get what a first neural network as learn as input to a second NN. 

## Computer Image

### Convolutional Neural Network

A [Neural Network](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) to process images by assigning learnable weights and biases to various aspects/objects in the image, and be able to differentiate one from the other
It can successfully capture the spatial and temporal dependencies in an image through the application of relevant filters.
Image has three matrices of values matching the size of the picture (H*W) and the RGB value.
CNN reduces the size of the matrices without loosing the meaning. For that it uses the concept of Kernel, a window, shifting over the image.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ciDgQEjViWLnCbmX-EeSrA.gif)

See example of dataset [Fashion NIST](https://github.com/zalandoresearch/fashion-mnist)

## Sources of information

* Big source of online book [Dive into Deep Learning from Amazoniens](https://d2l.ai).
* [Udemy PyTorch for deep learning](https://www.learnpytorch.io/)


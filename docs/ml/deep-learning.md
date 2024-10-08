# Deep learning

Deep learning is a machine learning techniques which uses neural networks with more than one layer.

## Neural Network

A Neural Network is a programming approach, based on the biological inspired neuron,  used to teach a computer from training data instead of programming it with structured code.

The basic structure of a neural network includes an input layer (called "feature vector"), where the data is fed into the model, hidden layers that perform the computational processing, and an output layer that generates the final result. (See YouTube video: ["Neural Network the ground up"](https://www.youtube.com/watch?v=aircAruvnKk)).

A classical learning example of neural network usage, is to classify images, like the hand written digits of the NIST dataset ([shallow_net_demo.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/neuralnetwork/shallow_net_demo.ipynb)).

A simple neuron holds a function that returns a number between 0 and 1. For example in simple image classification, neuron may hold the grey value of a pixel of a 28x28 pixels image (784 neurons). The number is called **activation**. At the output layer, the number in the neuron represents the percent of one output being the expected response. Neurons are connected together and each connection is weighted.

Convolutional neural networks (CNNs) ([lenet_in_keras.ipynb](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/neuralnetwork/lenet_in_keras.ipynb)) allows input size to change without retraining. For the grey digit classification, the CNN defines a neuron as a unique image pattern of 3x3. The output of the regression neural network is numeric, and the classification output is a class.

The value of the neuron 'j' in the next layer is computed by the classical logistic equation taking into account previous layer neurons (`a`) (from 1 to n (i being the index on the number of input)) and the weight of the connection (`a(i)` to `neuron(j)`):

![](https://latex.codecogs.com/svg.latex?neuron(j)=\sigma (\sum_{i} \omega_{i} * a_{i} - bias)){ width=300 }

To get the activation between 0 and 1, it uses the [sigmoid function](../concepts/maths.md#sigmoid-function), the bias is a number to define when the neuron should be active.

![](./images/basic-math-neuron-net.png)

Modern neural network does not use sigmoid function anymore but the [Rectifier Linear unit function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

![](https://latex.codecogs.com/svg.image?ReLu(a)=max(0,a))

Neural networks input and output can be an image, a series of numbers that could represent text, audio, or a time series...

The simplest architecture is the [perceptron](./classifier.md#perceptron), represented by the following diagram:

![](./images/perceptron.png)

There are four types of neurons in a neural network:

1. Input Neurons - We map each input neuron to one element in the feature vector.
1. Hidden Neurons - Hidden neurons allow the neural network to be abstract and process the input into the output. Each layer receives all the output of previous layer.
1. Output Neurons - Each output neuron calculates one part of the output.
1. Bias Neurons - Work similar to the y-intercept of a linear equation. It introduces a 1 as input.

Neurons is also named nodes, units or summations. See [the sigmoid play notebook to understand the effect of bias and weights](https://github.com/jbcodeforce/ML-studies/tree/master/deep-neural-net/sigmoid-play.ipynb) 

Training refers to the process that determines good weight values.

It is possible to use different Activation functions,(or transfer functions), such as hyperbolic tangent, sigmoid/logistic, linear activation function, Rectified Linear Unit (ReLU), Softmax (used for the output of classification neural networks), Linear (used for the output of regression neural networks (or 2-class classification)).

ReLU activation function is popular in deep learning because the gradiant descend function needs to take the derivative of the activation function. With sigmoid function, the derivative quickly saturates to zero as it moves from zero, which is not the case for ReLU.

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

Below is an example of a very simple NN in PyTorch, without any activation function:

```python
from torch import nn

model_0 = nn.Sequential(
    nn.Linear(in_features=2, out_features=5),  # layer 1
    nn.Linear(in_features=5, out_features=1)   # layer 2
).to(device)

model_0
```

We can use a subclass of pyTorch `nn.Module` to define the NN. See demonstration in [classifier.ipynb](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/classification/classifier.ipynb) notebook, to search for the circle classes in the sklearn circles dataset, or a multi classes classification in [multiclass-classifier.ipynb](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/classification/multiclass-classifier.ipynb).

## Learning

Same as previous ML problems, we can use supervised ( picture and corresponding classes) and unsupervised learning. For image or voice, the 'self-supervised learning' uses to generate supervisory signals for training data sets by looking at the relationships in the input data.

Transfer learning combine a first neural network as input to a second NN. 

### GPU vs CPU

1. When the training loss is way lower than the test loss, it means "overfitting" and so loosing time.
1. When both losses are identical, time will be wasted if we try to regularize the model.
1. To optimize deep learning we need to maximize the compute-bound processing by reducing time spent on memory transfer and other things. Bandwidth cost is by moving the data from CPU to GPU, from one node to another, or even from CUDA global memory to CUDA shared memory.

## Computer Image

Address how a computer sees, images.

### Convolutional Neural Network

A [Neural Network](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) to process images by assigning learnable weights and biases to various aspects/objects in the image, and be able to differentiate one from the other. It can successfully capture the spatial and temporal dependencies in an image through the application of relevant filters.

Image has three matrices of values matching the size of the picture (H*W) and the RGB value R matrix, G and B matrices.
CNN reduces the size of the matrices without loosing the meaning. For that, it uses the concept of Kernel, a window, shifting over the image.

![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*ciDgQEjViWLnCbmX-EeSrA.gif)

A typical structure of a convolutional neural network:

**Input layer -> [Convolutional layer -> activation layer -> pooling layer] -> Output layer**

The layers between [] can be replicated.

Every layer in a neural network is trying to compress data from higher dimensional space to lower dimensional space. Below is an example of this method:

```python
# Convolutional layer
nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
nn.ReLU(),  # activation layer
# pooling layer
nn.MaxPool2d(kernel_size=2, stride=2),    
```

* Conv2d is compressing the information stored in the image to a smaller dimension image
* MaxPool2d takes the maximum value from a portion of a tensor and disregard the rest.

See this [CNN explainer tool.](https://poloclub.github.io/cnn-explainer/)

Simple image dataset using the [Fashion NIST](https://github.com/zalandoresearch/fashion-mnist).

The non-linear classifier and one CNN is in [fashion_cnn.py](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/computer-vision/fashion_cnn.py).

[MIT - Convolutional Neural Network presentation - video](https://www.youtube.com/watch?v=iaSUYvmCekI&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=4)

## Transfer Learning

Take an existing pre-trained model, and use it on our own data to fine tune the parameters. It helps to get better results with less data, and lesser cost and time. In Computer Vision, [Image Net](https://www.image-net.org) includes million of images on which models were trained. 

[PyTorch has pre-trained models](https://pytorch.org/vision/stable/models.html), [Hugging Face](https://huggingface.co/models) too, [PyTorch Image Models - Timm](https://github.com/huggingface/pytorch-image-models) is a collection of image models, layers, utilities, optimizers, schedulers, data-loaders / augmentations, and reference training / validation scripts that can be reused. [Paper with code](https://paperswithcode.com/sota) is a collection of the latest state-of-the-art machine learning papers with code implementations attached to the article.

The custom data going into the model needs to be prepared in the same way as the original training data that went into the model: 

```python
# load existing NN weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
# Get the transforms used to create our pretrained weights
transformer= weights.transforms()
```

The transformer is used to create the data loaders:

```python
train_dl,test_dl, classes=data_setup.create_data_loaders(
                            train_dir,
                            test_dir,
                            transformer,
                            transformer,
                            batch_size=BATCH_SIZE)
```

Then, take an existing model. Often bigger models are better but results may also being linked to the type of device used and the hardware resource capacity. `efficientnet_b0` has 288,548 parameters.

```
model=torchvision.models.efficientnet_b0(weights=weights).to(device)
```

???- info "efficientnet_b0 parts"
    `efficientnet_b0` comes in three main parts:
    
    * **features**: A collection of convolutional layers and other various activation layers to learn a base representation of vision data.
    * **avgpool**: Takes the average of the output of the features layer(s) and turns it into a feature vector.
    * **classifier**: Turns the feature vector into a vector with the same dimensionality as the number of required output classes (since efficientnet_b0 is pretrained on ImageNet with 1000 classes.

The process of transfer learning usually goes: freeze some base layers of a pre-trained model (typically the features section) and then adjust the output layers (also called head/classifier layers) to suit the needs.

```python
for param in model.features.parameters():  # Freeze the features
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=len(classes), 
                        bias=True)).to(device)
```

Dropout layers randomly remove connections between two neural network layers with a probability of p.  This practice is meant to help regularize (prevent overfitting) a model by making sure the connections that remain learn features to compensate for the removal of the other connections.

See [PyTorch transfer learning for image classification code.](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/computer-vision/transfer_learning.py)

## Sources of information

* Big source of online book [Dive into Deep Learning from Amazons](https://d2l.ai).
* [Udemy PyTorch for deep learning](https://www.learnpytorch.io/)
* [Horace He- Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html)
* [MIT - CNConvolutional Neural Network presentation - video](https://www.youtube.com/watch?v=iaSUYvmCekI&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=4)



# Deep learning 

It is a machine learning techniques which uses neural networks. A Neural network processes numerical representation of unstructured data, injected in an 'input layer' to generate result as part of the output layer by using a mathematical construct of network of hidden layers (See YouTube video: ["Neural Network the ground up"](https://www.youtube.com/watch?v=aircAruvnKk)). A classical learning example of neural network, is to recognize the hand written digits, using the NIST dataset.

A neuron holds a function that returns a number between 0 and 1. It holds the grey value of a pixel of a 28x28 pixels image (784 neurons). The number is called **activation**. At the output layer, the number in the neuron represents the percent of this output being the expected response. Neurons are connected and each connection is weighted.

The value of the neuron 'j' in the next layer is computed by the classical logistic equation taking into account previous layer neurons (`a`) (from 1 to n (i being the index)) and the weight of the connection (`a(i)` to `neuron(j)`):

![](https://latex.codecogs.com/svg.latex?neuron(j)=\sigma (\sum_{i} \omega_{i} * a_{i} - bias)){ width=300 }

To get the activation between 0 and 1, it uses the [sigmoid function](../concepts/maths.md#sigmoid-function), the bias is a number to define when the neuron should be active.

![](./images/basic-math-neuron-net.png)

A deep neural network is nothing more than a neural network with many layers.

Modern neural network does not use sigmoid function anymore but the [Rectifier Linear unit function](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).

![](https://latex.codecogs.com/svg.image?ReLu(a)=max(0,a))

Neural networks input and output can be an image, a series of numbers that could represent text, audio, or another time series...

The two most Python frameworks for deep learning are [TensorFlow/Keras](https://www.tensorflow.org/) (Google) or [PyTorch](../coding/pytorch.md) (Facebook).

## Learning

Same as previous ML problems, we can use supervised ( picture and corresponding class) and unsupervised learning. For image or voice, the 'self-supervised learning' uses to generate supervisory signals for training data sets by looking at the relationships in the input data.

Transfer learning is used to get what a first neural network as learn as input to a second NN. 

## Sources of information

* Big source of online book [Dive into Deep Learning from Amazoniens](https://d2l.ai).
* [Udemy PyTorch for deep learning]()

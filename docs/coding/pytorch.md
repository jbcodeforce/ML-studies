# [Pytorch library](https://pytorch.org/)

The [most popular](https://paperswithcode.com/trends) Python ML and deep learning library to implement ML workflow and deep learning solution. It is open-source project. It helps to run code on GPU/TPU. PyTorch is also a low-level math library as NumPy, but built for deep learning. It compiles these compute graphs into highly efficient C++/CUDA code.

The sources for this content is from product documentation, [Zero to mastery - learning pytorch](https://www.learnpytorch.io/), and [WashU training](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_03_2_pytorch.ipynb) website.

## Environment setup

Use mini conda for package management and virtual environment management, and jupyter notebooks.

### Install

* Using Python 3 and pip3, install torch

    ```sh
    pip3 install torch torchvision torchaudio
    ```

* Using Anaconda:

    1. Install miniconda (it is installed in ~/miniconda3): 

        ```sh
        # under ~/bin
        curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
        sh Miniconda3-latest-MacOSX-arm64.sh -u
        ```

    1. Verify installed libraries: `conda list`
    1. Environments are created under `~/miniconda3/envs`. To create a conda environment named "torch", in miniconda3 folder do: `conda create anaconda python=3 -n torch`
    1. To activate conda environment: `conda activate torch`
    1. Install pytorch `conda install pandas pytorch::pytorch torchvision torchaudio -c pytorch`
    1. [optional] Install jupyter packaging: `conda install -y jupyter`
    1. Register a new runtime env for jupyter: `python -m ipykernel install --user --name pytorch --display-name "Python 3.9 (pytorch)"`

### Run once conda installed

1. To activate conda environment: `conda activate torch`
1. Test my first program: `python basic-torch.py ` 
1. If we need Jupyter: `jupyter notebook` in the torch env, and then [http://localhost:8888/tree](http://localhost:8888/tree).
1. Select the Kernel to be "Python 3.9 (pytorch)"

My code studies are in [pytorch](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch) folder.

## Concepts

### Tensor

Tensor is an important concept for deep learning. It is the numerical representation of data, a n dimension matrix.

Tensors are a specialized data structure, similar to NumPy’s `ndarrays`, except that tensors can run on GPUs. Tensor attributes describe their shape, datatype, and the device on which they are stored.

```python
import torch, numpy as np
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
```

Tensor created from NumPy array:

```python
import torch
X= torch.from_numpy(X).type(torch.float)
y= torch.from_numpy(y).type(torch.float)
X[:5],y[:5]
```

Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other. [See the set of basic operations on tensor](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/torch-tensor-basic.ipynb).

### Constructs

PyTorch has two important modules we can use to create neural network: `torch.nn, torch.optim`, and two primitives to work with data: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset. See [dataset examples](https://pytorch.org/text/stable/datasets.html)

* See basic code in [basic-torch.py](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/basic-torch.py) with inline explanations.

| Modules | Description |
| --- | --- |
| `torch.nn` | 	Contains all of the building blocks for computational graphs. |
| `torch.nn.Parameter` | Stores tensors that can be used with nn.Module. If requires_grad=True gradients descent are calculated automatically. |
| `torch.nn.Module` | The base class for all neural network modules. Need to subclass it. Requires a forward() method be implemented. |
| `torch.optim` |  various optimization algorithms to tell the model parameters how to best change to improve gradient descent and in turn reduce the loss |

### GPU

On Linux need the Cuda (Compute Unified Device Architecture) library for Nvidia GPU. See [AWS deep learning container](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/what-is-dlc.html).

Here is sample code to set the mps to access GPU (mps for Mac) for tensor computation

```python
has_mps = torch.backends.mps.is_built()
device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
```

NumPy uses only CPU, so we can move to tensor and then tensor.to(device) to move the tensor to GPU, do computation and move back to NumPy

```python
tensor=torch.tensor([1,2,3])
tensor_on_gpu = tensor.to(device)
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
```

See [Tim Dettmers has guide](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/).

### Basic Algebra with Pytorch

See [Algebra using Pytorch python code.](https://github.com/jbcodeforce/ML-studies/blob/master/pytorch/get_started/AlgebraPyTorch.py)

### [Loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

[Cost / loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) selection depends on the problem to solve. 

| Loss function/Optimizer |	Problem type | PyTorch module |
| --- | --- | --- |
| **Stochastic Gradient Descent** (SGD) optimizer |	Classification, regression, many others. |	torch.optim.SGD() |
| **Adam Optimizer** | Classification, regression, many others. | torch.optim.Adam() |
| **Binary cross entropy loss** | Binary classification	| torch.nn. BCELossWithLogits or torch.nn.BCELoss | 
| **Cross entropy loss** |	Mutli-class classification | torch.nn.CrossEntropyLoss |
| **Mean absolute error** (MAE) or L1 Loss | Regression | torch.nn.L1Loss |
| **Mean squared error** (MSE) or L2 Loss | Regression| torch.nn.MSELoss|

The [binary cross-entropy / log loss](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a) is used to compute how good are the predicted probabilities. The function uses a  negative log probability for a label to be one of the expected class: {0,1}, so when a class is not 1 the loss function result is big.

### Neural network

A [PyTorch neural network](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) declaration is a class that extends `nn.Module`. The constructor includes the neural network structure, and the class must implement the `forward(x)` function to pass the input to the network and get the output. This is the most flexible way [to declare a NN](https://github.com/jbcodeforce/ML-studies/blob/4271cbd2fa3094cf672e038ee7559997e9d90443/pytorch/classification/nn-classifier.py#L17). As an alternate the following code uses the Sequential method using non linear layers (nn.ReLu()).

```python
model = nn.Sequential(
    nn.Linear(x.shape[1], 50),
    nn.ReLU(),
    nn.Linear(50, 25),
    nn.ReLU(),
    nn.Linear(25, 1)
).to(device)
```

Neural network has an input layer equal to the number of input features, and output equal to the number of response (1 output for binary classification). For activation function between hidden layers, ReLU is often used when we want non-linearity. The output layer will not use a transfer function for a regression neural network, or use the logistic for binary classification (just two classes) or log softmax for two or more classes.

The hyper-parameters to tune are:

* The number of neuron in hidden layer: In general, more hidden neurons means more capability to fit complex problems. But too many, will lead to overfitting. Too few, may lead to underfitting the problem and will sacrifice accuracy.

* The number of layers: more layers allow the neural network to perform more of its feature engineering and data preprocessing.
* The activation function between hidden layers and for the output layer.
* The loss and optimizer functions.
* The learning rate of the optimization functions
* Number of epochs to train the model. An epoch as one complete pass over the training set.

For multi class training, `LogSoftmax` is used as transfer function and `CrossEntropyLoss` as loss function.
With Softmax, the outputs are normalized probabilities that sum up to one.

Some code samples:

* Basic NN in dual class [classifier notebook](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/classification/classifications.ipynb)
* [Multi-class classifier notebook](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/classification/multiclass-classifier.ipynb)
* Python code for a PyTorch neural network for a binary classification on (Sklearn moons dataset) using Loss : [nn-classifier.py](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/classification/nn-classifier.py).
* Computer vision and the CNN [A notebook](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/computer-vision/computer_vision.ipynb) and [Python code](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/computer-vision/fashion_cnn.py)

```Output
{'model_name': 'FashionMNISTModel', 'model_loss': 0.41334256529808044, 'model_acc': tensor(0.8498, device='mps:0')}
{'model_name': 'FashionNISTCNN', 'model_loss': 0.3709910213947296, 'model_acc': tensor(0.8716, device='mps:0')}
```

## Model training

### PyTorch training loop

For the training loop, the steps to build:

| Step	| What does it do? | Code example |
| --- | --- | --- |
| Forward pass | The model goes through all of the training data once, performing its forward() function calculations.	| model(x_train) |
| Calculate the loss | The model's predictions are compared to the ground truth and evaluated to see how wrong they are. | loss = loss_fn(y_pred, y_train) |
| Zero gradients | The optimizers gradients are set to zero to be recalculated for the specific training step. | optimizer.zero_grad() |
| Perform back propagation on the loss | Computes the gradient of the loss with respect for every model parameter to be updated (each parameter with requires_grad=True) | loss.backward() |
| Update the optimizer (gradient descent)| Update the parameters with requires_grad=True with respect to the loss gradients in order to improve them. | optimizer.step() |

Example of code for training on multiple epochs:

```python
loss_fn=nn.CrossEntropyLoss()
optimizer= torch.optim.SGD(params=model.parameters(), lr=0.1)

for epoch in range(epochs):
    model.train()
    # 1. Forward pass
    y_logits = model(X_train).squeeze()
    # from logits -> prediction probabilities -> prediction labels
    y_pred = torch.softmax(y_logits,dim=1).argmax(dim=1)
    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```

The rules for performing inference with PyTorch models:

```python
model.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    y_preds = model(X_test)
```


### PyTorch testing loop

The typical steps include:

| Step | Description| 	Code example |
| --- | --- | --- |
| Forward pass	| The model goes through all of the test data |	model(x_test) |
| Calculate the loss | The model's predictions are compared to the ground truth. | loss = loss_fn(y_pred, y_test) |
| Calculate evaluation metrics | Calculate other evaluation metrics such as accuracy on the test set. | Custom function |

```python
model.eval()
with torch.inference_mode():
    # 1. Forward pass
    test_logits = model(X_test).squeeze() 
    test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
    # 2. Caculate loss/accuracy
    test_loss = loss_fn(test_logits,y_test)
    test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

```

### Improving a model

| Model improvement technique | What does it do? |
| --- | --- |
| **Add more layers**	| Each layer potentially increases the learning capabilities of the model with each layer being able to learn some kind of new pattern in the data, more layers is often referred to as making the neural network deeper. |
| **Add more hidden units**	| Similar to the above, more hidden units per layer means a potential increase in learning capabilities of the model, more hidden units is often referred to as making the neural network wider.|
| **Fitting for longer (more epochs)** | The model might learn more if it had more opportunities to look at the data.  |
| **Changing the activation functions** | Some data just can't be fit with only straight lines, using non-linear activation functions can help. |
| **Change the learning rate** |	Less model specific, the learning rate of the optimizer decides how much a model should change its parameters each step, too much and the model over corrects, too little and it doesn't learn enough. |
| **Change the loss function** | Different problems require different loss functions. |
| **Use transfer learning** |	Take a pre-trained model from a problem domain similar to ours and adjust it to our own problem. |

### Evaluate classification models

Classification model can be measured using the at least the following metrics (see more [PyTorch metrics](https://lightning.ai/docs/torchmetrics/stable/)):

| Metric name/Evaluation method	| Definition | Code |
| --- | --- | --- |
| **Accuracy**	| Out of 100 predictions, how many does your model get correct? E.g. 95% accuracy means it gets 95/100 predictions correct. |	torchmetrics.Accuracy() or sklearn.metrics.accuracy_score() |
| **Precision**	| Proportion of true positives over total number of samples. Higher precision leads to less false positives (model predicts 1 when it should've been 0). | torchmetrics.Precision() or sklearn.metrics.precision_score() |
| **Recall** | Proportion of true positives over total number of true positives and false negatives (model predicts 0 when it should've been 1). Higher recall leads to less false negatives. |	torchmetrics.Recall() or sklearn.metrics.recall_score()|
| **F1-score** | Combines precision and recall into one metric. 1 is best, 0 is worst. | torchmetrics.F1Score() or sklearn.metrics.f1_score() |
| **Confusion matrix** | Compares the predicted values with the true values in a tabular way, if 100% correct, all values in the matrix will be top left to bottom right.| [torchmetrics.classification.ConfusionMatrix]()https://lightning.ai/docs/torchmetrics/stable/classification/confusion_matrix.html or sklearn.metrics.plot_confusion_matrix() |
| **Classification report** | Collection of some of the main classification metrics such as precision, recall and f1-score. | [sklearn.metrics.classification_report()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) |


## Pytorch datasets

PyTorch includes many existing functions to load in various custom datasets in the [TorchVision](https://pytorch.org/vision/stable/index.html), [TorchText](https://pytorch.org/text/stable/index.html), [TorchAudio](https://pytorch.org/audio/stable/index.html) and [TorchRec](https://pytorch.org/torchrec/) domain libraries.

### Data augmentation

Data augmentation is the process of altering the data in such a way that this artificially increases the diversity of the training set.

The purpose of [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) is to alter the images in some way and turning them into a tensor, or cropping an image or randomly erasing a portion or randomly rotating them.

Training a model on this artificially altered dataset hopefully results in a model that is capable of better **generalization** (the patterns it learns are more robust to future unseen examples).

Researches show that random transforms (like `transforms.RandAugment()` and `transforms.TrivialAugmentWide()`) generally perform better than hand-picked transforms.

We usually don't perform data augmentation on the test set. The idea of data augmentation is to artificially increase the diversity of the training set to better predict on the testing set.

See also in [PyTorch's Illustration of Transforms](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html) examples.



## Some How to

???- question "How to set the device dynamically"
    ```python
    def getDevice():
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.backends.cuda.is_available():
            device = torch.device("cuda")
        else: 
            device = torch.device("cpu")
        return device   
    ```

???- question "How to save and load a model?"
    ```python
    # saving using Pytorch
    MODEL_SAVE_PATH = MODEL_PATH / filename
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    # Load is reusing the class declaration
    model=FashionNISTCNN(input_shape=1,hidden_units=10,output_shape=10)
    model.load_state_dict(torch.load("models/fashion_cnn_model.pth"))
    ```

???- question "Display the confusion matrix for a multiclass prediction"
    ```python
    def make_confusion_matrix(pred_tensor, test_labels, class_names):
        # Present a confustion matrix between the predicted labels and the true labels from test data
        cm = MulticlassConfusionMatrix(num_classes=len(class_names))
        cm.update(pred_tensor, test_labels)
        fig,ax = cm.plot(labels=class_names)
        plt.show()
    ```

???- question "Transform an image into a Tensor"
    Use [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) module
    ```python
    train_transformer=v2.Compose([v2.Resize((224,224)), v2.TrivialAugmentWide(num_magnitude_bins=31), v2.ToTensor()])
    ```

???- question "How to get visibility into a neural network"
    ```
    import torchinfo
    torchinfo.summary()
    ```

## Code samples

* [Basic operations on tensor: my own notebook](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/get_started/torch-tensor-basic.ipynb) and [Learn Pytorch introduction](https://www.learnpytorch.io/00_pytorch_fundamentals/#introduction-to-tensors).
* [Pytorch workflow for training and testing model](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/get_started/workflow-basic.ipynb)
* [Compute image classification on Fashion NIST images in pythons](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/computer_vision/fashion_cnn.py) and [use_fashion_cnn.pn](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/computer_vision/use_fashion_cnn.py)

## Resources

* [Udemy content repo](https://github.com/mrdbourke/pytorch-deep-learning)
* [Zero to mastery - learning pytorch](https://www.learnpytorch.io/)
* [The incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch):  curated list of tutorials, projects, libraries, videos, papers, books..
* [Dan Fleisch's video: What's a tensor?](https://www.youtube.com/watch?v=f5liqUk0ZTw)
* [How to train state of the art models using Torchvision - PyTorch blog.](https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives)
* [Jeff Heaton - Using Convolutional Neural Networks.](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_05_2_cnn.ipynb)
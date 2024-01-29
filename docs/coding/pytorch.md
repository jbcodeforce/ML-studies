# [Pytorch library](https://pytorch.org/)

The [most popular](https://paperswithcode.com/trends) Python ML and deep learning library to implement ML workflow and deep learning solution. It is open-source project. It helps to run code on GPU/TPU.

The sources for this content is from product documentation, [Zero to mastery - learning pytorch](https://www.learnpytorch.io/), and [WashU training](https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_03_2_pytorch.ipynb) website.

## Environment setup

Use mini conda, and jupyter notebooks:

### Install

* Using Python 3 and pip:

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

[Cost / loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions) that work well are: the SGD (stochastic gradient descent) or Adam optimizer; the MAE (mean absolute error) for regression problems (predicting a number) or binary cross entropy loss function for classification problems (predicting one thing or another).

### GPU

On Linux need the Cuda (Compute Unified Device Architecture) library for Nvidia GPU. See [AWS deep learning container](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/what-is-dlc.html).

Here is sample code to set the mps to access GPU for tensor computation

```python
if torch.backends.mps.is_available():
   mps_device = torch.device("mps")
   x = torch.ones(1, device=mps_device)
```

NumPy uses only CPU, so we can move to tensor and then tensor.to(device) to move the tensor to GPU, do computation and move back to NumPy

```python
tensor=torch.tensor([1,2,3])
tensor_on_gpu = tensor.to(device)
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
```

See [Tim Dettmers has guide](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/).

## PyTorch training loop

For the training loop, the steps to build:

| Step	| What does it do? | Code example |
| --- | --- | --- |
| Forward pass | The model goes through all of the training data once, performing its forward() function calculations.	| model(x_train) |
| Calculate the loss | The model's predictions are compared to the ground truth and evaluated to see how wrong they are. | loss = loss_fn(y_pred, y_train) |
| Zero gradients | The optimizers gradients are set to zero to be recalculated for the specific training step. | optimizer.zero_grad() |
| Perform back propagation on the loss | Computes the gradient of the loss with respect for every model parameter to be updated (each parameter with requires_grad=True) | loss.backward() |
| Update the optimizer (gradient descent)| Update the parameters with requires_grad=True with respect to the loss gradients in order to improve them. | optimizer.step() |

The rules for performing inference with PyTorch models:

```python
model_0.eval()

# 2. Use the inference mode context manager to make predictions
with torch.inference_mode():
    y_preds = model_0(X_test)
```


### PyTorch testing loop

The typical steps include:

| Step | Description| 	Code example |
| --- | --- | --- |
| Forward pass	| The model goes through all of the test data |	model(x_test) |
| Calculate the loss | The model's predictions are compared to the ground truth. | loss = loss_fn(y_pred, y_test) |
| Calculate evaluation metrics | Calculate other evaluation metrics such as accuracy on the test set. | Custom function |

## Code samples

* [Basic operations on tensor: my own notebook](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/torch-tensor-basic.ipynb) and [Learn Pytorch introduction](https://www.learnpytorch.io/00_pytorch_fundamentals/#introduction-to-tensors).
* [Pytorch workflow for training and testing model](https://github.com/jbcodeforce/ML-studies/tree/master/pytorch/workflow-basic.ipynb)

## Resources

* [Udemy content repo](https://github.com/mrdbourke/pytorch-deep-learning)
* [Zero to mastery - learning pytorch](https://www.learnpytorch.io/)
* [The incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch):  curated list of tutorials, projects, libraries, videos, papers, books..
* [Dan Fleisch's video: What's a tensor?](https://www.youtube.com/watch?v=f5liqUk0ZTw)

# [Pytorch library](https://pytorch.org/)

The [most popular](https://paperswithcode.com/trends) Python ML and deep learning library to implement ML workflow and deep learning solution. It is open-source project. It helps to run code on GPU/TPU.

See [Zero to mastery - learning pytorch](https://www.learnpytorch.io/) website.

## Concepts

### Tensor

Tensor is an important concept for deep learning. It is the numerical representation of data. 

Tensors are a specialized data structure, similar to NumPy’s `ndarrays`, except that tensors can run on GPUs. Tensor attributes describe their shape, datatype, and the device on which they are stored. 

```python
import torch, numpy as np
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
```

Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.

### Constructs

PyTorch has two primitives to work with data: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset. See [dataset examples](https://pytorch.org/text/stable/datasets.html)

* See basic code in [basic-torch.py](https://github.com/jbcodeforce/ML-studies/tree/master/deep-neural-net/basic-torch.py) with inline explanations.

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


## Code samples

## Resources

* [Udemy content repo](https://github.com/mrdbourke/pytorch-deep-learning)
* [Zero to mastery - learning pytorch](https://www.learnpytorch.io/)
* [The incredible pytorch](https://github.com/ritchieng/the-incredible-pytorch):  curated list of tutorials, projects, libraries, videos, papers, books..
* [Dan Fleisch's video: What's a tensor?](https://www.youtube.com/watch?v=f5liqUk0ZTw)

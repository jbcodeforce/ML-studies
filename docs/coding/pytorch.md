# [Pytorch library](https://pytorch.org/)

A Python ML library to implement ML workflow and deep learning solution.

PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset. See [dataset examples](https://pytorch.org/text/stable/datasets.html)

See basic code in [basic-torch.py](https://github.com/jbcodeforce/ML-studies/tree/master/deep-neural-net/basic-torch.py) with inline explanations.

Tensors are a specialized data structure, similar to NumPy’s `ndarrays`, except that tensors can run on GPUs. Tensor attributes describe their shape, datatype, and the device on which they are stored. 

```python
import torch, numpy as np
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
```

Tensors on the CPU and NumPy arrays can share their underlying memory locations, and changing one will change the other.


## Resources

* [Udemy content repo](https://github.com/mrdbourke/pytorch-deep-learning)

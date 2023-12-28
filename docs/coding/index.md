# Coding

???- info "Update"
    05/02/2023 Move to python 3.10 in docker, retest docker env with all code. See [samples section](#code-samples) below.

    09/10/2023: Add PyTorch

    12/2023: Clean Jupyter

## Environments

To avoid impacting the laptop (Mac) python core installation, use virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Then in each main folder there is a `requirements.txt` to get the necessary modules.

There are a lot of other solutions we can use, like the Amazon [scikit-learn image](https://raw.githubusercontent.com/aws/sagemaker-scikit-learn-container/master/docker/1.2-1/base/Dockerfile.cpu). The SageMaker team uses this repository to build its official [Scikit-learn image](https://github.com/aws/sagemaker-scikit-learn-container).  we can build an image via:

```sh
docker build -t sklearn-base:1.2-1 -f https://raw.githubusercontent.com/aws/sagemaker-scikit-learn-container/master/docker/1.2-1/base/Dockerfile.cpu .
```

### VSCode

#### [Jupyter Notebook](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

* To select an environment, use the `Python: Select Interpreter` command from the Command Palette (⇧⌘P)
* Use `Create: New Jupyter Notebook` from command Palette
* Select a kernel using the kernel picker in the top right.
* Within a Python Notebook, it's possible to view, inspect, sort, and filter the variables within the current Jupyter session, using `Variables` in toolbar.
* We can offload intensive computation in a Jupyter Notebook to other computers by connecting to a remote Jupyter server. Use server URL with security token.

### Run Kaggle image

As an alternate Kaggle has a more complete [docker image](https://github.com/Kaggle/docker-python) to start with. 

```sh
# CPU based
docker run --rm -v $(pwd):/home -it gcr.io/kaggle-images/python /bin/bash
# GPU based
docker run -v $(pwd):/home --runtime nvidia --rm -it gcr.io/kaggle-gpu-images/python /bin/bash
```

### Conda

Conda provides package, dependency, and environment management for any language. 

On Mac M1 we need ARM64 architecture. 

1. Install miniconda: [projects/miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
1. To create a conda environment named "torch", in miniconda3 folder do: `conda env create -f torch-conda-nightly.yml -n torch`
1. Activate conda environment: 

    ```sh
    conda activate torch
    ```

1. Register environment: `python -m ipykernel install --user --name pytorch --display-name "Python 3.9 (pytorch)"`
1. Install the following: `conda install pytorch pandas scikit-learn`
1. Start Jupiter: `jupiter notebook`
1. Execute the notebook in to test [test-env.ipynb](https://github.com/jbcodeforce/ML-studies/tree/master/deep-neural-net/test-env.ipynb)

### Run Jupyter notebooks

* We can use jupyter lab (see [installation options](https://jupyter.org/install.html)) or conda and miniconda.

    ```sh
    conda install -y jupyter
    ```
### JupyterLab

The following works as of April 2023:

```sh
pip3 install jupyterlab
# build the assets
jupyter-lab build
# The path is something like
# /opt/homebrew/Cellar/python@3.10/3.10.9/Frameworks/Python.framework/Versions/3.10/share/jupyter/lab
# Start the server
jupyter-lab
```

Once started, in VScode select a remote Python kernel and Jupiter extension to run the notebook inside it. 

## Important Python Libraries

### [numpy](https://numpy.org/)

* Array computing in Python. [Getting started](https://numpy.org/devdocs/user/quickstart.html)
* NumPy dimensions are called axes.

```python
import numpy as np
a = np.array([2, 3, 4])
b = np.ones((2, 3, 4), dtype=np.int16)
c = np.zeros((3, 4))
```
* Create a sequence of number: `np.arange(10, 30, 5)`
* Matrix product: using .dot or @

    ```python
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[2, 0], [3, 4]])
    A @ B
    A.dot(B)
    ```

### [scipy](https://scipy.org/)

SciPy is a collection of mathematical algorithms and convenience functions built on NumPy .

* Get a normal distribution function: use the probability density function (pdf)

```python
from scipy.stats import norm
x = np.arange(-3, 3, 0.01)
y=norm.pdf(x)
```

### [MatPlotLib](https://matplotlib.org/stable/users/index.html)

Persent figure among multiple axes, from our data.


* Classic import

```
import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl
```

* See [Notebook](https://github.com/jbcodeforce/ML-studies/blob/master/notebooks/MatPlotLib.ipynb)

### [Seaborn](https://seaborn.pydata.org/)

Seaborn provides a high-level interface for drawing attractive and informative statistical graphics. Based on top of MatPlotLib and integrated with Pandas.

[See the introduction for different examples](https://seaborn.pydata.org/tutorial/introduction.html)


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
sns.relplot(
    data=masses_data,
    x="age", y="shape", 
    hue="density", size="density"
)
plt.show()
```

### PyTorch

Via conda or pip, install `pytorch torchvision torchaudio`.

Example of getting started code in deep-neural-net folder. 

[Summary of the library](./pytorch.md)

## Code samples

| Link | Description |
| --- | --- |
| [Perceptron](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestPerceptron.py) |  To classify the iris flowers. Use identity activation function |
| [Adaline](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestAdaline.py) | ADAptive LInear NEuron with weights updated based on a linear activation function |
| [Fischer](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestBatteryClassifier.py) | Fisher classification for sentences |
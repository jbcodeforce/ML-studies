# Coding

???+ "Update"
    05/02/2023 Move to python 3.10 in docker, retest docker env with all code. See [samples section](#code-samples) below.

    09/10/2023: Add PyTorch

## Environments

To avoid impacting my laptop (Mac) python installation, use virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Then in each main folder there is a `requirements.txt` to get the necessary modules.


There are a lot of other solutions we can use, like the Amazon [scikit-learn image](https://raw.githubusercontent.com/aws/sagemaker-scikit-learn-container/master/docker/1.2-1/base/Dockerfile.cpu). The SageMaker team uses this repository to build its official [Scikit-learn image](https://github.com/aws/sagemaker-scikit-learn-container).  we can build an image via:

```sh
docker build -t sklearn-base:1.2-1 -f https://raw.githubusercontent.com/aws/sagemaker-scikit-learn-container/master/docker/1.2-1/base/Dockerfile.cpu .
```


### Run Kaggle image

As an alternate Kaggle has a more complete [docker image](https://github.com/Kaggle/docker-python) to start with. 

```sh
# CPU based
docker run --rm -v $(pwd):/home -it gcr.io/kaggle-images/python /bin/bash
# GPU based
docker run -v $(pwd):/home --runtime nvidia --rm -it gcr.io/kaggle-gpu-images/python /bin/bash
```

## Run Jupyter notebooks

We can use jupyter lab (see [installation options](https://jupyter.org/install.html)) or conda and miniconda.

### Conda

On Mac M1 we need ARM64 architecture. 

1. Install miniconda: https://docs.conda.io/projects/miniconda/en/latest/
1. Install conda with the jupyter packaging: `conda install -y jupyter`
1. To create a conda environment named "torch", in miniconda3 folder do: `conda env create -f torch-conda-nightly.yml -n torch`
1. Activate conda environment: `conda activate torch`
1. Register environment: `python -m ipykernel install --user --name pytorch --display-name "Python 3.9 (pytorch)"`
1. Install the following: `conda install pytorch pandas scikit-learn`
1. Start Jupiter: `jupiter notebook`
1. Execute the notebook in to test [test-env.ipynb](https://github.com/jbcodeforce/ML-studies/tree/master/deep-neural-net/test-env.ipynb)

### JupyterLab

The ones which works as of April 2023:

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

## PyTorch

Via conda or pip, install `pytorch torchvision torchaudio`.

Example of getting started code in deep-neural-net folder. 

[Summary of the library](./pytorch.md)

## Code samples

| Link | Description |
| --- | --- |
| [Perceptron](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestPerceptron.py) |  To classify the iris flowers. Use identity activation function |
| [Adaline](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestAdaline.py) | ADAptive LInear NEuron with weights updated based on a linear activation function |
| [Fischer](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestBatteryClassifier.py) | Fisher classification for sentences |
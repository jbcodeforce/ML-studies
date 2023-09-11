# Deep neural network studies

## Preparing Mac M1/M2 with GPU for deep learning

See [WashU Jeff Heaton Applications of Deep Neural Networks](https://github.com/jeffheaton/t81_558_deep_learning) course.

On Mac M1 we need ARM64 architecture. 

1. Install miniconda: https://docs.conda.io/projects/miniconda/en/latest/
1. Install conda with the jupyter packaging: `conda install -y jupyter`
1. To create a conda environment named "torch", in miniconda3 folder do: `conda env create -f torch-conda-nightly.yml -n torch`
1. Activate conda environment: `conda activate torch`
1. Register environment: `python -m ipykernel install --user --name pytorch --display-name "Python 3.9 (pytorch)"`
1. Install the following: `conda install pytorch pandas scikit-learn`
1. Start Jupiter: `jupiter notebook`
1. Execute the notebook in to test [test-env.ipynb](https://github.com/jbcodeforce/ML-studies/tree/master/deep-neural-net/test-env.ipynb)


## Playing with PyTorch

Add some specifics libraries:  conda install torchvision

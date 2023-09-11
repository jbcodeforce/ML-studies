# Coding

???+ "Update"
    05/02/2023 Move to python 3.10 in docker, retest docker env with all code. See [samples section](#code-samples) below.

    09/10/2023: Add PyTorch

## Environments

To avoid impacting my laptop (Mac) python installation, we can use docker image with python and the minimum set of needed libraries. 
The dockerfile is in this folder is used to build a development image.

### Run my python development shell

As some of the python codes are using matplotlib and graphics, it is possible to use the MAC display 
with docker and X11 display (see [this blog](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc)) for details, but it can be summarized as.

* By default XQuartz will listen on a UNIX socket, which is private and only exists on local our filesystem. Install `socat` to then create a two bidirectional streams between Xquartz and a client endpoints: `brew install socat`.
* Install XQuartz with `brew install xquartz`. Then start Xquartz from the application or using: `open -a Xquartz`. A white terminal window will pop up.  
* The first time Xquartz is started, within the X11 Preferences window, select the Security tab and ensure the `allow connections from network clients` is ticked.
* Set the display using the ipaddress pod the host en0 interface: `ifconfig en0 | grep "inet " | awk '{print $2}` 
* Dry test the X server works from a docker client: `docker run -e DISPLAY=$(ifconfig en0 | grep "inet " | awk '{print $2}'):0 gns3/xeyes`
* Build environment images: `docker build -t jbcodeforce/python .`


Then run the following command to open a two bi-directional streams between the docker container and the X window system of Xquartz.

```shell
source ./setDisplay.sh
# If needed install Socket Cat: socat with:  brew install socat
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
```

which is what the `socatStart.sh` script does.

Start a docker container with an active bash shell with the command:

```shell
./startPythonDocker.sh
```

Then navigate to the python code from the current `/app` folder, and call python

```
$ python deep-net-keras.py
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
| [Perceptron](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestPerceptron.py) |  To classify of iris flowers image. Use identity activation function |
| [Adaline](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestAdaline.py) | ADAptive LInear NEuron with weights updated based on a linear activation function |
| [Fischer](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/Test) | Fisher classification for sentences |
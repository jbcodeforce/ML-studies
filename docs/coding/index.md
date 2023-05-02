# Coding

???+ "Update"
    05/02/2023 Move to python 3.10 in docker, retest docker env with all code. See [samples section](#code-samples) below.

## Environments

To avoid impacting my laptop (Mac) python installation, I use docker image with python and the minimum set of needed libraries. 
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

Need to use jupyter lab. See [installation options](https://jupyter.org/install.html). The ones which works as o April 2023:

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

The mkdocs in this project has a `jupyter-notebook` plugin so notebooks can be html page in the published site. 

## Code samples

| Link | Description |
| --- | --- |
| [Perceptron](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestPerceptron.py) |  To classify of iris flowers image. Use identity activation function |
| [Adaline](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/TestAdaline.py) | ADAptive LInear NEuron with weights updated based on a linear activation function |
| [Fischer](https://github.com/jbcodeforce/ML-studies/blob/master/ml-python/classifiers/Test) | Fisher classification for sentences |
# Coding

## Environment

To avoid impacting my laptop (Mac) python installation, I use docker image with python and all the minimum needed library. 
The dockerfile is in this folder is used to build a development image.

As an alternate Kaggle has a more complete [docker image](https://github.com/Kaggle/docker-python) to start with. 

```sh
# CPU based
docker run --rm -v $(pwd):/home -it gcr.io/kaggle-images/python /bin/bash
# GPU based
docker run -v $(pwd):/home --runtime nvidia --rm -it gcr.io/kaggle-gpu-images/python /bin/bash
```

As some of the python codes are using matplotlib and graphics, it is possible to use the MAC display 
with docker and X11 display (see [this blog](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc)) for details.

* Install XQuartz with `brew install xquartz`. Then start Xquartz from the application or using: `open -a Xquartz`. A white terminal window will pop up. The first time Xquartz is started, open up the preferences menu and go to the security tab. Then select “allow connections from network clients” to check it on.
* Build environment images: `docker build -t jbcodeforce/python37 .`

### Run my python development shell

When using graphic: start **Xquartz** from (applications/utilities), be sure the security settings allow connections from network clients.

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

Then navigate to the python code from the current `/home` folder, and call python

```
$ python deep-net-keras.py
```
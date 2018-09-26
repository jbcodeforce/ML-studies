# Machine Learning Studies - Basic
This repository includes different code to study how to do machine learning with Python and jupyter notebook.

It is based from different sources like 'Machine learning with Python' book;
collective intelligence book; Standford Machine learning trainng, Intel ML 101, and internet articles.

To avoid impact my laptop (Mac) python installation I use docker image with python and all the needed library. The dockerfile in this folder can help create an image.

As some of the python codes are using matplotlib and graphics, it is possible to use the MAC display with docker and X11 display (see [this blog](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc)).

## Run
When using graphic: start Xquartz from (applications/utilities), be sure the security settings allow connections from network clients.

Then the command to open two bi-directional streams between the docker container and the X window system of Xquartz.

```
$ socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
which is in the socatStart.sh script.
```

Verify the ip address of the host: `ifconfig en0`. Modify if needed the startPython.sh DISPLAY variable.

Start a docker container with an active bash shell with the command:
```
docker run -v $(pwd):/home/jovyan/work -it pystacktf /bin/bash
```
Then navigate to the python code from the work folder, you want to execute and call python
```
$ python3 deep-net-keras.py
```

### Classifiers
#### perceptron
```
python3 TestPerceptron.py
```

#### Adaline

In ADALINE the weights are updated based on a linear activation function
(the identity function) rather than a unit step function like in the perceptron.
```
python3 TestAdaline.py
```

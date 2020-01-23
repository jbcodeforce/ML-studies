# Machine learning in Python studies

This repository includes different code to help me learn how to do machine learning with Python and jupyter notebook.

It is based from different sources like:

* [Python Machine learning - Sebastian Raschka's book](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130/ref=asc_df_1783555130/?tag=hyprod-20&linkCode=df0&hvadid=312140868236&hvpos=1o7&hvnetw=g&hvrand=12056535591325453294&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9032152&hvtargid=pla-406163981473&psc=1)
* [Collective intelligence - Toby Segaran's book](https://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325/ref=sr_1_2?crid=1UBVCJKMM17Q6&keywords=collective+intelligence&qid=1553021611&s=books&sprefix=collective+inte%2Cstripbooks%2C236&sr=1-2)
* [Stanford Machine learning training - Andrew Ng](https://www.coursera.org/learn/machine-learning)
* Intel ML 101 tutorial

## Environment

To avoid impact to my laptop (Mac) python installation, I use docker image with python and all the needed library. The dockerfile is in this folder is used to build a development image.

As some of the python codes are using matplotlib and graphics, it is possible to use the MAC display with docker and X11 display (see [this blog](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc)) for details.

* Install XQuartz with `brew install xquartz`. Then start Xquartz from the application or using: `open -a Xquartz`. A white terminal window will pop up. The first time Xquartz is started, open up the preferences menu and go to the security tab. Then select “allow connections from network clients” to check it on.
* Build environment images: `docker build -t jbcodeforce/python37 .`

### Run python development shell

When using graphic: start **Xquartz** from (applications/utilities), be sure the security settings allow connections from network clients.

Then run the following command to open a two bi-directional streams between the docker container and the X window system of Xquartz.

```shell
$ source ./setDisplay.sh
$ socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
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

## Code

### Classifiers

#### Perceptron

```
python TestPerceptron.py
```

#### Adaline

In ADALINE the weights are updated based on a linear activation function
(the identity function) rather than a unit step function like in the perceptron.

```
python TestAdaline.py
```

### Anomaly detection



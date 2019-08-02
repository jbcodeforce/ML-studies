# Machine Learning Studies - Basic
This repository includes different code to help me learn how to do machine learning with Python and jupyter notebook.

It is based from different sources like [Python Machine learning - Sebastian Raschka](https://www.amazon.com/Python-Machine-Learning-Sebastian-Raschka/dp/1783555130/ref=asc_df_1783555130/?tag=hyprod-20&linkCode=df0&hvadid=312140868236&hvpos=1o7&hvnetw=g&hvrand=12056535591325453294&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9032152&hvtargid=pla-406163981473&psc=1) book;
[Collective intelligence - Toby Segaran](https://www.amazon.com/Programming-Collective-Intelligence-Building-Applications/dp/0596529325/ref=sr_1_2?crid=1UBVCJKMM17Q6&keywords=collective+intelligence&qid=1553021611&s=books&sprefix=collective+inte%2Cstripbooks%2C236&sr=1-2) book; [Stanford Machine learning training - Andrew Ng](https://www.coursera.org/learn/machine-learning), Intel ML 101, and internet articles.

To avoid impact my laptop (Mac) python installation I use docker image with python and all the needed library. The dockerfile in this folder can help create this development image.

As some of the python codes are using matplotlib and graphics, it is possible to use the MAC display with docker and X11 display (see [this blog](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc)) for details.

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
Then navigate to the python code from the current work folder, you want to execute and call python
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

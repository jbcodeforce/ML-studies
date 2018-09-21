# Machine Learning Studies - Basic
This repository includes different code to study how to do machine learning with Python and jupyter notebook.

It is based from different sources like 'Machine learning with Python' book;
collective intelligence book; Standford Machine learning trainng, Intel ML 101, and internet articles.

To avoid impact my laptop (Mac) python installation I use docker image with python and all the needed library. The dockerfile in this folder can help create an image.

## Run
Start a docker container with an active bash shell with the command:
```
docker run -v $(pwd):/home/jovyan/work -it pystacktf /bin/bash
```
Then navigate to the python code you want to execute and call python
```
$ python3 deep-net-keras.py
```

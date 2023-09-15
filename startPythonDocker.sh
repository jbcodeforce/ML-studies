#!/bin/bash

echo "##########################################################"
echo "Python 3.11.3+ docker image includes spark, keras, jupyter, tensorflow "
echo
source ./setDisplay.sh
NAME=pythonenv
IMAGE=jbcodeforce/python

docker run -e DISPLAY=$DISPLAY \
   --name $NAME -v $(pwd):/app/ -v $HOME/.aws:/root/.aws -it \
   --rm -p 5002:5000 -p 8888:8888 \
   $IMAGE /bin/bash 

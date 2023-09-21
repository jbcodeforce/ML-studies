#!/bin/bash

echo "##########################################################"
echo "Python 3.11.3+ docker image includes spark, keras, boto3, aws cli"
echo
source ./setDisplay.sh
NAME=pythonenv

# give me a test argument
if [ $# -eq 0 ]
then
   TAG=latest
else
    TAG=$1
fi

IMAGE=jbcodeforce/python:$TAG

docker run -e DISPLAY=$DISPLAY \
   --name $NAME -v $(pwd):/app/ -v $HOME/.aws:/root/.aws -it \
   --rm -p 5002:5000 -p 8888:8888 \
   $IMAGE /bin/bash 

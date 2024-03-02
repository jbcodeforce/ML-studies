#!/bin/bash
echo "##########################################################"
echo " docker image includes spark, keras, jupyter, tensorflow "
echo "##########################################################"
docker run --name pysparktf -v $(pwd):/home/jovyan/work -it --rm -p 8888:8888 jbcodeforce/python
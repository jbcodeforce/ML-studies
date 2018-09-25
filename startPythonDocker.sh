#!/bin/bash
echo "##########################################################"
echo " docker image includes spark, keras, jupyter, tensorflow "
echo
docker run -e DISPLAY=192.168.1.89:0 --name pysparktf -v $(pwd):/home/jovyan/work -it --rm -p 8888:8888 ibmcase/mlpython /bin/bash 

### Run my python development shell

As some of the python codes are using matplotlib and graphics, it is possible to use the MAC display 
with docker and X11 display (see [this blog](https://cntnr.io/running-guis-with-docker-on-mac-os-x-a14df6a76efc)) for details, which can be summarized as:

* By default XQuartz will listen on a UNIX socket, which is private and only exists on local filesystem. Install `socat` to create a two bidirectional streams between Xquartz and a client endpoints: `brew install socat`.
* Install XQuartz with `brew install xquartz`. Then start Xquartz from the application or using: `open -a Xquartz`. A white terminal window will pop up.  
* The first time Xquartz is started, within the X11 Preferences window, select the Security tab and ensure the `allow connections from network clients` is ticked.
* Set the display using the ipaddress pod the host en0 interface: `ifconfig en0 | grep "inet " | awk '{print $2}` 


Then run the following command to open a two bi-directional streams between the docker container and the X window system of Xquartz.

```shell
source ./setDisplay.sh
# If needed install Socket Cat: socat with:  brew install socat
socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
```

which is what the `socatStart.sh` script does.

Start a docker container with the command from the root folder:

```shell
./startPythonDocker.sh
root@...$ cd /app/ml-python
```

Then navigate to the python code from the current `/app` folder, and call python

```sh
$ python deep-net-keras.py
```

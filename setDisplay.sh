#!/bin/bash
IPADDR=$(ifconfig en0 | grep "inet " | awk '{print $2}')
export DISPLAY=$IPADDR:0
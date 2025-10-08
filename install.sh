#!/usr/bin/env bash

#########
# setup.py
#########
pip install -e .


#########
# Manual setup
#########

# common requirements
# pip install -r requirements.txt --no-cache-dir

# gym (& mujoco) requirements
# conda install -c conda-forge glfw glew -y
# conda install -c menpo mesa-libgl-cos6-x86_64 glfw3 osmesa -y
# conda install -c borismarin libx11 -y

# pip install -r req_gym.txt --no-cache-dir

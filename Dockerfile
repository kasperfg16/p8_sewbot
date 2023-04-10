# syntax=docker/dockerfile:1

# Start your image with a node base image. We use ubuntu 16.04 as OS
FROM ubuntu:16.04
FROM ros:kinetic
# Start from the lastest version of Tensorflow with GPU support
# FROM tensorflow/tensorflow:nightly-gpu

# install necessary extensions
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev &&\
    set -xe && apt-get -yqq update && apt-get -yqq install python3.5 python3-pip && pip3 install --upgrade pip &&\
    apt-get install -y vim nano git tmux wget curl net-tools iputils-ping  -y
    #apt-get install wget python3-retrying 

# Start from the lastest version of Tensorflow with GPU support
# FROM tensorflow/tensorflow:nightly-gpu

# Version of dependencies
ENV UBUNTU="16.04" \
	MUJOCO="2.3.3" \
	ROS="Kinetic" \
	Gymnasium="0.26.2"

####################################################################
# ROS (http://wiki.ros.org/kinetic/Installation/Ubuntu)
####################################################################

# NOTE: Kinetic requires Ubuntu >15

# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
#     #apt-key del 421C365BD9FF1F717815A3895523BAEEB01FA116 &&\
#  	#apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 0xB01FA116 && \
#     apt-get install ca-certificates &&\
#     apt-get install curl &&\
#     curl -0 https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - &&\
#  	apt-get update && apt-get install -y \
#  		ros-kinetic-desktop \
#  		python3-rosinstall

# RUN rosdep init \
# 	&& rosdep update

####################################################################
# P8 SEWBOT Repository
####################################################################


ENV SEWBOT_WS=/root/sewbot_ws
WORKDIR $SEWBOT_WS
RUN git clone https://github.com/kasperfg16/p8_sewbot.git $SEWBOT_WS \
    && git checkout simulation \
    && git pull origin simulation 

####################################################################
# MuJoCo & OpenAI Gymnasium
####################################################################
RUN	wget https://github.com/deepmind/mujoco/releases/download/2.3.3/mujoco-2.3.3-linux-x86_64.tar.gz -O mujoco.tar.gz \
	&& tar -xf mujoco.tar.gz -C /root/sewbot_ws \
    && rm mujoco.tar.gz 
RUN pip3 install gymnasium[mujoco] 


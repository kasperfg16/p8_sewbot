# syntax=docker/dockerfile:1

# Start your image with a node base image. We use ubuntu 16.04 as OS
#FROM ubuntu:16.04
FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
FROM ros:kinetic
# Start from the lastest version of Tensorflow with GPU support
# FROM tensorflow/tensorflow:nightly-gpu

# install necessary extensions
RUN apt-get update && apt-get install -y build-essential kmod libjpeg-dev libssl-dev mesa-common-dev \
	libncurses5-dev libsqlite3-dev libreadline-dev libtk8.6 libgdm-dev libdb4o-cil-dev libpcap-dev\
	# libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common net-tools 
	vim nano git tmux wget curl net-tools iputils-ping \
	# Sim shit
	xorg-dev libglfw3 libglfw3-dev	&&\ 

    cd /opt && wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz &&\
	tar -xvf Python-3.8.1.tgz &&\
	cd Python-3.8.1 && ./configure &&\
	apt-get install zlib1g-dev -y &&\
	make && make install &&\
	pip3 install --upgrade pip 


# RUN rm root/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 root/usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 root/usr/share/glvnd/egl_vendor.d/50_mesa.json
# COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
# COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
# ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
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
RUN pip install mujoco && pip install gymnasium[mujoco] \
# clone the pre-built distribution.
	&& wget https://github.com/deepmind/mujoco/releases/download/2.3.3/mujoco-2.3.3-linux-x86_64.tar.gz -O mujoco.tar.gz \
	&& tar -xf mujoco.tar.gz -C /root/sewbot_ws/src && rm mujoco.tar.gz 


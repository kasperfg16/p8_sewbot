# syntax=docker/dockerfile:1

# Start your image with a node base image. We use ubuntu 16.04 as OS
FROM ubuntu:16.04
FROM ros:kinetic

# install necessary extensions
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev &&\
    apt-get install -y vim nano git tmux wget curl pyhton3.5 python3-pip net-tools iputils-ping  -y &&\
    apt-get install wget && \    
    apt-get install -y git 
    #openssh-server

ENV SEWBOT_WS=/root/sewbot_ws
RUN git clone https://github.com/kasperfg16/p8_sewbot.git 
#\    pip3 install -e /home/gymuser/isaac_rover_mars_gym/.

WORKDIR /root/sewbot_ws



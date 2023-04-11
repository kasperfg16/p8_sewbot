# syntax=docker/dockerfile:1

# Start your image with a node base image. We use ubuntu 16.04 as OS
FROM ubuntu:16.04
FROM ros:kinetic

# install necessary extensions
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev &&\
    apt-get install -y vim nano git tmux wget curl net-tools iputils-ping  -y &&\
    apt-get install wget && \    
    apt-get install -y git python3-pip

ENV SEWBOT_WS=/root/sewbot_ws
RUN git clone https://github.com/kasperfg16/p8_sewbot.git $SEWBOT_WS
    WORKDIR $SEWBOT_WS
    
RUN rosdep install --from-paths src --ignore-src -y

RUN apt-get update && apt-get install -y apt-utils

RUN sudo sh \
    -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" \
        > /etc/apt/sources.list.d/ros-latest.list'

RUN wget http://packages.ros.org/ros.key -O - | sudo apt-key add -

RUN apt-get update

RUN sudo apt-get install python3-catkin-tools -y &&\
    catkin_make
# syntax=docker/dockerfile:1

# Start your image with a node base image. We use ubuntu 16.04 as OS
FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
#FROM ubuntu:16.04
FROM ros:kinetic

# install necessary extensions
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev &&\
    apt-get install -y vim nano git tmux wget curl net-tools iputils-ping  -y &&\
    apt-get install wget && \    
    apt-get install -y git python3-pip\
    rosdep install --from-paths src --ignore-src -r -y
    
ENV SEWBOT_WS=/root/sewbot_ws
RUN git clone https://github.com/kasperfg16/p8_sewbot.git $SEWBOT_WS

WORKDIR /root/sewbot_ws

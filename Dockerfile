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

WORKDIR /root/sewbot_ws
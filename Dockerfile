# syntax=docker/dockerfile:1

# Start your image with a node base image. We use ubuntu 16.04 as OS
FROM ubuntu:16.04
FROM python:3
FROM ros:kinetic

# Install necessary extensions
RUN apt-get update && apt-get install -y build-essential git libjpeg-dev &&\
    apt-get install -y vim nano git tmux wget curl net-tools iputils-ping  -y &&\
    apt-get install wget && \    
    apt-get install -y git python3-pip

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

####################################################################
# P8 SEWBOT Repository
####################################################################

ENV SEWBOT_WS=/root/sewbot_ws
RUN git clone https://github.com/kasperfg16/p8_sewbot.git $SEWBOT_WS

WORKDIR $SEWBOT_WS
    
RUN rosdep install --from-paths src --ignore-src -y

RUN apt-get update && apt-get install -y apt-utils

RUN sh \
    -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" \
        > /etc/apt/sources.list.d/ros-latest.list'

RUN wget http://packages.ros.org/ros.key -O - | apt-key add -

RUN apt-get update

RUN apt-get install python3-catkin-tools -y

# Set up ROS environment
RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

RUN apt-get install ros-kinetic*joint-trajectory-controller* -y && \
    apt-get install ros-kinetic-joint-state-controller && \
    apt-get install ros-kinetic-roslint && \
    apt-get install ros-kinetic-object-recognition-msgs && \
    apt-get install ros-kinetic-transmission-interface

RUN apt update && \
    apt install python3-venv python3-pip -y && \
    apt install -y python3 python3-pip

####################################################################
# MuJoCo & OpenAI Gymnasium
####################################################################
RUN pip install mujoco && pip install gymnasium[mujoco] \
# clone the pre-built distribution.
	&& wget https://github.com/deepmind/mujoco/releases/download/2.3.3/mujoco-2.3.3-linux-x86_64.tar.gz -O mujoco.tar.gz \
	&& tar -xf mujoco.tar.gz -C /root/sewbot_ws/src && rm mujoco.tar.gz

RUN cd src && \
    git clone https://github.com/ipa-lth/weiss_gripper_ieg76.git

RUN . /opt/ros/kinetic/setup.sh && \
    catkin build

RUN git clone https://github.com/gamleksi/mujoco_ros_control.git
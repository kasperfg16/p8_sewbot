# Start your image with a node base image. We use ubuntu 20.04 as OS
FROM ubuntu:22.04

ENV DIRPATH=/sewbot_ws
WORKDIR $DIRPATH

# Install ROS 2
RUN locale && \
    apt-get update && apt-get install locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && \
    export LANG=en_US.UTF-8 && \
    locale

# Set Timezone
ARG TZ="Europe/London"
ENV TZ ${TZ}

# Setup Environment
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get install software-properties-common -y && \
    add-apt-repository universe && \
    apt-get update && apt-get install curl -y && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null &&\
    apt-get update && \
    apt-get dist-upgrade -y && \
    apt-get install ros-humble-desktop -y

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

RUN pip install mujoco gymnasium[mujoco]

RUN apt-get install ros-humble-ur-robot-driver -y
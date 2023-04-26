# Start your image with a node base image. We use ubuntu 20.04 as OS
FROM ubuntu:20.04

ENV SEWBOT_WS=/root/sewbot_ws
WORKDIR $SEWBOT_WS

## Set Timezone
ARG TZ="Europe/London"
ENV TZ ${TZ}

## Setup Environment
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install python3 -y && \
    apt-get install python3-distutils -y && \
    apt-get install python3-apt -y && \
    apt-get install gnupg -y

# Install ROS noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-get install curl -y && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install ros-noetic-desktop -y

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
    apt-get install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential && \
    apt-get install -y python3-rosdep && \
    rosdep init && \
    rosdep update

# Install pip
RUN apt-get install curl -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py

# Install mujoco and gymnasium
RUN pip install mujoco gymnasium[mujoco]

# Install the ur-robot-driver
RUN . /opt/ros/noetic/setup.bash && \
    git clone https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git src/Universal_Robots_ROS_Driver && \
    git clone -b melodic-devel https://github.com/ros-industrial/universal_robot.git src/universal_robot && \
    apt-get update -qq && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -y && \
    catkin_make

# Make the terminal automatically source the /opt/ros/humble/setup.bash file every time you open a new terminal window

RUN apt-get -y update && \
    apt-get -y install git
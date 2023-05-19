# This is an auto generated Dockerfile for ros:ros-core
# generated from docker_images/create_ros_core_image.Dockerfile.em
FROM ros:noetic

ENV SEWBOT_WS=/root/sewbot_ws
WORKDIR $SEWBOT_WS

# RUN apt-get update

# RUN apt-get install software-properties-common -y

# RUN add-apt-repository universe && \
#     add-apt-repository multiverse && \
#     apt-get update

# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# RUN apt-get update

# RUN apt-get update

# RUN apt-get install gnupg -y

# RUN apt-get install curl -y && \
#     curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# RUN apt-get update

# RUN apt install ros-noetic-desktop-full

WORKDIR $SEWBOT_WS/src

RUN apt-get update && \
    apt-get install git-all -y

RUN apt-get install ros-noetic-catkin

RUN apt-get update && apt-get install -y \
    python3-pip

RUN pip3 install git+https://github.com/catkin/catkin_tools.git

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

RUN apt-get install -y \
    ros-noetic-object-recognition-msgs \
    ros-noetic-controller-manager \
    ros-noetic-control-toolbox \
    ros-noetic-transmission-interface \
    ros-noetic-joint-limits-interface \
    ros-noetic-roslint \
    ros-noetic-angles \
    ros-noetic-eigen-conversions \
    ros-noetic-tf

RUN git clone https://github.com/shadow-robot/mujoco_ros_pkgs.git

RUN apt-get install -y libglfw3-dev build-essential libgl1-mesa-dev

RUN apt-get install -y xorg-dev libglu1-mesa-dev

RUN apt-get install -y libglfw3-dev xorg-dev libglu1-mesa-dev -y

RUN mkdir include && \
    cd include && \
    git clone https://github.com/shadow-robot/mujoco_ros_pkgs.git

#RUN cmake include/mujoco/

# RUN cmake --build .

# RUN cmake --install .

#############################################
# RUN pip3 install mujoco

# RUN apt-get install libgl-dev -y

# RUN git clone https://github.com/saga0619/mujoco_ros_sim.git

# RUN catkin build

# /root/sewbot_ws/include/mujoco/include/mujoco
# /root/sewbot_ws/include/mujoco/include
# /root/sewbot_ws/include/glfw/include/GLFW
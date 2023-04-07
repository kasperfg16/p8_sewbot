# p8_sewbot

This reposotory builds on a Docker container, since the ros interface with KUKA is based on EOL ROS kinetic. Consequently, the Docker image is based on ubuntu 16.04 and ros kinetic

## Table of Contents

- [Rules](#rules)
- [Install Docker](#install-docker)
  - [Build and run this container in terminal](#build-and-run-this-container-in-terminal)
  - [To test Ubuntu and ROS version](#to-test-ubuntu-and-ros-version)
- [Develop on container in VScode](#develop-on-container-in-vscode)
- [Setup real robot](#setup-real-robot)
- [Setup simulation robot](#setup-simulation-robot)
    - [System architecture](#system-architecture)

## Rules

- All files that only relates to your own pc should never be included in commits, make sure to add them to gitignore!.

## Install Docker

### Build and run this container in terminal

1. Follow this https://docs.docker.com/desktop/install/linux-install/

2. In a terminal run this:

    ``` bash
        cd "$(find . -type d -name 'p8_sewbot' -print -quit)"
        sudo docker build -t p8_sewbot .
        clear
        sudo docker run -it --rm p8_sewbot
    ```

### To test Ubuntu and ROS version

``` bash
    cat /etc/os-release
    roscore
```

- Close all terminals

## Develop on container in VScode

- In VSCode, get extension Dev container//remote explorer. 

- Then run this in a terminal to start the docker container:

    ``` bash
        sudo docker run -it --rm p8_sewbot
    ```

- In remote explorer, the running container should appear under Dev Containers. Right click and attach to container.

- To stop a docker image; in terminal

    ``` bash
        exit
    ```

- If you want to stop a docker image in another terminal:

    ``` bash
        sudo docker stop p8_sewbot
    ```

- If you want delete an image:

    ``` bash
        docker image rm -f p8_sewbot
    ```

https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes

## Setup real robot

1. In a terminal

    Install kuka_experimental:

    ``` bash
        cd src
        git clone https://github.com/ros-industrial/kuka_experimental.git
        cd ..
        rosdep install --from-paths src --ignore-src
        catkin_make
        
    ```

    We use the KR C4 controller for the kuka, therefore we follow this tutorial:

    https://github.com/ros-industrial/kuka_experimental/tree/melodic-devel/kuka_rsi_hw_interface/krl/KR_C4

## Setup simulation robot



roslaunch kuka_rsi_simulator kuka_rsi_simulator.launch

### System architecture

[![System architecture](system_architecture.drawio.svg)](https://app.diagrams.net/#Hkasperfg16%2Fp8_sewbot%2Fmain%2Fsystem_architecturedrawio.svg)

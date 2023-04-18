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

2. In a terminal run this to build the docker container:

    Open docker desktop to start the docker engine. It is requred to go through some configurations before the docker engine starts.

    ``` bash
        systemctl --user start docker-desktop
    ```

    Then run this:

    ``` bash
        cd "$(find . -type d -name 'p8_sewbot' -print -quit)"
        sudo docker build -t p8_sewbot .
    ```

    Start the docker container (First start docker-desktop)

    Start docker-desktop:

    ``` bash
        systemctl --user start docker-desktop
    ```

    Run this to test if a container can be created with the image

    ``` bash
        sudo docker run -it --rm p8_sewbot
    ```

    Exit the docker again

    ``` bash
        exit
    ```

    Open docker desktop and do the following

    1. Open Docker Preferences by clicking on the Docker icon in the system tray and selecting "Preferences...".
    2. Go to the "Resources" tab.
    3. Under "File Sharing", click on the "+" button to add a new shared directory.
    4. Navigate to /tmp/.X11-unix in the host file system and click "Add".
    Click "Apply & Restart" to apply the changes and restart Docker.

3. Follow this:
    [Nvidia toolkit install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

4. In a terminal run this:

    ``` bash
        sudo ldconfig
    ```

    ``` bash
        systemctl --user start docker-desktop
    ```

5. In a terminal run this:

    ``` bash
        xhost +
    ```

    ``` bash
        sudo docker run -e DISPLAY=$DISPLAY -it --rm --privileged --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix --name en_syg_container p8_sewbot /bin/bash
    ```

### To test Ubuntu and ROS version

``` bash
    cat /etc/os-release
    roscore
```

- Close all terminals

## Develop on container in VScode

- In VSCode, get extension Dev container//remote explorer.

- Run this in a terminal to enable docker desktop and thereby also VScode to see images and containers created with 'sudo' docker:

    ``` bash
        docker context use default
    ```

- Then run this in a terminal to start the docker container:

    ``` bash
        docker run -it --rm p8_sewbot
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

- To delete everything related to docker images, volumes, and containers. but not docker itself:

    ``` bash
        sudo docker system prune -a
    ```

## Setup real robot

1. In a terminal

    We use the KR C4 controller for the kuka, therefore we follow this tutorial:

    https://github.com/ros-industrial/kuka_experimental/tree/melodic-devel/kuka_rsi_hw_interface/krl/KR_C4

## Setup simulation robot

``` bash
    source devel/setup.bash
    roslaunch kuka_rsi_simulator kuka_rsi_simulator.launch
```

``` bash
    sudo docker exec -it en_syg_container bash
    roslaunch kuka_kr6_support roslaunch_test_kr6r700sixx.xml
```

### System architecture

[![System architecture](system_architecture.drawio.svg)](https://app.diagrams.net/#Hkasperfg16%2Fp8_sewbot%2Fmain%2Fsystem_architecturedrawio.svg)

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

    ``` bash
        cd "$(find . -type d -name 'p8_sewbot' -print -quit)"
        docker build -t p8_sewbot .
    ```

    Start the docker container (First start docker-desktop)

    1. Open Docker Preferences by clicking on the Docker icon in the system tray and selecting "Preferences...".
    2. Go to the "Resources" tab.
    3. Under "File Sharing", click on the "+" button to add a new shared directory.
    4. Navigate to /tmp/.X11-unix in the host file system and click "Add".
    Click "Apply & Restart" to apply the changes and restart Docker.

3. Follow this:
    [Nvidia toolkit install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#:~:text=the%20requested%20devices.-,Docker,%EF%83%81,-For%20installing%20Docker)

4. In a terminal run this:

    ``` bash
        sudo ldconfig
    ```
    
    ``` bash
        systemctl --user start docker-desktop
    ```

5. In a terminal run this:

docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix p8_sewbot

    ``` bash
        xhost +

        sudo docker run -e DISPLAY=$DISPLAY -it --rm --privileged --net=host --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all --env=DISPLAY --env=QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix p8_sewbot /bin/bash
    ```

6. In the same terminal run this:

    ``` bash
        rosdep install --from-paths src --ignore-src -y
        catkin_make
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

xhost +
docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host --gpus=all --name=en_syg_container p8_sewbot /bin/bash

xhost - fi

https://www.digitalocean.com/community/tutorials/how-to-remove-docker-images-containers-and-volumes

## Setup real robot

1. In a terminal

    Install kuka_experimental:

    ``` bash
        rosdep install --from-paths src --ignore-src -y
        catkin_make
    ```

    We use the KR C4 controller for the kuka, therefore we follow this tutorial:

    https://github.com/ros-industrial/kuka_experimental/tree/melodic-devel/kuka_rsi_hw_interface/krl/KR_C4

## Setup simulation robot


docker run -it -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY --network=host p8_sewbot /bin/bash #11. bash docker/run.sh DISPLAYPORT -e DISPLAY=:0

``` bash
    apt-get update
    apt-get install -y dbus
    dbus-uuidgen > /etc/machine-id
    source devel/setup.bash
    roslaunch kuka_rsi_simulator kuka_rsi_simulator.launch
```

``` bash
    roslaunch kuka_kr6_support roslaunch_test_kr6r700sixx.xml
```


### System architecture

[![System architecture](system_architecture.drawio.svg)](https://app.diagrams.net/#Hkasperfg16%2Fp8_sewbot%2Fmain%2Fsystem_architecturedrawio.svg)

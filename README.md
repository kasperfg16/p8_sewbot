# p8_sewbot

This reposotory builds on a Docker container, since the ros interface with KUKA is based on EOL ROS kinetic. Consequently, the Docker image is based on ubuntu 16.04 and ros kinetic

## Install Docker

[DEB package download link](https://desktop.docker.com/linux/main/amd64/docker-desktop-4.17.0-amd64.deb?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-linux-amd64)

## Build and run this container in terminal

May need to run two times:
``` bash
    sudo apt purge docker.io -y
    sudo apt remove docker.io -y
    sudo apt autoremove -y
    sudo groupdel docker -y
    sudo apt update -y
    sudo apt install docker.io -y
    export PATH="/usr/bin:$PATH"
    sudo systemctl restart docker
    sudo service docker restart
```

Then run this:
``` bash
    cd "$(find . -type d -name 'p8_sewbot' -print -quit)"
    sudo docker build -t p8_sewbot .
    clear
    sudo docker run -it --rm p8_sewbot
```

## To test Ubuntu and ROS version

``` bash
    cat /etc/os-release
    roscore 
```

- In VSCode, get extension Dev container//remote explorer. 
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

## Rules

- All files that only relates to your own pc should never be included in commits, make sure to add them to gitignore!.

## Prerequisites

1. Install python

    <https://www.python.org/downloads/>

2. Install ROS2 galaxtic:

    <https://docs.ros.org/en/galactic/Installation/Ubuntu-Development-Setup.html>

3. Install pip

    <https://www.geeksforgeeks.org/how-to-install-pip-on-windows/>

4. In a terminal

    Install dependecies, extra ros-packeages etc.

    ``` bash
    pip install -r pip install -r requirements.txt 
    ```

### ROS packages

Install extra ROS packages

1. In a terminal

    Install packages:

    ``` bash
    rosdep install --from-paths src --ignore-src -r -y
    ```

[![System architecture](system_architecture.drawio.svg)](https://app.diagrams.net/#Hkasperfg16%2Fp8_sewbot%2Fmain%2Fsystem_architecturedrawio.svg)

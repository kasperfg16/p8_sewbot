# p8_sewbot
This reposotory builds on a Docker container, since the ros interface with KUKA is based on EOL ROS kinetic. Consequently, the Docker image is based on ubuntu 16.04 and ros kinetic

### Install Docker:

[DEB package download link](https://desktop.docker.com/linux/main/amd64/docker-desktop-4.17.0-amd64.deb?utm_source=docker&utm_medium=webreferral&utm_campaign=docs-driven-download-linux-amd64)

## To build and run this container in terminal

``` bash
    docker build -t p8_sewbot .
    sudo docker run -it --rm p8_sewbot
```

## To test Ubuntu and ROS version;

``` bash
    cat /etc/os-release
    roscore 
```

- In VSCode, get extension Dev container//remote explorer. 
- In remote explorer, the running container should appear under Dev Containers. Right click and attach to container.

- To stop a docker image; in terminal, 
``` bash
    exit
```

- If you want to stop a docker image in another terminal:
``` bash
    sudo docker stop p8_sewbot
```

- If you want delete an image:
``` bash
    docker images -r p8_sewbot
```

## Rules

- All files that only relates to your own pc should never be included in commits, make sure to add them to gitignore!.

- All custom environments should be added to gitignore.

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
    sudo apt install ros-galactic-joint-state-publisher-gui
    ```

    ``` bash
    sudo apt install ros-galactic-xacro
    ```

    ``` bash
    sudo apt install ros-galactic-ros2-control
    ```

    ``` bash
    sudo apt install ros-galactic-ros2-controllers
    ```

    ``` bash
    sudo apt install ros-galactic-gazebo-ros-pkgs
    ```

    ``` bash
    sudo apt install ros-galactic-ros-core ros-galactic-geometry2
    ```

    ``` bash
    sudo apt-get install ros-galactic-turtle-tf2-py ros-galactic-tf2-tools ros-galactic-tf-transformations
    ```

    ``` bash
    sudo apt install ros-galactic-robot-localization
    ```

    ``` bash
    sudo apt install ros-galactic-tf2-geometry-msgs
    ```

    ``` bash
    pip install transformations
    ```

    ``` bash
    sudo apt install ros-galactic-gazebo-ros-pkgs
    ```

## General setup

1. In a terminal:

    - a)

        Create a workspace folder with a **PATH** of your choice. Remember/write down the **PATH** for later:

        ``` bash
        mkdir PATH
        ```

    - b)

        Clone the reposetory:

        ``` bash
        git clone https://github.com/kasperfg16/p6-swarm.git
        ```

    - c)

        Go into the workspace folder and build the package:

        ``` bash
        cd PATH
        mkdir config launch maps meshes models params rviz worlds
        colcon build
        ```

    - d)

        Add some commands to .bashrc file, so you don't have to source the project every time.

        ``` bash
        echo 'source /opt/ros/galactic/setup.bash' >> ~/.bashrc
        echo 'source ~/ros2_galactic/install/local_setup.bash' >> ~/.bashrc
        ```

        In this command remember to change **"PATH"** to the **PATH** where you cloned to reposetory to:

        ``` bash
        echo 'source PATH/install/setup.bash' >> ~/.bashrc
        ```

        Now everytime you open a terminal in root it automaticly sources the project.

## System Diagram

[![Test Embedding draw.io](./system_architecturedrawio.svg)](https://app.diagrams.net/#Hkasperfg16%2Fp8_sewbot%2Fmain%2Fsrc%2FUnavngivet%20diagram.drawio.svg)

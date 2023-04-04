# p8_sewbot

### Install Docker:
- $ sudo apt-get install docker.io
- $ sudo apt-get install dockerd -> to get docker deamon. Might not be necessary


## To test Ubuntu and ROS version;
- $ cat /etc/os-release
- $ roscore 

- To stop a docker image, $ sudo docker stop container_id.
## Rules

- All files that only relates to your own pc should never be included in commits, make sure to add them to gitignore!.

- All custom environments should be added to gitignore.

## Recommmendations

1. We recommend to read the project report for an understanding of the system. At least the "Prototype" section.

2. This project is using ROS2 galactic on ubuntu 20.04. Other configurations are used at own dispare and misery

3. We recommend using dualboot via USB: <https://www.youtube.com/watch?v=cHF1ByFKtZo&t=315s>. In this way you can transfer all files between systems on the go.

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

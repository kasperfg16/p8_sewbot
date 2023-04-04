# syntax=docker/dockerfile:1

# Start your image with a node base image. We use ubuntu 16.04 as OS
FROM ubuntu:16.04
FROM ros:kinetic

RUN apt-get update && apt-get install -y build-essential git libjpeg-dev &&\
    apt-get install -y vim nano git tmux wget curl python-pip net-tools iputils-ping  -y &&\
    apt-get install wget && \    
    apt-get install -y git openssh-server

# # 1. Create the SSH directory.
# RUN mkdir -p /root/.ssh/
# # 2. Populate the private key file.
# RUN echo "$SSH_KEY" > /root/.ssh/id_rsa
# # 3. Set the required permissions.
# RUN chmod -R 600 /root/.ssh/
# # 4. Add github to our list of known hosts for ssh.
# RUN ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# inject a datestamp arg which is treated as an environment variable and
# will break the cache for the next RUN command
#ARG DATE_STAMP

ENV SEWBOT_WS=/root/sewbot_ws
RUN wget git@github.com:kasperfg16/p8_sewbot-main.zip $SEWBOT_WS \
    unzip p8_sewbot.zip

# Create ROS workspace
#ENV SEWBOT_WS=/root/sewbot_ws
# Copy project folder
#RUN git clone https://github.com:mariebogestrand/p8_sewbot.git $SEWBOT_WS

WORKDIR /root/sewbot_ws
#WORKDIR /home/marie/Desktop/P8_sewbot
#COPY src/tester.py ./
#COPY ./ ./

#CMD ["python", "tester.py"]


#WORKDIR /root

#COPY ./entrypoint.sh /
#ENTRYPOINT ["/entrypoint.sh"]


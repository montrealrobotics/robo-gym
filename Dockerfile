FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

FROM ros:noetic-robot 


ARG ROBOGYM_WS=/root/robogym_ws
ARG ROS_DISTRO=noetic

SHELL ["/bin/bash", "-c"]

# RUN apt-get update && apt-get install -y gnupg2 lsb-release && apt-get clean all
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

RUN apt-get update && apt-get install -y git vim apt-utils build-essential psmisc vim-gtk git swig sudo libcppunit-dev python3-catkin-tools python3-rosdep python3-pip python3-rospkg python3-future python3-osrf-pycommon
RUN apt-get install -y curl tmux unzip libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev xvfb patchelf ffmpeg


# RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# RUN apt-get update && apt-get install -y ros-noetic-desktop-full
RUN ln -s /usr/bin/python3 /usr/bin/python
# Set robo-gym ROS workspace folder
RUN export ROBOGYM_WS=~/robogym_ws 
# Set ROS distribution
RUN export ROS_DISTRO=noetic
RUN rm -rf $ROBOGYM_WS
RUN mkdir -p $ROBOGYM_WS/src && cd $ROBOGYM_WS/src && git clone https://github.com/montrealrobotics/robo-gym-robot-servers.git

WORKDIR /root/robogym_ws/src
RUN git clone https://github.com/montrealrobotics/ur_kinematics.git
RUN git clone -b $ROS_DISTRO https://github.com/jr-robotics/mir_robot.git
RUN git clone -b $ROS_DISTRO https://github.com/jr-robotics/universal_robot.git
RUN git clone -b v0.7.1-dev https://github.com/jr-robotics/franka_ros_interface
RUN git clone https://github.com/jr-robotics/franka_panda_description
RUN git clone -b ${ROS_DISTRO}-devel https://github.com/jr-robotics/panda_simulator
RUN git clone https://github.com/orocos/orocos_kinematics_dynamics
RUN cd orocos_kinematics_dynamics && git checkout b35c424e77ebc5b7e6f1c5e5c34f8a4666fbf5bc
RUN cd $ROBOGYM_WS
RUN apt-get update
RUN rm -rf /etc/ros/rosdep/sources.list.d/20-default.list


WORKDIR /root/robogym_ws
RUN rosdep init
RUN rosdep update
RUN rosdep install --from-paths src -i -y --rosdistro $ROS_DISTRO
# RUN catkin init
RUN source /opt/ros/noetic/setup.bash
# RUN catkin config --cmake-args -Dcatkin_DIR=CATKIN_CMAKE_CONFIG_PATH
# RUN catkin build
# RUN apt-get install ros-noetic-catkin
RUN catkin config \
      --extend /opt/ros/$ROS_DISTRO && \
    catkin build
RUN pip3 install robo-gym-server-modules scipy numpy
RUN pip3 install protobuf==3.20.0


RUN printf "source /opt/ros/$ROS_DISTRO/setup.bash\nsource $ROBOGYM_WS/devel/setup.bash" >> ~/.bashrc
RUN pip install pytest pytest-rerunfailures

## Download robo-gym 
WORKDIR /root/
RUN git clone -b dev https://github.com/montrealrobotics/robo-gym.git
RUN git clone -b robo-gym https://github.com/montrealrobotics/cleanrl.git
RUN pip install -r cleanrl/requirements/requirements.txt 


## If you are on RTX 30 devices 
# RUN pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu110/torch_nightly.html -U

WORKDIR /root
COPY . /usr/local/robo-gym/
WORKDIR /usr/local/robo-gym/
RUN pip install .
ENTRYPOINT ["/usr/local/robo-gym/bin/docker_entrypoint"]
CMD ["bash"]

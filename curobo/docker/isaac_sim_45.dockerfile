#################
# This docker file builds:
# Ubuntu 22.04
# CUDA 11.8
# Isaac Sim 4.5.0 (omni_python is 3.10)
# Curobo as in original (as in original curobo docker file of isaac sim =< 4.0.0)
# Isaac_ros_ws 3.2.0 
# Ros 2 humble (python 3.10 as same as omni_python)

# Build result (image created): should be identical to the image:
# de257/curobo_isaac45
# Tag version: sha256:5f76f7fdf0a7caf1327eeb73392d7b490d851e008aacef14b4ae5d66209098f6
# can be pulled from:https://hub.docker.com/repository/docker/de257/curobo_isaac45/tags/latest/sha256:5f76f7fdf0a7caf1327eeb73392d7b490d851e008aacef14b4ae5d66209098f6
#################



#############################################
#################################################
#################################################
# PART 1: 
# GOAL: MIRRORING THE ORIGINAL CUROBO'S ISAAC SIM(=< 4.0.0) DOCKER FILE : curobo/docker/isaac_sim.dockerfile
# ONLY VERY VERY SMALL CHANGES ARE MADE (like the version of isaac sim).
# ISAAC SIM 4.5.0 IS ON DOCKER CONTAIENR WITH UBUNTU 22.04
# ISAAC SIM 4.0.0 IS ON DOCKER CONTAIENR WITH UBUNTU 20.04
# SO THE FILE CONTAINS SOME SMALL CHANGES TO MAKE IT WORK WITH 4.5.0...

# See: 
# https://curobo.org/get_started/5_docker_development.html
# https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_container.html
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim/tags
################################################
#################################################
################################################
################################################# 

ARG DEBIAN_FRONTEND=noninteractive
ARG BASE_DIST=ubuntu22.04
ARG CUDA_VERSION=11.8
ARG ISAAC_SIM_VERSION=4.5.0

# build cores: new var for performance- not mandatory
ARG BUILD_CORES=8


# -----------------------------------------------------------------------------
# Stage 1: pull Isaac-Sim SDK files (unchanged from legacy structure)
# -----------------------------------------------------------------------------
FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION} AS isaac-sim

# -----------------------------------------------------------------------------
# Stage 2: working image based on standard CUDA devel image (GL/Vulkan libs added later)
# -----------------------------------------------------------------------------
# a small modification to the original docker file:
# instead of using cudagl, we use cuda:${CUDA_VERSION}.0-devel-${BASE_DIST}
# this is because the cudagl image is not available for cuda 11.8
# and the cuda devel image is a smaller image that only contains the cuda toolkit
# and the base distribution (ubuntu 22.04)
# this is a small modification to the original docker file:
# instead of using cudagl, we use cuda:${CUDA_VERSION}.0-devel-${BASE_DIST}
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}.0-devel-${BASE_DIST}


LABEL maintainer "User Name"

# vulkan sdk version - had problems with version in past
# might neeed to see which is the latest
# o3 said:
# On Jammy we already get the Vulkan loader, headers and tools from apt (libvulkan1, vulkan-tools, libegl-mesa0, etc.), which is all Isaac-Sim and RViz need.
# ARG VULKAN_SDK_VERSION=1.3.224.1 - # o3 said: removed the "Installing Vulkan SDK..."" block...
# Deal with getting tons of debconf messages
# See: https://github.com/phusion/baseimage-docker/issues/58
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


# Set timezone info


# libssl3 is the new libssl package for ubuntu 22.04
# it is a replacement for libssl1.1
# see: https://askubuntu.com/questions/1440000/how-to-install-libssl3-in-ubuntu-22-04
RUN apt-get update && apt-get install -y \
  tzdata \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/* \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Los_Angeles" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata \
  && add-apt-repository universe \
  && add-apt-repository -y ppa:git-core/ppa \
  && apt-get update && apt-get install -y \
  curl \
  lsb-core \
  wget \
  build-essential \
  cmake \
  git \
  git-lfs \
  iputils-ping \
  make \
  openssh-server \
  openssh-client \
  libeigen3-dev \
  libssl3 \
  python3-pip \
  python3-ipdb \
  python3-tk \
  python3-wstool \
  sudo git bash unattended-upgrades \
  apt-utils \
  terminator \
  && rm -rf /var/lib/apt/lists/*

# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/cudagl

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatomic1 \
    libegl1 \
    libglu1-mesa \
    libgomp1 \
    libsm6 \
    libxi6 \
    libxrandr2 \
    libxt6 \
    libfreetype-dev \
    libfontconfig1 \
    openssl \
    libssl3 \
    wget \
    vulkan-tools \
&& apt-get -y autoremove \
&& apt-get clean autoclean \
&& rm -rf /var/lib/apt/lists/*

# note:
# o3 said: removed the "Installing Vulkan SDK..."" block...
# Vulkan headers and loader were installed via apt (libvulkan1 & vulkan-tools),
# so the Lunarg SDK tarball download has been removed to avoid 403/registration issues.


# Setup the required capabilities for the container runtime
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all

# Open ports for live streaming
EXPOSE 47995-48012/udp \
        47995-48012/tcp \
        49000-49007/udp \
        49000-49007/tcp \
        49100/tcp \
        8011/tcp \
        8012/tcp \
        8211/tcp \
        8899/tcp \
        8891/tcp

ENV OMNI_SERVER http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/${ISAAC_SIM_VERSION}
ENV MIN_DRIVER_VERSION 525.60.11
# Copy Isaac Sim files
COPY --from=isaac-sim /isaac-sim /isaac-sim
RUN mkdir -p /root/.nvidia-omniverse/config
COPY --from=isaac-sim /root/.nvidia-omniverse/config /root/.nvidia-omniverse/config
COPY --from=isaac-sim /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY --from=isaac-sim /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

WORKDIR /isaac-sim


ENV TORCH_CUDA_ARCH_LIST="7.0+PTX"




# create an alias for omniverse python
ENV omni_python='/isaac-sim/python.sh'

RUN echo "alias omni_python='/isaac-sim/python.sh'" >> ~/.bashrc


# Add cache date to avoid using cached layers older than this
ARG CACHE_DATE=2024-04-11


# if you want to use a different version of curobo, create folder as docker/pkgs and put your
# version of curobo there. Then uncomment below line and comment the next line that clones from
# github

# COPY pkgs /pkgs

RUN mkdir /pkgs && cd /pkgs && git clone https://github.com/NVlabs/curobo.git

RUN $omni_python -m pip install ninja wheel tomli


RUN cd /pkgs/curobo && $omni_python -m pip install .[dev] --no-build-isolation

WORKDIR /pkgs/curobo

# Optionally install nvblox:

RUN apt-get update && \
    apt-get install -y curl tcl && \
    rm -rf /var/lib/apt/lists/*



# install gflags and glog statically, instructions from: https://github.com/nvidia-isaac/nvblox/blob/public/docs/redistributable.md


RUN cd /pkgs && wget https://cmake.org/files/v3.27/cmake-3.27.1.tar.gz && \
    tar -xvzf cmake-3.27.1.tar.gz && \
    apt update &&  apt install -y build-essential checkinstall zlib1g-dev libssl-dev && \
    cd cmake-3.27.1 && ./bootstrap && \
    make -j8 && \
    make install &&  rm -rf /var/lib/apt/lists/*


ENV USE_CX11_ABI=0
ENV PRE_CX11_ABI=ON



RUN cd /pkgs && git clone https://github.com/sqlite/sqlite.git -b version-3.39.4 && \
    cd /pkgs/sqlite && CFLAGS=-fPIC ./configure --prefix=/pkgs/sqlite/install/ && \
    make && make install



RUN cd /pkgs && git clone https://github.com/google/glog.git -b v0.6.0 && \
    cd glog && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=/pkgs/glog/install/ \
    -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install


RUN cd /pkgs && git clone https://github.com/gflags/gflags.git -b v2.2.2 && \
    cd gflags &&  \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=/pkgs/gflags/install/ \
    -DGFLAGS_BUILD_STATIC_LIBS=ON -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install

RUN cd /pkgs &&  git clone https://github.com/valtsblukis/nvblox.git && cd /pkgs/nvblox/nvblox && \
    mkdir build && cd build && \
    cmake ..  -DBUILD_REDISTRIBUTABLE=ON \
    -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI}  -DPRE_CXX11_ABI_LINKABLE=${PRE_CX11_ABI} \
    -DSQLITE3_BASE_PATH="/pkgs/sqlite/install/" -DGLOG_BASE_PATH="/pkgs/glog/install/" \
    -DGFLAGS_BASE_PATH="/pkgs/gflags/install/" -DCMAKE_CUDA_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    -DBUILD_TESTING=OFF && \
    make -j32 && \
    make install

# we also need libglog for pytorch:
RUN cd /pkgs/glog && \
    mkdir build_isaac && cd build_isaac && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install

RUN cd /pkgs && git clone https://github.com/nvlabs/nvblox_torch.git && \
    cd /pkgs/nvblox_torch && \
    sh install_isaac_sim.sh $($omni_python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)') && \
    $omni_python -m pip install -e .

# install realsense for nvblox demos:
RUN $omni_python -m pip install pyrealsense2 opencv-python transforms3d

RUN $omni_python -m pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"


#############################################
#################################################
#################################################
# PART 2: EXTRA SW THAT WASN'T PART OF ORIGINAL DOCKER FILE (ros 2 and isaac_ros_ws and optionally ore)

# INSTALL ROS2 (HUMBLE? THE MOST IMPORTANT THING IS THAT IS HAS SAME PHYTHON AS omni_isaac (i think its 3.10))
# INSTALL isaac_ros_ws packages (also-must be concise with isaacsim and ros env vars: python version, cuda version, torch version)
 
################################################
#################################################
################################################
################################################# 

# Create a non-root user for development (optional but recommended)
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN if ! id -u $USERNAME >/dev/null 2>&1; then \
        useradd -m -s /bin/bash $USERNAME; \
    fi && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers


# -----------------------------------------------------------------------------
# ROS2
# -----------------------------------------------------------------------------
# Configure official ROS 2 apt repository and install the minimal (non-GUI)
# ros-humble-ros-base package plus build tools.  This provides the core ROS 2
# runtime that Isaac ROS binaries depend on.

# change to one of these packages if need a lighter install:
# ros-humble-desktop-full # heaviest (includes everything, like rviz, rqt, gazebo, etc)
# ros-humble-ros-desktop # lighter (with gui)
# ros-humble-ros-base # lightest (no gui)




RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl gnupg lsb-release && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
         http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ros-humble-desktop-full python3-colcon-common-extensions bash-completion && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/ros/humble/setup.bash" >> /etc/profile.d/ros.sh

# -----------------------------------------------------------------------------
# Isaac ROS stacks (binary install via NVIDIA apt repository)
# -----------------------------------------------------------------------------
# (Disabled – binary repo is gone; we build from source instead)
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends curl gnupg ca-certificates && \
#     curl -fsSL https://nvidia-isaac-ros.s3.us-west-2.amazonaws.com/apt/isaac-ros.gpg \
#       | gpg --dearmor -o /usr/share/keyrings/isaac-ros.gpg && \
#     echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/isaac-ros.gpg] \
#          https://nvidia-isaac-ros.s3.us-west-2.amazonaws.com/apt/isaac-ros $(lsb_release -cs) main" \
#          > /etc/apt/sources.list.d/isaac_ros.list

# Add NVIDIA VPI runtime/dev packages (needed by several Isaac-ROS stacks on x86).
RUN apt-get update && apt-get install -y --no-install-recommends gnupg software-properties-common && \
    apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
    add-apt-repository 'deb https://repo.download.nvidia.com/jetson/x86_64/jammy r36.4 main' && \
    apt-get update && \
    apt-get install -y --no-install-recommends libnvvpi3 vpi3-dev && \
    rm -rf /var/lib/apt/lists/*

# Restore source build of Isaac-ROS stacks (previously commented by mistake)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-rosdep python3-vcstool git-lfs && \
    git lfs install --skip-repo && \
    mkdir -p /workspaces/isaac_ros_ws/src && \
    cd /workspaces/isaac_ros_ws && \
    git clone --depth 1 --branch release-3.2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git src/isaac_ros_common && \
    curl -fsSL https://raw.githubusercontent.com/NVIDIA-ISAAC-ROS/isaac_ros_common/release-3.2/scripts/isaac_ros_common.repos | vcs import src || true && \
    git lfs pull --include "*" --exclude "" || true && \
    rosdep init && rosdep update && \
    rosdep install --from-paths src -y --rosdistro humble --ignore-src && \
    /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install --cmake-args -DCMAKE_CUDA_ARCHITECTURES=70" && \
    echo 'source /workspaces/isaac_ros_ws/install/setup.bash' >> /etc/profile.d/isaac_ros_ws.sh

# Add rclpy & matplotlib to Isaac-Sim's Python so scripts inside /isaac-sim can
# talk directly to ROS 2 and plot.
RUN $omni_python -m pip install --no-cache-dir matplotlib

# Python.  If you really need rclpy inside omni_python, uncomment the next line
# and accept the long compile time.
# RUN $omni_python -m pip install --no-cache-dir rclpy

# -----------------------------------------------------------------------------
# Create non-root user for Isaac-Sim
# -----------------------------------------------------------------------------
ARG USERNAME=isaac
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Give the user access to Isaac-Sim and workspace directories
RUN chown -R $USERNAME:$USERNAME /isaac-sim /workspaces /pkgs

# Set the default user
USER $USERNAME
WORKDIR /home/$USERNAME

# note:
# In the image, run:
# RUN cd / && git clone https://github.com/Dan7171/rl_for_curobo.git
# WARNING: use omni_python's pip, not the system's pip (system pip is set to ros2's python env and we want to use omni_python's python env)
# RUN cd /rl_for_curobo/curobo && SETUPTOOLS_SCM_PRETEND_VERSION=1.0.0 omni_python -m pip install -e .[isaacsim] --no-build-isolation
# RUN cd /rl_for_curobo && omni_python - pip  install -e . # not sure if this is needed


# # -----------------------------------------------------------------------------
# # Entry point
# # -----------------------------------------------------------------------------
# ENTRYPOINT ["bash"] 


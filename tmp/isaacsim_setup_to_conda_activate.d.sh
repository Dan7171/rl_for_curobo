#!/bin/bash

######################################
######################################
echo configuring conda terminal...
echo ""
# arguments to set by user:
# arg 1: isaacsim executables dir
is_exe=~/isaacsim500/isaacsim/_build/linux-x86_64/release # set to where built executable files are (After build) (not a must)
# arg 2 (optional) ros distro:
export ROS_DISTRO=jazzy # set this variable (humble/jazzy) manually depending on your ros2. 
# arg 3 (optional) isaac ros workspaces dir
export isaac_ros_ws_root=~/IsaacSim-ros_workspaces # set this to your isaac sim ros workspaces dir

######################################
######################################

echo "Configuring isaacsim&ros&curobo conda terminal!"
echo "_______________________________________________"
isaacsim=$is_exe/isaac-sim.sh
echo isaac sim version:
cat $is_exe/VERSION
echo ""

echo linking current conda env terminal to isaacsim...
source $is_exe/setup_conda_env.sh 
echo "conda env now supporting python isaacsim standalone script"
echo ""

##### ADD THIS IF YOU WANT TO USE ROS2 (OPTIONAL) #####

####### configure ros 2 #######

echo linking current terminal to ros 2: $ROS_DISTRO...
source /opt/ros/$ROS_DISTRO/setup.bash # this is a must in ros2, it has nothing to do with isaac sim
# echo "(optional test: run ros2 run demo_nodes_cpp talker to verify... (for more info see: https://# docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html))"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp 
# Can only be set once per terminal.
# Setting this command multiple times will append the internal library path again potentially leading to conflicts
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$is_exe/exts/isaacsim.ros2.bridge/$ROS_DISTRO/lib
echo ""


###### configure isaac sim ros workspaces #########
cwd_tmp=$(pwd)  # Capture current directory
# CRITICAL: Tell colcon to use conda Python instead of system Python
export COLCON_PYTHON_EXECUTABLE=$(which python)
cd $isaac_ros_ws_root/${ROS_DISTRO}_ws
source install/local_setup.bash
cd $cwd_tmp










echo "_______done!________"
echo to run the app: run '$isaacsim'
echo to run ros2: run '$ros2 run demo_nodes_cpp talker (or any other ros2 command)'
echo to run python code optinonally initializing isaacsim app: run '$python %your_code.py%'
echo "sources:"
echo "isaacsim conda setup: https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html"
echo "ros2 setup: https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_ros.html"


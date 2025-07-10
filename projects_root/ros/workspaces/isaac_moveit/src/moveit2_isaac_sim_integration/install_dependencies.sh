#!/bin/bash
# Installation script for MoveIt2 Isaac Sim Integration dependencies
# Run as: bash install_dependencies.sh

set -e  # Exit on error

echo "Installing MoveIt2 Isaac Sim Integration dependencies..."

# Update package list
echo "Updating package list..."
sudo apt update

# Install core MoveIt2 packages
echo "Installing MoveIt2 packages..."
sudo apt install -y \
    ros-humble-moveit \
    ros-humble-moveit-core \
    ros-humble-moveit-msgs \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-kinematics \
    ros-humble-moveit-planners \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-ompl

# Install ROS2 communication packages  
echo "Installing ROS2 communication packages..."
sudo apt install -y \
    ros-humble-rclpy \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-trajectory-msgs \
    ros-humble-std-msgs \
    ros-humble-visualization-msgs \
    ros-humble-builtin-interfaces \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-rviz2 \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher

# Install Python dependencies
# echo "Installing Python dependencies..."
# $omni_python # -m pip install numpy pyyaml already installed in curobo_isaac45v3 docker container

# Optional: Install additional useful packages
echo "Installing additional useful packages..."
sudo apt install -y \
    ros-humble-xacro \
    ros-humble-urdf \
    ros-humble-kdl-parser \
    ros-humble-moveit-resources-panda-moveit-config

# Try to install moveit_commander if available
echo "Attempting to install moveit_commander..."
omni_python -m pip install moveit_commander || echo "Note: moveit_commander not available via pip. This is optional."

echo "✓ All dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Source your ROS2 environment: source /opt/ros/humble/setup.bash"
echo "2. Build the package: colcon build --packages-select moveit2_isaac_sim_integration"
echo "3. Source the workspace: source install/setup.bash"
echo "4. Run the test: omni_python scripts/test_integration.py" 
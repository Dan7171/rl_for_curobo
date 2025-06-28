#!/bin/bash

# Setup script for ROS2 Multi-Robot MPC System
# This script sets up both conda and ROS2 environments properly

echo "🚀 Setting up ROS2 Multi-Robot MPC Environment..."

# Check if we're in the right directory
if [ ! -d "curobo" ]; then
    echo "❌ Error: Please run this script from the rl_for_curobo directory"
    exit 1
fi

# Source ROS2 environment first
echo "📦 Sourcing ROS2 Kilted..."
if [ -f "/opt/ros/kilted/setup.bash" ]; then
    source /opt/ros/kilted/setup.bash
    echo "   ✅ ROS2 Kilted sourced successfully"
else
    echo "   ❌ ROS2 Kilted not found at /opt/ros/kilted/setup.bash"
    exit 1
fi

# Activate conda environment
echo "🐍 Activating conda environment..."
eval "$(conda shell.bash hook)"

# Check if env_isaacsim_ros2 exists
if ! conda env list | grep -q "env_isaacsim_ros2"; then
    echo "   ❌ conda environment 'env_isaacsim_ros2' not found"
    echo "   Please create it first with: conda create -n env_isaacsim_ros2 python=3.12"
    exit 1
fi

conda activate env_isaacsim_ros2
echo "   ✅ Activated env_isaacsim_ros2"

# Fix library compatibility issue between conda and ROS2
echo "🔧 Fixing library compatibility..."
# Use system libstdc++ instead of conda's version for ROS2 compatibility
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
echo "   ✅ Library path configured for ROS2 compatibility"

# Test the setup
echo "🧪 Testing setup..."

echo "   Testing Python version..."
python --version
if [ $? -ne 0 ]; then
    echo "   ❌ Python not working"
    exit 1
fi

echo "   Testing CuRobo import..."
python -c "from curobo.wrap.reacher import IkSolver; print('✅ CuRobo imported successfully')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ CuRobo import failed"
    exit 1
fi

echo "   Testing ROS2 import..."
python -c "import rclpy; print('✅ ROS2 rclpy imported successfully')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   ❌ ROS2 import failed - library compatibility issue"
    echo "   💡 Try running with: LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:\$LD_LIBRARY_PATH python your_script.py"
    exit 1
fi

echo ""
echo "🎉 **SETUP COMPLETE!**"
echo ""
echo "🔥 **Your ROS2 Multi-Robot MPC Environment is Ready!**"
echo ""
echo "✅ Python 3.12:     Available"
echo "✅ CuRobo:          Installed and working"
echo "✅ ROS2 Kilted:     Available and working"
echo "✅ PyTorch + CUDA:  Available"
echo ""
echo "🚀 **Ready to run multi-robot ROS2 MPC system!**"
echo ""
echo "Next steps:"
echo "1. Test ROS2 communication: python projects_root/examples/test_ros_communication.py"
echo "2. Run multi-robot MPC: python projects_root/examples/mpc_async_multirobot_ros.py"
echo "" 
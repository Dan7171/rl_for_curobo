#!/bin/bash

# Setup development environment for Isaac Sim
# This script reinstalls the rl_for_curobo package from the mounted source

echo "Setting up development environment..."

# Change to the mounted repository
cd /workspace/rl_for_curobo

# Source Isaac Sim environment
if [ -f "/opt/isaac-sim/setup_conda_env.sh" ]; then
    source /opt/isaac-sim/setup_conda_env.sh
    echo "Sourced Isaac Sim conda environment"
elif [ -f "/isaac-sim/setup_conda_env.sh" ]; then
    source /isaac-sim/setup_conda_env.sh
    echo "Sourced Isaac Sim conda environment"
else
    echo "Warning: Could not find Isaac Sim setup script"
fi

# Uninstall existing package
echo "Uninstalling existing rl_for_curobo package..."
/opt/isaac-sim/omni/python/bin/python -m pip uninstall -y rl_for_curobo || true

# Install from mounted source
echo "Installing rl_for_curobo from mounted source..."
/opt/isaac-sim/omni/python/bin/python -m pip install -e /workspace/rl_for_curobo

# Source ROS if available
if [ -f "/opt/ros/humble/setup.sh" ]; then
    source /opt/ros/humble/setup.sh
    echo "Sourced ROS Humble environment"
fi

echo "Development environment setup complete!"
echo "You can now run: omni_python your_script.py"

# Execute the original command
exec "$@" 
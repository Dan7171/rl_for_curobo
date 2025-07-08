#!/bin/bash

# Simple setup script that avoids omni_python variable issues
cd /workspace/rl_for_curobo

# Try different possible Isaac Sim Python paths
PYTHON_PATHS=(
    "/opt/isaac-sim/omni/python/bin/python"
    "/isaac-sim/omni/python/bin/python"
    "/opt/isaac-sim/python/bin/python"
    "/isaac-sim/python/bin/python"
)

# Find the correct Python path
ISAAC_PYTHON=""
for path in "${PYTHON_PATHS[@]}"; do
    if [ -f "$path" ]; then
        ISAAC_PYTHON="$path"
        echo "Found Isaac Sim Python at: $ISAAC_PYTHON"
        break
    fi
done

if [ -z "$ISAAC_PYTHON" ]; then
    echo "ERROR: Could not find Isaac Sim Python executable"
    echo "Available Python executables:"
    find /opt -name "python" -type f 2>/dev/null | head -10
    exit 1
fi

# Uninstall existing package
echo "Uninstalling existing rl_for_curobo package..."
$ISAAC_PYTHON -m pip uninstall -y rl_for_curobo || true

# Install from mounted source
echo "Installing rl_for_curobo from mounted source..."
$ISAAC_PYTHON -m pip install -e /workspace/rl_for_curobo

# Source ROS if available
if [ -f "/opt/ros/humble/setup.sh" ]; then
    source /opt/ros/humble/setup.sh
    echo "Sourced ROS Humble environment"
fi

echo "Setup complete! You can now run your scripts."
exec "$@" 
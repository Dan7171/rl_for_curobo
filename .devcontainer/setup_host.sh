#!/bin/bash

echo "Setting up host system for Isaac Sim dev container..."

# Create cache directories for Isaac Sim
echo "Creating cache directories..."
mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache,logs,data,documents}

# Set up X11 forwarding permissions
echo "Setting up X11 forwarding..."
xhost +
echo "Host setup complete!"
echo ""
echo "You can now open this project in VSCode/Cursor and select 'Reopen in Container'"
echo ""
echo "Quick start commands once inside the container:"
echo "- Run Isaac Sim: /isaac-sim/isaac-sim.sh"
echo "- Run CuRobo example: omni_python /workspace/rl_for_curobo/projects_root/examples/isaac_sim/mpc_example.py"
echo "- Install your package: /isaac-sim/python.sh -m pip install -e /workspace/rl_for_curobo" 
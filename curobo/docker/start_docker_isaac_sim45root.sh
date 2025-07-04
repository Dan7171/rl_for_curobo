#############
# ROOT LEVEL
#############


# THIS SCRIPT RUNS ISAAC SIM 4.5.0 FROM *ROOT* LEVEL (NOT ROOT)

# # SYNTAX:
# ./start_docker_isaac_sim45user.sh [IMAGE_TAG]

# EXAMPLES:
# ./start_docker_isaac_sim45root.sh           # Uses :latest
# ./start_docker_isaac_sim45root.sh root_level_1# Uses :root_level_1

# Inside the container, the user is root
# run omni_python /pkgs/curobo/examples/isaac_sim/mpc_example.py
# /isaac-sim/isaac-sim.sh


#!/bin/bash

IMAGE_TAG="${1:-latest}"
docker run --name curobo_isaac45_root_container --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host \
  --privileged \
  -e "PRIVACY_CONSENT=Y" \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -e "OMNI_KIT_ALLOW_ROOT=1" \
  -e DISPLAY \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  --volume /dev:/dev \
  de257/curobo_isaac45:${IMAGE_TAG}


# THIS SCRIPT RUNS ISAAC SIM 4.5.0 FROM *USER* LEVEL (NOT ROOT)

# # SYNTAX:
# ./start_docker_isaac_sim45user.sh [IMAGE_TAG]

# EXAMPLES:
# ./start_docker_isaac_sim45user.sh           # Uses :latest
# ./start_docker_isaac_sim45user.sh user_level_1# Uses :user_level_1

# Inside the container, the user is root
# run $omni_python /pkgs/curobo/examples/isaac_sim/mpc_example.py
# /isaac-sim/isaac-sim.sh


#!/bin/bash

set -e

# How to run. e.g.:
# "ACCEPT_EULA=Y ./isaac-sim.docker.gui.sh ./runapp.sh"
# Note: This script is recommended to be run on a workstation with a physical display.

echo "Setting variables..."
command="$@"
if [[ -z "$@" ]]; then
    command="bash"
fi
# Set to desired Nucleus
omni_server="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5"
if ! [[ -z "${OMNI_SERVER}" ]]; then
        omni_server="${OMNI_SERVER}"
fi
# Set to desired Nucleus username
omni_user="admin"
if ! [[ -z "${OMNI_USER}" ]]; then
        omni_user="${OMNI_USER}"
fi
# Set to desired Nucleus password
omni_password="admin"
if ! [[ -z "${OMNI_PASS}" ]]; then
        omni_password="${OMNI_PASS}"
fi
# Set to "Y" to accept EULA
accept_eula=""
if ! [[ -z "${ACCEPT_EULA}" ]]; then
        accept_eula="${ACCEPT_EULA}"
fi
# Set to "Y" to opt-in
privacy_consent=""
if ! [[ -z "${PRIVACY_CONSENT}" ]]; then
        privacy_consent="${PRIVACY_CONSENT}"
fi
# Set to an email or unique user name
privacy_userid="${omni_user}"
if ! [[ -z "${PRIVACY_USERID}" ]]; then
        privacy_userid="${PRIVACY_USERID}"
fi

# echo "Logging in to nvcr.io..."
# docker login nvcr.io

# echo "Pulling docker image..."
# docker pull nvcr.io/nvidia/isaac-sim:4.5.0
IMAGE_TAG="${1:-user_level_1}"
echo "Running Isaac Sim container with X11 forwarding..."
xhost +
docker run --name curobo_isaac45_user_container --entrypoint bash --runtime=nvidia --gpus all -e "ACCEPT_EULA=${accept_eula}" -it --rm --network=host \
        --privileged \
        --user $(id -u):$(id -g) \
        -v $HOME/.Xauthority:/home/isaac/.Xauthority \
        -e DISPLAY \
        -e "OMNI_USER=${omni_user}" -e "OMNI_PASS=${omni_password}" \
        -e "OMNI_SERVER=${omni_server}" \
        -e "PRIVACY_CONSENT=${privacy_consent}" -e "PRIVACY_USERID=${privacy_userid}" \
        -e "OMNI_KIT_ACCEPT_EULA=Y" \
        -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        -v ~/docker/isaac-sim/cache/ov:/home/isaac/.cache/ov:rw \
        -v ~/docker/isaac-sim/cache/pip:/home/isaac/.cache/pip:rw \
        -v ~/docker/isaac-sim/cache/glcache:/home/isaac/.cache/nvidia/GLCache:rw \
        -v ~/docker/isaac-sim/cache/computecache:/home/isaac/.nv/ComputeCache:rw \
        -v ~/docker/isaac-sim/logs:/home/isaac/.nvidia-omniverse/logs:rw \
        -v ~/docker/isaac-sim/data:/home/isaac/.local/share/ov/data:rw \
        -v ~/docker/isaac-sim/documents:/home/isaac/Documents:rw \
        -v ~/docker/isaac-sim/documents/Kit/shared:/isaac-sim/kit/data/documents/Kit/shared:rw \
        -v ~/docker/isaac-sim/documents/Kit:/isaac-sim/kit/data/documents/Kit:rw \
        -v ~/docker/isaac-sim/curobo_assets:/isaac-sim/kit/python/lib/python3.10/site-packages/curobo/content/assets:rw \
        de257/curobo_isaac45:${IMAGE_TAG} 

# -v ~/docker/isaac-sim/cache/asset_browser:/isaac-sim/exts/isaacsim.asset.browser/cache:rw \
# -v ~/docker/isaac-sim/pkg:/home/isaac/.local/share/ov/pkg:rw \
        
#   #-v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
#   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
#   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
#   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
#   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
#   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
#   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
#   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
#   --volume /dev:/dev \

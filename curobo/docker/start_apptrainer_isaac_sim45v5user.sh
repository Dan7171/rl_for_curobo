

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
echo "--------------------------------"
echo "apptainer run script started!"
echo "--------------------------------"

# Initialize default values
# CONTAINER_REGISTRY='de257' # change this to your own registry after pulling the image from the registry
IMAGE_NAME='curobo_isaac45v5' 
INSTANCE_NAME='my_instance'
DC_ENABLED='false' # enable depth camera
DC_DEV_ID='002' # device id for depth camera. You need to run 'lsusb' in linux host to find the device id and pass it
REPO_PATH_HOST=$(realpath ~/rl_for_curobo) # must be absolute path, change this to where you cloned the repo
CMD_IN_CONTAINER='' # No command by default
DEV_CONTAINER_VSCODE_ENABLED='false' # Change this to 'true' if you want to use VSCode in the container
HOST_USER=$(whoami)
HOST_USER_ID=$(id -u)
HOST_GROUP_ID=$(id -g)

CONTAINER_USER=$HOST_USER # dont chagnge this


# Help method
function show_help() {
  echo "Usage: ./start_docker_isaac_sim45root.sh --image-name [IMAGE_NAME] --image-tag [IMAGE_TAG] --container-name [CONTAINER_NAME] --dc-enabled [DC_ENABLED] --dc-dev-id [DC_DEV_ID]"
    echo " --image-name: Name of the Docker image (default: de257/curobo_isaac45)"
    echo " --image-tage: Tag of the Docker image (default: latest)"
    echo " --container-name: Name of the Docker container (default: curobo_isaac45_root_container)"
    echo " --dc-enabled: Enable depth camera (default: true)"
    echo " --dec-dev-id: Device ID for depth camera (default: 002). NOTE:# device id for depth camera. You need to run 'lsusb' in linux host to find the device id and pass it"
    echo " --repo-path-host: project directory location on the host machine, e.g. the directory you cloned this repository into (default: "")"
    echo " --cmd (currently not working): command to run in the container after entering the bash terminal in container (default: No command)"
}


# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;  
        --image-tag)
            IMAGE_TAG="$2"
            shift 2
            ;;  
        --container-name)
            CONTAINER_NAME="$2"
            shift 2
            ;;  
        --dc-enabled)
            DC_ENABLED="$2"
            shift 2
            ;;  
        --dc-dev-id)
            DC_DEV_ID="$2"
            shift 2
            ;;  
        --repo-path-host)
          REPO_PATH_HOST="$2"
          shift 2
          ;;
        --cmd)
          CMD_IN_CONTAINER="$2"
          shift 2
          ;;
        *)
          shift
          ;;  
    esac
done



#REPO_PATH_CONTAINER="/home/isaac/$(basename "$REPO_PATH_HOST")" # /home/isaac/{repo_name}
REPO_NAME=$(basename "$REPO_PATH_HOST")

if [[ -n "$CMD_IN_CONTAINER" ]]; then # if command is provided, use it, otherwise just run "bash" ()
    # DOCKER_CMD=$CMD_IN_CONTAINER
    DOCKER_CMD="$CMD_IN_CONTAINER"
else
    DOCKER_CMD="bash"
fi

if [[ "$DEV_CONTAINER_VSCODE_ENABLED" == "true" ]]; then
    DEV_CONTAINER_FLAG=--detach
else
    DEV_CONTAINER_FLAG=
fi

if [[ "$CONTAINER_USER" == "root" ]]; then
    OMNI_KIT_ALLOW_ROOT=1
    CONTAINER_HOME=/workspace  # DO NOT CHANGE THIS CURRENTLY, IT EXPECTS MODULES TO BE INSTALLED THERE
    user_id=root
    group_id=root
else
    OMNI_KIT_ALLOW_ROOT=0
    CONTAINER_HOME=/workspace
    user_id=$(id -u)
    group_id=$(id -g)
    IMAGE_TAG=latest
fi

REPO_PATH_CONTAINER=$CONTAINER_HOME/$REPO_NAME # dont change this because modules are installed there (currently works just for root)

# Check if depth camera is enabled
if [[ "$DC_ENABLED" == "true" ]]; then
    DC_OPTIONS="--device /dev/bus/usb/$DC_DEV_ID:/dev/bus/usb/$DC_DEV_ID/$DC_DEV_ID \
    --device /dev/video0:/dev/video0 \
    "
    echo using depth camera, you can now run examples like omni_python $REPO_PATH_CONTAINER/curobo/examples/isaac_sim/realsense_reacher.py
else
    DC_OPTIONS=""
fi

# Fix X11 forwarding issues
echo "Setting up X11 forwarding..."
if [[ "$CONTAINER_USER" == "root" ]]; then
  xhost +local:root
else
  xhost +local:developer
fi


set -e

# How to run. e.g.:
# ACCEPT_EULA=Y ./isaac-sim.apptainer.gui.sh ./runapp.sh
# Note: This script is recommended to be run on a workstation with a physical display.

echo "Setting variables..."
command="$@"
if [[ -z "$@" ]]; then
    command="bash"
fi

# Set to desired Nucleus
omni_server="${OMNI_SERVER:-http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5}"
omni_user="${OMNI_USER:-admin}"
omni_password="${OMNI_PASS:-admin}"

# Set to "Y" to accept EULA and privacy
accept_eula="${ACCEPT_EULA:-}"
privacy_consent="${PRIVACY_CONSENT:-}"
privacy_userid="${PRIVACY_USERID:-$omni_user}"

# Ensure X11 access
xhost +

echo "Running Isaac Sim Apptainer container with X11 forwarding..."
echo "--------------------------------"
echo "Instance name: $INSTANCE_NAME will start in background"
echo "to ENTER the instance run: apptainer shell instance://$INSTANCE_NAME "
echo "to STOP the instance run: apptainer instance stop $INSTANCE_NAME to STOP the instance"
echo "to VIEW the living instances run: apptainer instance list"
echo "--------------------------------"
apptainer instance start \
  --writable-tmpfs \
  --nv \
  --env "ACCEPT_EULA=${accept_eula}" \
  --env "OMNI_SERVER=${omni_server}" \
  --env "OMNI_USER=${omni_user}" \
  --env "OMNI_PASS=${omni_password}" \
  --env "PRIVACY_CONSENT=${privacy_consent}" \
  --env "PRIVACY_USERID=${privacy_userid}" \
  --env DISPLAY=$DISPLAY \
  -B $REPO_PATH_HOST:$REPO_PATH_CONTAINER:rw \
  -B $HOME/.Xauthority:/root/.Xauthority \
  -B ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -B ~/docker/isaac-sim/cache/ov:$CONTAINER_HOME/.cache/ov:rw \
  -B ~/docker/isaac-sim/cache/pip:$CONTAINER_HOME/.cache/pip:rw \
  -B ~/docker/isaac-sim/cache/glcache:$CONTAINER_HOME/.cache/nvidia/GLCache:rw \
  -B ~/docker/isaac-sim/cache/computecache:$CONTAINER_HOME/.nv/ComputeCache:rw \
  -B ~/docker/isaac-sim/logs:$CONTAINER_HOME/.nvidia-omniverse/logs:rw \
  -B ~/docker/isaac-sim/data:$CONTAINER_HOME/.local/share/ov/data:rw \
  -B ~/docker/isaac-sim/pkg:$CONTAINER_HOME/.local/share/ov/pkg:rw \
  -B ~/docker/isaac-sim/documents:$CONTAINER_HOME/Documents:rw \
  ~/${IMAGE_NAME}.sif \
  $INSTANCE_NAME \
  bash -c "cd $REPO_PATH_CONTAINER && \
  /isaac-sim/python.sh -m pip install --user ninja && \
  /isaac-sim/python.sh -m pip install --user -e $REPO_PATH_CONTAINER --verbose && \
  source /opt/ros/humble/setup.sh && \
  $DOCKER_CMD"

apptainer exec instance://$INSTANCE_NAME bash -c "cd $REPO_PATH_CONTAINER && \
  /isaac-sim/python.sh -m pip install  tomli wheel ninja && \
  /isaac-sim/python.sh -m pip install --user -e $REPO_PATH_CONTAINER --verbose && \
  source /opt/ros/humble/setup.sh && \
  $DOCKER_CMD"



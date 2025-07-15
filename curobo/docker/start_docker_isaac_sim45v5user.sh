

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
echo "Docker run script started!"
echo "--------------------------------"

# Initialize default values
CONTAINER_REGISTRY='de257' # change this to your own registry after pulling the image from the registry
IMAGE_NAME='curobo_isaac45v5' 
IMAGE_TAG='latest' # 'v1_rl_for_curobo_module_installed' # 'latest'
CONTAINER_NAME='cu_is45_v5' # change this to your desired container name
DC_ENABLED='true' # enable depth camera
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



docker run $DC_OPTIONS \
  --name $CONTAINER_NAME \
  --user $user_id:$group_id \
  $DEV_CONTAINER_FLAG \
  -it \
  --entrypoint bash \
  --gpus all \
  -e "ACCEPT_EULA=Y" \
  --rm \
  --network=host \
  --privileged \
  -e HOME=$CONTAINER_HOME \
  -e "PRIVACY_CONSENT=Y" \
  -v $HOME/.Xauthority:$CONTAINER_HOME/.Xauthority \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e "OMNI_KIT_ALLOW_ROOT=$OMNI_KIT_ALLOW_ROOT" \
  -e DISPLAY \
  -e "NVIDIA_DRIVER_CAPABILITIES=all" \
  -e "NVIDIA_VISIBLE_DEVICES=all" \
  -e "__NV_PRIME_RENDER_OFFLOAD=1" \
  -e "__GLX_VENDOR_LIBRARY_NAME=nvidia" \
  -v "$REPO_PATH_HOST:$REPO_PATH_CONTAINER:rw" \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:$CONTAINER_HOME/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:$CONTAINER_HOME/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:$CONTAINER_HOME/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:$CONTAINER_HOME/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:$CONTAINER_HOME/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:$CONTAINER_HOME/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:$CONTAINER_HOME/Documents:rw \
  --volume /dev:/dev \
  ${CONTAINER_REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG} \
  -c "cd $REPO_PATH_CONTAINER && \
  /isaac-sim/python.sh -m pip uninstall -y rl_for_curobo && \
  /isaac-sim/python.sh -m pip install -e $REPO_PATH_CONTAINER && \
  source /opt/ros/humble/setup.sh && \
  $DOCKER_CMD"



# chown -R $CONTAINER_USER:$CONTAINER_USER /workspace /workspaces /home/developer /isaac-sim && \
# su $CONTAINER_USER -c 




# Ensure cache directories exist
# mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache,logs,data,documents}

# echo "--------------------------------"
# echo "Running docker container..."
# echo "--------------------------------"
# echo "*Run variables:*"
# echo "IMAGE_NAME: $IMAGE_NAME"
# echo "IMAGE_TAG: $IMAGE_TAG"
# echo "CONTAINER_NAME: $CONTAINER_NAME"
# echo "DC_ENABLED: $DC_ENABLED"
# echo "DC_DEV_ID: $DC_DEV_ID"
# echo "REPO_PATH_CONTAINER: $REPO_PATH_CONTAINER"
# echo "DC_OPTIOCDNS: $DC_OPTIONS"
# echo ""
# echo "*Quick Start examples:*"
# echo "- isaac sim only:"
# echo "/isaac-sim/isaac-sim.sh"
# echo "- change cd to mounted repo:"
# echo " cd $REPO_PATH_CONTAINER"
# echo "- change cd to curobo original repo:"
# echo " cd /pkgs/curobo"
# echo "- MPC example:"
# echo "In mounted repo, run: cd $REPO_PATH_CONTAINER && omni_python $REPO_PATH_CONTAINER/projects/rl_for_curobo/examples/isaac_sim/mpc_example.py" # must make sure first the package rl_for_curobo is installed, for that run image tag v1 or above 
# echo "In original repo, run: omni_python /pkgs/curobo/examples/isaac_sim/mpc_example.py"
# echo ""
# echo "*Toolkit:*"
# echo "- Making a snapshot of the container (to save the state of the container):"
# echo "Step 1-RUN THIS SCRIPT: let this script (we are in) run in first terminal (you are here, so you are already done with that step)"
# echo "Step 2-MAKE CHANGES: if you need to make any changes to the cointaier-do them now...(installing packages, re-arrange dirs etc..)"
# echo "Step 3-COMMIT: After you are done, run this command in terminal 2 (also in host machine):"
# echo "General syntax: docker commit [OPTIONS] $CONTAINER_REGISTRY/$IMAGE_NAME:Your_new_tag_name_here"
# echo "In our case: docker commit -m 'describe the changes...' $CONTAINER_NAME $CONTAINER_REGISTRY/$IMAGE_NAME:Your_new_tag_name_here (an existing name is also possible)"
# echo "Step 4-PUSH: Push the new image to the registry (to back them up remotely):"
# echo "docker push $CONTAINER_REGISTRY/$IMAGE_NAME:Your_new_tag_name_here" 

# echo "--------------------------------"
# HI

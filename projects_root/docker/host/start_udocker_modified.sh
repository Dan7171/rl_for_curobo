#!/bin/bash
# Activate conda environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
source conda
conda activate curobo_env

input_arg="isaac_sim_4.0.0"
container_name="curobo_docker"

# Setup udocker
udocker setup --nvidia --force curobo_docker

# Check if container exists
if ! udocker ps | grep -q "$container_name"; then
    echo "Container $container_name not found. Please create it first."
    exit 1
fi

# Create cache directories if they don't exist
mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache,logs,data,documents}

if [[ "$input_arg" == *isaac_sim* ]]; then
    udocker run \
        --env="NVIDIA_DISABLE_REQUIRE=1" \
        --env="NVIDIA_DRIVER_CAPABILITIES=all" \
        --env="NVIDIA_VISIBLE_DEVICES=all" \
        --env="ACCEPT_EULA=Y" \
        --env="PRIVACY_CONSENT=Y" \
        --env="ISAAC_PATH=/isaac-sim" \
        --env="DISPLAY=$DISPLAY" \
        --env="XAUTHORITY=/host/.Xauthority" \
        --volume=$HOME/.Xauthority:/root/.Xauthority:rw \
        --volume=$HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
        --volume=$HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
        --volume=$HOME/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
        --volume=$HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
        --volume=$HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
        --volume=$HOME/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
        --volume=$HOME/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
        --volume=$HOME/docker/isaac-sim/documents:/root/Documents:rw \
        --volume=$HOME:/host \
        --volume=/dev:/dev \
        $container_name \
        bash -c "source /host/rl_for_curobo/projects_root/docker/guest/startup.sh && bash"
else
    echo "Unknown configuration"
    exit 1
fi
#!/bin/bash
# 
# Apptainer container run script for NVIDIA Isaac Sim
# Run by: ./start_apptainer_isaac_sim.sh #path_to_sif_file_without.sif# example: ./start_apptainer_isaac_sim.sh ~/apptainer_curobo.sif
# This manages to recognize gpu and display. Error is currently vulkan related.
# Usage: ./script.sh <tag>
#
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <tag>"
    exit 1
fi
mkdir -p ~/docker/isaac-sim/cache/{kit,ov,pip,glcache,computecache,logs,data,documents}
# mkdir -p /tmp/.X11-unix
# apptainer exec --nv \
# --nv : nvidia
# --no-eval : no evaluation
# --contain : contain the process # https://apptainer.org/docs/user/main/docker_and_oci.html#:~:text=Try%20running%20the%20container%20with%20the%20%2D%2Dcontain%20option%2C%20or%20the%20%2D%2Dcompat%20option%20(which%20is%20more%20strict).%20This%20disables%20the%20automatic%20mount%20of%20your%20home%20directory%2C%20which%20is%20a%20common%20source%20of%20issues%20where%20software%20in%20the%20container%20loads%20configuration%20or%20packages%20that%20may%20be%20present%20there.
# --compat : disable the automatic mount of your home directory
apptainer run --nv --no-eval --compat \
  --env ACCEPT_EULA=Y \
  --env PRIVACY_CONSENT=Y \
  --env DISPLAY=$DISPLAY \
  -B "$HOME/.Xauthority:/root/.Xauthority" \
  -B "$HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache" \
  -B "$HOME/docker/isaac-sim/cache/ov:/root/.cache/ov" \
  -B "$HOME/docker/isaac-sim/cache/pip:/root/.cache/pip" \
  -B "$HOME/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache" \
  -B "$HOME/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache" \
  -B "$HOME/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs" \
  -B "$HOME/docker/isaac-sim/data:/root/.local/share/ov/data" \
  -B "$HOME/docker/isaac-sim/documents:/root/Documents" \
  -B /dev:/dev \
  "$1".sif   


  # -c "source /host/rl_for_curobo/rl_module/docker/guest/startup.sh && bash"
# -B /tmp/.X11-unix:/tmp/.X11-unix \
  
  
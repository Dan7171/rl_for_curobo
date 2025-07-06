#!/bin/bash

echo "Fixing Isaac Sim cache from inside Docker container..."

# Clear Isaac Sim cache directories inside the container
echo "Clearing cache directories inside container..."

# Remove cache directories
rm -rf /isaac-sim/kit/cache/*
rm -rf /root/.cache/ov/*
rm -rf /root/.cache/pip/*
rm -rf /root/.cache/nvidia/GLCache/*
rm -rf /root/.nv/ComputeCache/*
rm -rf /root/.nvidia-omniverse/logs/*
rm -rf /root/.local/share/ov/data/*
rm -rf /root/Documents/*

# Also clear any potential cache in other locations
rm -rf /tmp/omni_cache
rm -rf /tmp/isaac_cache
rm -rf /tmp/.omni_cache

# Clear Python cache
find /isaac-sim -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /pkgs -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clear any lock files
find /isaac-sim -name "*.lock" -delete 2>/dev/null || true
find /root/.cache -name "*.lock" -delete 2>/dev/null || true

# Recreate empty directories
mkdir -p /isaac-sim/kit/cache
mkdir -p /root/.cache/ov
mkdir -p /root/.cache/pip
mkdir -p /root/.cache/nvidia/GLCache
mkdir -p /root/.nv/ComputeCache
mkdir -p /root/.nvidia-omniverse/logs
mkdir -p /root/.local/share/ov/data
mkdir -p /root/Documents

echo "Cache cleared inside container!"
echo ""
echo "Now try running your Isaac Sim example again:"
echo "omni_python /root/rl_for_curobo/projects_root/examples/mpc_example.py" 
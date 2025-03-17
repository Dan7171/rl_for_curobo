#!/bin/bash

# Running helpful commands on guest (container) before starting to run isaac-sim and curobo files

# Function to find the active display
find_active_display() {
    # First try to get the display from the host's environment
    if [ -n "$DISPLAY" ]; then
        echo "$DISPLAY"
        return 0
    fi
    
    # If not found, look for X11 sockets
    for socket in /tmp/.X11-unix/X*; do
        if [ -S "$socket" ]; then
            display_num=$(basename "$socket" | sed 's/^X//')
            echo ":$display_num"
            return 0
        fi
    done
    
    # If still not found, try common display numbers
    for i in {0..9}; do
        if [ -S "/tmp/.X11-unix/X$i" ]; then
            echo ":$i"
            return 0
        fi
    done
    
    # Default to :0 if nothing else is found
    echo ":0"
}

# Set up X11 permissions
xhost +local:root

# Find and set the active display
ACTIVE_DISPLAY=$(find_active_display)
export DISPLAY=$ACTIVE_DISPLAY

# Print debug information
echo "Setting up display configuration:"
echo "DISPLAY=$DISPLAY"
echo "XAUTHORITY=$XAUTHORITY"
ls -l /tmp/.X11-unix/

# Test X11 connection
if xhost >/dev/null 2>&1; then
    echo "X11 connection successful"
else
    echo "Warning: X11 connection failed"
fi

cp /host/rl_for_curobo/projects_root/docker/guest/nucleus.py /isaac-sim/exts/omni.isaac.nucleus/omni/isaac/nucleus/nucleus.py 
echo "- fast boot of isaac-sim 4.0.0: configured successfully V" # for more info see https://forums.developer.nvidia.com/t/detected-a-blocking-function-this-will-cause-hitches-or-hangs-in-the-ui-please-switch-to-the-async-version/271191/12#:~:text=I%20found%20a,function%20to%20below

echo "- ready to execute curobo simulations! (example: omni_python /core/pkgs/curobo/examples/isaac_sim/mpc_example.py)"


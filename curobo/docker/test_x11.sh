#!/bin/bash

echo "Testing X11 forwarding..."

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "ERROR: DISPLAY environment variable is not set"
    exit 1
fi

echo "DISPLAY=$DISPLAY"

# Check if X11 socket exists
if [ ! -S "/tmp/.X11-unix/X${DISPLAY#:}" ]; then
    echo "ERROR: X11 socket not found at /tmp/.X11-unix/X${DISPLAY#:}"
    exit 1
fi

echo "X11 socket found at /tmp/.X11-unix/X${DISPLAY#:}"

# Test X11 connection
if xhost >/dev/null 2>&1; then
    echo "SUCCESS: X11 connection is working"
else
    echo "ERROR: X11 connection failed"
    exit 1
fi

# Test if we can create a simple window
if command -v xterm >/dev/null 2>&1; then
    echo "Testing window creation with xterm..."
    timeout 3s xterm -e "echo 'X11 forwarding is working!' && sleep 2" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Window creation test passed"
    else
        echo "WARNING: Window creation test failed (this might be normal in some environments)"
    fi
else
    echo "xterm not available, skipping window creation test"
fi

echo "X11 forwarding test completed successfully!" 
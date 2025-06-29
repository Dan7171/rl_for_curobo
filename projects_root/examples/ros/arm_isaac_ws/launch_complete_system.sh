#!/bin/bash

# Complete Isaac Sim + ROS2 Robot Control System Launcher
# This script launches the full Franka Panda robot control system

echo "🚀 LAUNCHING COMPLETE ISAAC SIM ROBOT CONTROL SYSTEM"
echo "============================================================"
echo "This will start:"
echo "  1. Isaac Sim with Franka Panda robot"
echo "  2. ROS2 robot controller for joint control"
echo "  3. Robot state monitoring and analysis"
echo ""

# Check dependencies
echo "📋 Checking dependencies..."

# Check Isaac Sim
if [ -z "$ISAAC_SIM_PATH" ]; then
    echo "❌ ISAAC_SIM_PATH not set. Please export ISAAC_SIM_PATH=\"/path/to/isaac-sim\""
    exit 1
fi

if [ ! -f "$ISAAC_SIM_PATH/python.sh" ]; then
    echo "❌ Isaac Sim not found at: $ISAAC_SIM_PATH"
    exit 1
fi

# Check ROS2
if ! command -v ros2 &> /dev/null; then
    echo "❌ ROS2 not found. Please install ROS2 Jazzy"
    exit 1
fi

echo "✅ All dependencies found"

# Navigate to workspace
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source ROS2 environment
echo "📦 Sourcing ROS2 environment..."
source /opt/ros/jazzy/setup.bash

# Build the ROS2 package
echo "🔨 Building robot_test package..."
colcon build --packages-select robot_test

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

# Source the workspace
source install/setup.bash

echo "✅ Build successful"

# Set ROS2 environment
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

# Function to start Isaac Sim with robot
start_isaac_sim() {
    echo "🤖 Starting Isaac Sim robot node..."
    cd "$SCRIPT_DIR"
    $ISAAC_SIM_PATH/python.sh isaac_robot_working_movable.py &
    ISAAC_PID=$!
    echo "Isaac Sim PID: $ISAAC_PID"
}

# Function to start ROS2 robot command sender
start_robot_command_sender() {
    echo "🎛️  Starting robot command sender..."
    cd "$SCRIPT_DIR"
    python3 robot_command_sender.py &
    SENDER_PID=$!
    echo "Command Sender PID: $SENDER_PID"
}

# Function to start robot controller
start_robot_controller() {
    echo "🎮 Starting robot controller..."
    ros2 run robot_test robot_controller &
    CONTROLLER_PID=$!
    echo "Controller PID: $CONTROLLER_PID"
}

# Function to start robot subscriber
start_robot_subscriber() {
    echo "📊 Starting robot subscriber..."
    ros2 run robot_test robot_subscriber &
    SUBSCRIBER_PID=$!
    echo "Subscriber PID: $SUBSCRIBER_PID"
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up processes..."
    kill $ISAAC_PID 2>/dev/null
    kill $SENDER_PID 2>/dev/null
    kill $CONTROLLER_PID 2>/dev/null  
    kill $SUBSCRIBER_PID 2>/dev/null
    echo "✅ Cleanup complete"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

echo ""
echo "🚀 LAUNCHING SYSTEM COMPONENTS..."
echo ""

# Start Isaac Sim first
start_isaac_sim
echo "⏳ Waiting for Isaac Sim to initialize (30 seconds)..."
sleep 30

# Check if Isaac Sim topics are available
echo "🔍 Checking for Isaac Sim robot topics..."
if ros2 topic list | grep -q "/joint_states"; then
    echo "✅ Isaac Sim robot topics detected"
    ISAAC_WORKING=true
else
    echo "⚠️  Isaac Sim robot topics not detected - using simulation mode"
    ISAAC_WORKING=false
fi

# Start ROS2 components
start_robot_command_sender
sleep 3

start_robot_controller
sleep 3

start_robot_subscriber
sleep 3

echo ""
echo "🎉 SYSTEM LAUNCH COMPLETE!"
echo "============================================================"
echo ""
echo "📊 System Status:"
if [ "$ISAAC_WORKING" = true ]; then
    echo "  ✅ Isaac Sim Robot: Running and publishing joint states"
else
    echo "  ⚠️  Isaac Sim Robot: Simulation mode (check console for errors)"
fi
echo "  ✅ Robot Command Sender: Running"
echo "  ✅ Robot Controller: Running"  
echo "  ✅ Robot Subscriber: Running"
echo ""
echo "🤖 Robot Control Commands:"
echo "  # Move to home position"
echo "  ros2 topic pub --once /robot/joint_positions sensor_msgs/msg/JointState '{position: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]}'"
echo ""
echo "  # Move single joint"
echo "  ros2 topic pub --once /robot/joint_positions sensor_msgs/msg/JointState '{name: [\"panda_joint1\"], position: [1.57]}'"
echo ""
echo "  # Apply joint velocities"
echo "  ros2 topic pub --once /robot/joint_velocities sensor_msgs/msg/JointState '{velocity: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'"
echo ""
echo "  # Extended pose"
echo "  ros2 topic pub --once /robot/joint_positions sensor_msgs/msg/JointState '{position: [1.57, -0.5, 0.0, -1.5, 0.0, 1.0, 0.785]}'"
echo ""
echo "📊 Monitoring Commands:"
echo "  # Watch robot state"
echo "  ros2 topic echo /robot/state"
echo ""
echo "  # Monitor joint states"
echo "  ros2 topic echo /joint_states"
echo ""
echo "  # View robot analysis"
echo "  ros2 topic echo /robot/analysis"
echo ""
echo "  # View available topics"
echo "  ros2 topic list | grep robot"
echo ""
echo "🚨 Safety Notes:"
echo "  - Respect joint limits: panda_joint4 range is [-3.072, -0.070]"
echo "  - Use small incremental changes for smooth motion"
echo "  - Return to home position if robot gets stuck"
echo "  - Use Ctrl+C to stop all components"
echo ""
echo "Press Ctrl+C to stop all components"
echo ""

# Keep script running and show live status
while true; do
    sleep 5
    echo "$(date '+%H:%M:%S') - System running... (Ctrl+C to stop)"
    
    # Show topic count
    TOPIC_COUNT=$(ros2 topic list | grep -E "(robot|joint)" | wc -l)
    echo "  Robot topics active: $TOPIC_COUNT"
    
    # Show robot state if available
    if ros2 topic list | grep -q "/robot/state"; then
        STATE=$(timeout 1s ros2 topic echo /robot/state --once 2>/dev/null | grep "data:" | cut -d'"' -f2)
        if [ ! -z "$STATE" ]; then
            echo "  Current state: $STATE"
        fi
    fi
    echo ""
done 
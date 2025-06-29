# Workspace Structure - Isaac Sim Robot Control

This document outlines the structure and organization of the Isaac Sim Franka Panda robot control workspace.

## 📁 Directory Structure

```
arm_isaac_ws/
├── isaac_robot_working_movable.py          # Main Isaac Sim robot node
├── robot_command_sender.py                 # ROS2 to Isaac Sim bridge
├── launch_complete_system.sh               # System launcher script
├── README.md                               # Main documentation
├── WORKSPACE_STRUCTURE.md                  # This file
├── build/                                  # ROS2 build outputs (auto-generated)
├── install/                                # ROS2 install outputs (auto-generated)
├── log/                                    # ROS2 log files (auto-generated)
└── src/
    └── robot_test/                         # ROS2 package
        ├── package.xml                     # Package metadata
        ├── setup.py                        # Python package setup
        ├── setup.cfg                       # Setup configuration
        ├── resource/
        │   └── robot_test                  # Resource marker
        └── robot_test/                     # Python package
            ├── __init__.py                 # Package init
            ├── robot_controller.py         # Robot control interface
            └── robot_subscriber.py         # Robot state monitoring
```

## 🎯 Component Overview

### Core Components

1. **Isaac Sim Node** (`isaac_robot_working_movable.py`)
   - Loads Franka Panda robot in Isaac Sim
   - Creates ROS2 action graph for joint state publishing
   - Processes joint position commands from file
   - Provides visual feedback in Isaac Sim GUI

2. **Command Bridge** (`robot_command_sender.py`)
   - Subscribes to ROS2 joint command topics
   - Converts ROS2 messages to JSON commands
   - Writes commands to file for Isaac Sim to read
   - Publishes robot status back to ROS2

3. **Robot Controller** (`robot_controller.py`)
   - Provides high-level robot control interface
   - Supports predefined poses and sequences
   - Handles joint position and velocity commands
   - Includes demonstration routines

4. **Robot Subscriber** (`robot_subscriber.py`)
   - Monitors robot joint states
   - Analyzes robot pose and joint limits
   - Provides end-effector position estimation
   - Publishes analysis data

### Communication Flow

```
ROS2 Topics → Command Bridge → JSON File → Isaac Sim Node → Robot Movement
     ↑                                            ↓
Robot Analysis ← Robot Subscriber ← Joint States ← Action Graph
```

## 🔧 Build System

The workspace uses ROS2's `colcon` build system:

- **Build**: `colcon build --packages-select robot_test`
- **Source**: `source install/setup.bash`
- **Clean**: `rm -rf build install log`

## 📊 Data Flow

### Input Commands
1. Joint Position Commands → `/robot/joint_positions`
2. Joint Velocity Commands → `/robot/joint_velocities`
3. Position Array Commands → `/robot/position_array`

### File Communication
- Commands written to: `/tmp/isaac_robot_commands.json`
- Isaac Sim reads and processes commands every frame
- File is deleted after processing

### Output Topics
1. Joint States → `/joint_states` (from Isaac Sim)
2. Robot Status → `/robot/state`
3. Current Joints → `/robot/current_joints`
4. Analysis Data → `/robot/analysis`

## 🚀 Launch Sequence

1. **Environment Setup**
   - Source ROS2 environment
   - Build robot_test package
   - Set environment variables

2. **Isaac Sim Launch**
   - Start Isaac Sim with robot scene
   - Load Franka Panda robot
   - Create ROS2 action graph
   - Wait for initialization (30s)

3. **ROS2 Components Launch**
   - Start command bridge
   - Start robot controller
   - Start robot subscriber
   - Begin monitoring loop

## 🎮 Control Interface

### ROS2 Topics Interface
- Standard sensor_msgs/JointState for joint commands
- std_msgs/Float64MultiArray for simple position arrays
- Real-time feedback through joint_states topic

### File-based Communication
- JSON format for command exchange
- Atomic write operations to prevent corruption
- Automatic cleanup after processing

## 🛡️ Safety Features

1. **Joint Limit Monitoring**
   - Real-time joint limit checking
   - Warnings when approaching limits
   - Joint range validation

2. **Pose Analysis**
   - Automatic pose classification
   - End-effector position tracking
   - Motion analysis and feedback

3. **Error Handling**
   - Graceful error recovery
   - Status monitoring and reporting
   - Safe shutdown procedures

## 🔍 Debugging and Monitoring

### Log Files
- Isaac Sim logs in console output
- ROS2 logs in `log/` directory
- Component-specific logging

### Status Monitoring
- Real-time system status display
- Topic availability checking
- Process health monitoring

### Troubleshooting Tools
- Topic list and echo commands
- System status display in launch script
- Automatic dependency checking

---

**Note**: This workspace structure is based on the working camera control example but adapted for robot joint control with the Franka Panda robot. 
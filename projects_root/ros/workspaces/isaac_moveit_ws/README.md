# QUICK REAME TLDR:
## MUST KNOW!
1. IN ANY CHANCE TO CODE UDNER projects_root/ros/workspaces/isaac_moveit_ws, YOU MUST RE-BUILD THE PACKAGES. DO THE FOLLOWING:

```bash
# AFTER MAKING CHANGES IN CODE...
echo $SHELL # make sure you are in bash
cd projects_root/ros/workspaces/isaac_moveit_ws
colcon build # build all packages in workspace
source install/setup.bash
```

## RUNNING:
1. RUN THE V3 IMAGE:
OPTION 1. FROM HOST-FIRST SET 

2. IN THE CONTAINER:

TERMINAL 1 - RVIZ AND MOVEIT

TODO: 
  1. MAKE CONFIG FILES FOR MORE ROBOTS AS IN https://www.youtube.com/watch?v=pGje2slp6-s&t=532s AND CHANGE launch_rviz_moveit CODE SO THAT IT TAKES NOT NECESSARILY UR5 CONFIG, BUT PASSING MOVEIT CONFIG FILES PATH/ROBOT NAME  

```bash
echo $SHELL # make sure you are in bash
source /opt/ros/hubmle/setup.bash # (or sh if not in bash)
cd /workspace/rl_for_curobo/projects_root/ros/workspaces/isaac_moveit_ws
source install/setup.bash # sourcing the workspace 
ros2 run isaac_moveit_main launch_rviz_moveit # starts moveit & rviz, with the moveit config created in moveit setup assistant for the particular robot you run for 
```

TERMINAL 2 - isaac sim:
TODO: 1. CURRENTLY GETTING FRANKA, NEED TO SUPPORT LOADING OF ANY ROBOT BY PARAMETER USING THE SAME URDF AS MOVEIT SETUP ASSISTANT ACCEPTS WHEN BUILDING THE MOVEIT CONFIG FILES.

```bash
su developer # isaac sim needs to run in user level
echo $SHELL # make sure you are in bash
source /opt/ros/hubmle/setup.bash # (or sh if not in bash)
cd /workspace/rl_for_curobo/projects_root/ros/workspaces/isaac_moveit_ws
source install/setup.bash # sourcing the workspace 

ros2 run isaac_moveit_main launch_isaac # running the launch isaac node (opening the omni graph that publishes clock (isaac sim time), the joint states (for rviz/moveit2) and subscribing to the joint state commands comming from moveit after planning directly (or indirectly by publishing from rviz, need to check that)
# Note: run /isaac-sim/isaac-sim/sh and open /workspaces/moveit2_UR5/ur5.usd manually, when running the ur5 example instead of running launch_isaac node)
```

TERMINAL 3 - moveit goal sender:
TODO: 1. MAKE CODE MORE NICE, CLEAN GARBAGE.   

```bash
echo $SHELL # make sure you are in bash
source /opt/ros/hubmle/setup.bash # (or sh if not in bash)
cd /workspace/rl_for_curobo/projects_root/ros/workspaces/isaac_moveit_ws
source install/setup.bash # sourcing the workspace 
ros2 run isaac_moveit_main launch_ # running the launch isaac node (opening the omni graph that publishes clock (isaac sim time), the joint states (for rviz/moveit2) and subscribing to the joint state commands comming from moveit after planning directly (or indirectly by publishing from rviz, need to check that)
```


# Isaac MoveIt Workspace

This ROS2 workspace contains packages for Isaac Sim and MoveIt integration.

## Workspace Structure

```
isaac_moveit_ws/
├── src/
│   ├── cfg_pkgs/
│   │   ├── other/          # Container for non-package files
│   │   └── pkgs/           # Container for configuration packages
│   │       └── ur5/        # UR5 robot configuration
│   └── code_pkgs/
│       └── isaac_moveit_main/  # Main package for Isaac Sim MoveIt integration
```

## Packages

### isaac_moveit_main

The main package containing:
- `launch_isaac.py`: Script to launch Isaac Sim with MoveIt integration
- `goal_sender.py`: Script to send goal poses to MoveIt for robot control
- `launch_rviz_moveit.py`: Script to launch RViz with MoveIt integration (replaces `ros2 launch ur5_moveit_config demo.launch.py`)

## Building the Workspace

```bash
# Navigate to the workspace
cd projects_root/ros/workspaces/isaac_moveit_ws

# Build the workspace
colcon build

# Source the workspace
source install/setup.bash
```

## Usage

### Launch Isaac Sim
```bash
ros2 run isaac_moveit_main launch_isaac
```

### Send Goals to MoveIt
```bash
ros2 run isaac_moveit_main goal_sender
```

### Launch RViz with MoveIt (Parameter-based)

The `launch_rviz_moveit.py` script now accepts parameters to work with any robot:

```bash
# For UR5 robot (default)
ros2 run isaac_moveit_main launch_rviz_moveit

# With explicit parameters
ros2 run isaac_moveit_main launch_rviz_moveit --ros-args \
  -p robot_name:=ur5 \
  -p config_dir:=src/cfg_pkgs/pkgs/ur5

# For other robots (example)
ros2 run isaac_moveit_main launch_rviz_moveit --ros-args \
  -p robot_name:=franka \
  -p config_dir:=src/cfg_pkgs/pkgs/franka

# With explicit file paths
ros2 run isaac_moveit_main launch_rviz_moveit --ros-args \
  -p robot_name:=ur5 \
  -p config_dir:=src/cfg_pkgs/pkgs/ur5 \
  -p urdf_file:=path/to/robot.urdf \
  -p rviz_config:=path/to/moveit.rviz
```

#### Parameters:
- `robot_name` (default: "ur5"): Name of the robot (e.g., ur5, franka, panda)
- `config_dir` (default: "src/cfg_pkgs/pkgs/ur5"): Directory containing robot configuration files
- `urdf_file` (default: ""): Path to URDF file (if empty, will search in config_dir)
- `rviz_config` (default: ""): Path to RViz config file (if empty, will search in config_dir)

This script reproduces the functionality of:
```bash
ros2 launch ur5_moveit_config demo.launch.py
```

It launches:
1. Robot State Publisher (RSP)
2. MoveIt MoveGroup
3. RViz with MoveIt plugin
4. Static transforms

## Robot Configurations

Robot configurations are stored in `src/cfg_pkgs/pkgs/` organized by robot type:

### UR5 Configuration
- **Location**: `src/cfg_pkgs/pkgs/ur5/`
- **Contents**:
  - `moveit_config/`: MoveIt configuration files
  - `robo_description/`: Robot description files (URDF, meshes, etc.)

## Dependencies

- ROS2 (Humble or later)
- Isaac Sim
- MoveIt2
- Python packages: numpy, rclpy, geometry_msgs, sensor_msgs, visualization_msgs, moveit_msgs
- Additional packages: tf2_ros, robot_state_publisher, moveit_ros_move_group, rviz2

## Notes

The `launch_rviz_moveit.py` script is designed to work with any robot configuration. It automatically searches for URDF and RViz configuration files in the specified configuration directory and falls back to common locations if not found. 
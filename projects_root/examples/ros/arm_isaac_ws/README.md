# Isaac Sim ROS2 Franka Panda Robot Control System

A complete Isaac Sim + ROS2 integration for real-time Franka Panda robot joint control.

## 🎯 Features

- **Real-time Joint Control**: Move robot joints in Isaac Sim via ROS2 commands
- **Live Joint State Streaming**: Joint positions published to ROS2
- **Multiple Control Modes**: Position control, velocity control, and predefined poses
- **Visual Feedback**: See robot movement in real-time within Isaac Sim interface

## 🚀 Quick Start

### Prerequisites
- Isaac Sim 5.0+ installed
- ROS2 Jazzy
- Python 3.11+ (for Isaac Sim compatibility)

### Launch the System

1. **Start Isaac Sim Robot Node**:
```bash
conda activate isaac-sim 
echo "make sure IS_EXE (see conda .activate.d) is set to the path of the dir contains isaac sim executbale: IS_EXE=$IS_EXE" 
echo  
ISAAC_SIM_PATH=$IS_EXE # set it 
cd projects_root/examples/ros/arm_isaac_ws
source /opt/ros/jazzy/setup.bash
$ISAAC_SIM_PATH/python.sh isaac_robot_working_movable.py 
# or (in conda env) currently unavailable (beacuse of ros bridge and python version issues. bridge needs py 3.12 and codna has 3.11):
# python isaac_robot_working_movable.py
```

2. **Start ROS2 Robot Command Bridge** (separate terminal):
```bash
cd projects_root/examples/ros/arm_isaac_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_test
source install/setup.bash
python3 robot_command_sender.py
```

## 🎮 Robot Control Commands

### Joint Position Control
```bash
# Move single joint
ros2 topic pub --once /robot/joint_positions sensor_msgs/msg/JointState '{name: ["panda_joint1"], position: [1.57]}'

# Move all joints (home pose)
ros2 topic pub --once /robot/joint_positions sensor_msgs/msg/JointState '{position: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]}'
```

### Joint Velocity Control
```bash
# Apply velocities to joints
ros2 topic pub --once /robot/joint_velocities sensor_msgs/msg/JointState '{velocity: [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}'
```

## 📊 ROS2 Topics

### Published Topics
- `/joint_states` - Robot joint states (from Isaac Sim)
- `/robot/current_joints` - Current joint positions
- `/robot/state` - Robot status information

### Subscribed Topics
- `/robot/joint_positions` - Joint position commands
- `/robot/joint_velocities` - Joint velocity commands
- `/robot/position_array` - Position array commands

## 🤖 Franka Panda Joints

| Joint | Name | Range (rad) |
|-------|------|-------------|
| 1 | panda_joint1 | [-2.897, 2.897] |
| 2 | panda_joint2 | [-1.763, 1.763] |
| 3 | panda_joint3 | [-2.897, 2.897] |
| 4 | panda_joint4 | [-3.072, -0.070] |
| 5 | panda_joint5 | [-2.897, 2.897] |
| 6 | panda_joint6 | [-0.018, 3.753] |
| 7 | panda_joint7 | [-2.897, 2.897] |

## 📺 Monitoring

```bash
# View topics
ros2 topic list | grep robot

# Monitor joint states
ros2 topic echo /joint_states

# Check robot status
ros2 topic echo /robot/state
```

## 🎯 Example Usage

### Home Position
```bash
ros2 topic pub --once /robot/joint_positions sensor_msgs/msg/JointState '{position: [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]}'
```

### Extended Position
```bash
ros2 topic pub --once /robot/joint_positions sensor_msgs/msg/JointState '{position: [1.57, -0.5, 0.0, -1.5, 0.0, 1.0, 0.785]}'
```

## 🛠️ Troubleshooting

- Ensure `ISAAC_SIM_PATH` is set correctly
- Check Isaac Sim logs for robot loading errors
- Verify ROS2 environment is sourced
- Test with home position first

---

**Status**: ✅ Functional ROS2 robot control system
**Robot**: Franka Panda (7 DOF) 
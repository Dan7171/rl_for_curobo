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
ISAAC_PY=/home/dan/isaacsim500/isaacsim/_build/linux-x86_64/release/python.sh # set it to isaac sim dir python.sh file
cd projects_root/examples/ros/arm_isaac_ws
source /opt/ros/jazzy/setup.bash
$ISAAC_PY isaac_robot_working_movable.py 
```
***HEALTH CHECK***
```bash

# check the /robot/* ros topics are active:
rostopic list
# should see:
# /parameter_events
# /robot/joint_command
# /robot/joint_states
# /rosout

# check if /robot/joint_states topic works
ros2 topic echo /robot/joint_states 
# should see (many messages like:):
# header:
#   stamp:
#     sec: 0
#     nanosec: 0
#   frame_id: ''
# name:
# - panda_joint1
# - panda_joint2
# - panda_joint3
# - panda_joint4
# - panda_joint5
# - panda_joint6
# - panda_joint7
# - panda_finger_joint1
# - panda_finger_joint2
# position:
# - 0.1064
# - 0.1084
# - 0.1079
# - -0.0698
# - 0.1081
# - 0.1078
# - 0.1083
# - 0.04
# - 0.04
# velocity:
# - 0.0167
# - 0.0114
# - 0.023
# - 0.0
# - 0.0199
# - 0.0206
# - 0.0191
# - 0.0029
# - 0.0001
# effort:
# - 0.0002
# - -9.8898
# - 0.0621
# - 0.1363
# - 0.0282
# - 0.8111
# - 0.0
# - -0.0091
# - 0.0091

# check if nodes (probably hidden) exist) 
ros2 node list --all
# should see:
# /_ros2cli_daemon_0_1b9d690a8d814305bbdf44fefa3e3494
# /robot/_ROS_Robot_publishJointState
# /robot/_ROS_Robot_subscribeJointState
# dan@de:~$ ros2 node list # will return an emty list (non hidden) 
# dan@de:~$ 

# check controller reacts to joint commands
# send position command:
ros2 topic pub --once /robot/joint_command sensor_msgs/msg/JointState '{name:["panda_joint1"], position:[0.5]}' # (can also try with other joints)
# should see ARM MOVING! + next messages in terminal:
# publisher: beginning loop
# publishing #1: sensor_msgs.msg.JointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id=''), name=['panda_joint1'], position=[0.5], velocity=[], effort=[])
# For more help see:
# ros2 topic pub --once /robot/joint_command --help

# send velocity command:
ros2 topic pub --once /robot/joint_command sensor_msgs/msg/JointState '{name:["panda_joint1"], velocity:[0.5]}'

# should see ARM MOVING! + next messages in terminal:
# publisher: beginning loop
# publishing #1: sensor_msgs.msg.JointState(header=std_msgs.msg.Header(stamp=builtin_interfaces.msg.Time(sec=0, nanosec=0), frame_id=''), name=['panda_joint1'], position=[], velocity=[0.5], effort=[])

```


2. **OPTIONAL: Start ROS2 Robot Command Bridge** (separate terminal):
```bash
cd projects_root/examples/ros/arm_isaac_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select robot_test
source install/setup.bash
python3 robot_command_sender.py # needs to use python 3.12 of ros python beacuse running rlcpy
```

## 🎮 Robot Control Commands

### Joint Position Control (single interface)
All joint commands use a single topic: **`/robot/joint_command`**.

```bash
# Move a single joint
ros2 topic pub --once /robot/joint_command sensor_msgs/msg/JointState '{name:["panda_joint1"], position:[0.5]}'

# Home pose – set all 7 joints
ros2 topic pub --once /robot/joint_command sensor_msgs/msg/JointState '{position:[0.0,-0.785,0.0,-2.356,0.0,1.571,0.785]}'

# Velocity control example (leave position empty)
ros2 topic pub --once /robot/joint_command sensor_msgs/msg/JointState '{velocity:[0.5,0,0,0,0,0,0]}'
```

## 🤝 Controlling the Robot

There are **two** equivalent ways to drive the Panda arm.  Choose the one that best fits your Python / ROS 2 environment.

### 1. Direct ROS 2 control (preferred – no extra helper script)
`isaac_robot_working_movable.py` already contains an OmniGraph `ROS2SubscribeJointState` node wired to the topic `/robot/joint_command`.  When you publish a `sensor_msgs/JointState` message on that topic the robot moves instantly inside Isaac Sim.

Example (one‐liner/home pose):
```bash
ros2 topic pub --once /robot/joint_command sensor_msgs/msg/JointState "{position:[0.0,-0.785,0.0,-2.356,0.0,1.571,0.785]}"
```

You **do not** need the helper script for this method – just make sure the Isaac Sim node is running and the ROS 2 environment (`source /opt/ros/jazzy/setup.bash`) is active in the terminal you use for `ros2 topic pub`.

### 2. Helper script `robot_command_sender.py` (file bridge)
If your workstation has Python version conflicts (e.g. Isaac Sim 3.11 vs. system ROS 2 3.12) you can keep Isaac Sim isolated and bridge the commands through a small JSON file.  The workflow is:

1. Run Isaac Sim robot node (same as before).
2. In **another** terminal run the command sender:
   ```bash
   cd projects_root/examples/ros/arm_isaac_ws
   source /opt/ros/jazzy/setup.bash
   python3 robot_command_sender.py
   ```
   You should see log lines similar to:
   ```
   [INFO] [robot_command_sender]: 🚀 Robot Command Sender started
   [INFO] [robot_command_sender]: 📂 Sending commands to: /tmp/isaac_robot_commands.json
   ```
3. Publish your joint targets on **`/robot/joint_positions`** (not `/robot/joint_command`).  For example:
   ```bash
   ros2 topic pub --once /robot/joint_positions sensor_msgs/msg/JointState '{name:["panda_joint1","panda_joint2"], position:[1.0, -0.5]}'
   ```
   The helper listens to that topic, appends the data to `/tmp/isaac_robot_commands.json`, and Isaac Sim will pick it up on the next simulation tick.

Behind the scenes the script:
* Subscribes to `sensor_msgs/JointState` on `/robot/joint_positions`.
* Each new message is wrapped in a JSON object and **appended** to `/tmp/isaac_robot_commands.json`.
* The simulation process periodically reads the file and feeds the values into an `IsaacArticulationController`.

ℹ️ **Tip:** If you ever need to inspect what is being sent, simply `cat /tmp/isaac_robot_commands.json` while the sim is running.

## 📊 ROS2 Topics (summary)

| Direction  | Topic                            | When it is used |
|------------|----------------------------------|-----------------|
| Publish ⇢ | `/joint_states`                   | Always (live feedback from Isaac Sim) |
| Subscribe ⇠ | `/robot/joint_command`           | Direct control path |
| Subscribe ⇠ | `/robot/joint_positions`         | File-bridge path via `robot_command_sender.py` |

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
ros2 topic pub --once /robot/joint_command sensor_msgs/msg/JointState '{position:[0.0,-0.785,0.0,-2.356,0.0,1.571,0.785]}'
```

### Extended Position
```bash
ros2 topic pub --once /robot/joint_command sensor_msgs/msg/JointState '{position:[1.57,-0.5,0.0,-1.5,0.0,1.0,0.785]}'
```

## 🛠️ Troubleshooting

- Ensure `ISAAC_SIM_PATH` is set correctly
- Check Isaac Sim logs for robot loading errors
- Verify ROS2 environment is sourced
- Test with home position first

---

**Status**: ✅ Functional ROS2 robot control system
**Robot**: Franka Panda (7 DOF) 
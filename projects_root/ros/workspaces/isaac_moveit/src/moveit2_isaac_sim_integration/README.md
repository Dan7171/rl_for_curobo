# MoveIt2 Isaac Sim Integration

A flexible ROS2 package that provides MoveIt2 motion planning integration with Isaac Sim for any robot configuration.

## Features

- **Flexible Robot Support**: Works with any robot by simply providing a URDF and configuration file
- **Multiple End Effectors**: Supports robots with multiple arms and end effectors
- **Dual Mode Operation**: 
  - RViz-based goal setting for interactive planning
  - Programmatic goal setting with automatic replanning
- **Isaac Sim Integration**: Seamless communication with Isaac Sim via ROS2 topics
- **Visual Target Indicators**: Shows target positions as colored cubes in the simulation
- **Automatic Replanning**: Detects target changes and replans accordingly
- **Configurable Planning**: Support for different MoveIt2 planners (RRT, RRTConnect, etc.)

## Requirements

- ROS2 Humble
- Isaac Sim 4.5+ (or 4.0+ with legacy support)
- MoveIt2
- Python 3.10
- NumPy
- PyYAML

## Installation

1. Clone this package into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
git clone <your-repo-url>
```

2. Install dependencies:
```bash
# Method 1: Use rosdep (now fixed)
cd moveit2_isaac_sim_integration
rosdep install --from-paths . --ignore-src -r -y

# Method 2: Use the provided installation script
bash install_dependencies.sh

# Method 3: Manual installation (see below)
```

3. Build the package:
```bash
cd this directory # optional - move it to your workspaces directory, like under "/workspaces"...
colcon build --packages-select moveit2_isaac_sim_integration
```

4. Source the workspace:
```bash
source install/setup.bash
```

## Configuration

### Robot Configuration

The package uses YAML configuration files to define robot properties. See `config/robot_config.yaml` for examples.

#### Basic Configuration Structure:
```yaml
your_robot:
  name: "your_robot"
  urdf_path: "/path/to/your/robot.urdf"
  
  isaac_sim:
    stage_path: "/YourRobot"
    usd_path: "/path/to/your/robot.usd"
    position: [0.0, 0.0, 0.0]
    orientation: [0.0, 0.0, 0.0, 0.0]
    
  moveit2:
    planning_group: "manipulator"
    
    end_effectors:
      - name: "end_effector"
        planning_group: "manipulator"
        link_name: "end_effector_link"
        reference_frame: "base_link"
        default_target_pose:
          position: [0.5, 0.0, 0.5]
          orientation: [0.0, 0.0, 0.0, 1.0]
          
    joint_names:
      - "joint_1"
      - "joint_2"
      # ... more joints
      
    home_position:
      - 0.0
      - 0.0
      # ... joint positions
      
  planning:
    default_planner: "RRTConnect"
    planning_time: 5.0
    
  simulation:
    target_change_interval: 100  # steps
    target_pose_tolerance: 0.01  # meters
```

### Multi-Arm Robot Example:
```yaml
dual_arm_robot:
  name: "dual_arm"
  
  moveit2:
    planning_group: "dual_arms"
    
    end_effectors:
      - name: "left_arm"
        planning_group: "left_arm_group"
        link_name: "left_end_effector"
        reference_frame: "base_link"
        default_target_pose:
          position: [0.5, 0.3, 0.5]
          orientation: [0.0, 0.0, 0.0, 1.0]
          
      - name: "right_arm"
        planning_group: "right_arm_group"  
        link_name: "right_end_effector"
        reference_frame: "base_link"
        default_target_pose:
          position: [0.5, -0.3, 0.5]
          orientation: [0.0, 0.0, 0.0, 1.0]
```

## Usage

### Method 1: Using the Main Script

#### Basic Usage (Programmatic Mode):
```bash
# Run with default Panda robot configuration
omni_python scripts/simulation_runner.py

# Run with custom robot configuration
omni_python scripts/simulation_runner.py --robot your_robot --config /path/to/config.yaml

# Run with specific planner
omni_python scripts/simulation_runner.py --planner RRTstar
```

#### With RViz (Interactive Mode):
```bash
# Run with RViz for interactive goal setting
omni_python scripts/simulation_runner.py --rviz

# Run with custom robot and RViz
omni_python scripts/simulation_runner.py --robot your_robot --rviz
```

### Method 2: Using Launch Files

```bash
# Launch with default settings
ros2 launch moveit2_isaac_sim_integration simulation_launch.py

# Launch with custom robot
ros2 launch moveit2_isaac_sim_integration simulation_launch.py robot:=your_robot

# Launch with RViz
ros2 launch moveit2_isaac_sim_integration simulation_launch.py use_rviz:=true

# Launch with specific planner
ros2 launch moveit2_isaac_sim_integration simulation_launch.py planner:=RRTstar
```

### Method 3: Individual Components

You can also run components separately:

#### 1. Start Isaac Sim:
```bash
omni_python scripts/start_sim.py --robot your_robot
```

#### 2. Start MoveIt2 Planning (in another terminal):
```bash
omni_python scripts/moveit2_utils.py
```

#### 3. Use the components in your own code:
```python
from moveit2_isaac_sim_integration.scripts.config_loader import ConfigLoader
from moveit2_isaac_sim_integration.scripts.moveit2_utils import MoveIt2Planner

# Load configuration
config_loader = ConfigLoader()
robot_config = config_loader.get_robot_config("your_robot")

# Create planner
planner = MoveIt2Planner(robot_config)

# Set target and plan
pose = create_pose_from_list([0.6, 0.1, 0.4])
planner.set_target_pose("end_effector", pose)
results = planner.plan_to_targets()

# Execute plan
if results:
    for result in results.values():
        planner.execute_plan(result)
```

## How It Works

### Architecture

The package consists of three main components:

1. **`start_sim.py`**: Launches Isaac Sim with the appropriate action graph for ROS2 communication
2. **`moveit2_utils.py`**: Provides MoveIt2 planning interface that works with any robot configuration
3. **`simulation_runner.py`**: Orchestrates the entire system with automatic replanning

### Workflow

1. **Initialization**: 
   - Isaac Sim is started with the robot model
   - ROS2 action graph is created for joint state communication
   - MoveIt2 planner is initialized with the robot configuration
   - Target visualizations are created

2. **Planning Loop**:
   - Targets are set (either via RViz or programmatically)
   - MoveIt2 plans trajectories to reach targets
   - Trajectories are executed by sending joint commands to Isaac Sim
   - System monitors for target changes and replans as needed

3. **Automatic Replanning**:
   - Every N simulation steps, a random target is generated
   - If targets change beyond tolerance, replanning is triggered
   - New trajectories are planned and executed

### ROS2 Topics

The package uses the following ROS2 topics:

- `/isaac_joint_states`: Joint states from Isaac Sim
- `/isaac_joint_commands`: Joint commands to Isaac Sim
- `/panda_arm_controller/joint_trajectory`: MoveIt2 trajectory commands (optional)
- `/target_markers`: Visualization markers for RViz

## Advanced Usage

### Custom Planners

To use a different MoveIt2 planner:

```bash
omni_python scripts/simulation_runner.py --planner STOMP
```

Supported planners include:
- RRTConnect (default)
- RRT
- RRTstar
- STOMP
- Pilz Industrial Motion Planner
- OMPL planners

### Multiple End Effectors

The system automatically handles multiple end effectors. Each end effector gets:
- Its own target visualization
- Independent planning
- Coordinated execution

### Real Hardware Integration

**Note**: Real hardware support is planned but not yet implemented.

```bash
# TODO: Real hardware mode
omni_python scripts/simulation_runner.py --real-hardware
```

## Troubleshooting

### Common Issues

1. **Dependency installation fails**:
   - Try the installation script: `bash install_dependencies.sh`
   - For Docker: Install packages manually without `sudo`
   - If `moveit_commander` fails: This is optional, the package will work without it
    (update: now replaced by find_package(moveit_ros_move_group REQUIRED))

2. **Isaac Sim doesn't start**:
   - Ensure Isaac Sim is properly installed and in your PATH
   - Check that the robot USD file exists
   - Verify ROS2 domain ID matches

2. **MoveIt2 planning fails**:
   - Check that the URDF is valid
   - Verify joint limits are reasonable
   - Ensure planning group names match SRDF

3. **No joint states received**:
   - Verify Isaac Sim action graph is running
   - Check ROS2 topic connections: `ros2 topic list`
   - Ensure ROS2 bridge is enabled in Isaac Sim

### Debug Mode

For debugging, run with verbose output:

```bash
omni_python scripts/simulation_runner.py --robot your_robot 2>&1 | tee debug.log
```

### Configuration Validation

Test your robot configuration:

```bash
omni_python scripts/config_loader.py
```

## Examples

### Example 1: Panda Robot with Random Targets

```bash
# Start with default Panda configuration
omni_python scripts/simulation_runner.py

# The system will:
# 1. Load Panda robot in Isaac Sim
# 2. Initialize MoveIt2 with Panda configuration
# 3. Set initial target position
# 4. Plan and execute trajectory
# 5. Change target every 100 steps and replan
```

### Example 2: Custom Robot with RViz

```bash
# Create your robot configuration in config/robot_config.yaml
# Then run:
omni_python scripts/simulation_runner.py --robot your_robot --rviz

# Use RViz interactive markers to set new target poses
```

### Example 3: Multi-Arm Robot

```bash
# Configure dual-arm robot in config/robot_config.yaml
omni_python scripts/simulation_runner.py --robot dual_arm_robot

# System will plan for both arms simultaneously
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This package is licensed under the Apache 2.0 License.

## Acknowledgments

- Based on the MoveIt2 tutorials and Isaac Sim integration examples
- Inspired by the PickNik Isaac Sim integration
- Uses the MoveIt2 planning framework

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed description

---

**Note**: This package is designed to be flexible and work with any robot configuration. The key is properly configuring the YAML file with your robot's specifications. 
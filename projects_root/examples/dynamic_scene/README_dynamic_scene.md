# Dynamic Scene Manager for Multi-Robot Systems

This system provides reusable components for creating dynamic multi-robot scenarios with changing targets, visual robot markers, and central obstacles.

## Features

✅ **Dynamic Targets**: Targets automatically change every 5-10 seconds  
✅ **Initial Central Pose**: All robots start with targets at the same center position  
✅ **Robot Color Markers**: Visual spheres/cubes on robots matching their target colors  
✅ **Central Obstacles**: Configurable bin or table that robots must avoid  
✅ **Modular Design**: Independent scene logic that can be reused across scripts  
✅ **Smart Target Placement**: Targets positioned around/inside the central obstacle area  

## Quick Start

### 1. Run the Demo

```bash
cd /path/to/your/workspace
python projects_root/examples/mpc_multirobot_dynamic_targets.py
```

This will start a 4-robot simulation with:
- Robots positioned around a central bin
- All targets initially at the center
- Automatic target updates every 5-10 seconds
- Visual color markers on each robot base
- Collision avoidance with the central bin

### 2. Customize the Scene

Edit the configuration at the top of `mpc_multirobot_dynamic_targets.py`:

```python
# Scene configuration
SCENE_TYPE = "bin"  # Change to "table" for flat surface
CENTER_POSITION = [0.0, 0.0, 0.2]  # Adjust center height
TARGET_UPDATE_INTERVAL = (5.0, 10.0)  # Change update frequency
MARKER_TYPE = "sphere"  # Change to "cube" for cubic markers

# Robot configuration  
input_args = ['franka', 'franka', 'ur5e', 'franka']  # Modify robot types
```

## Integration Guide

### Adding to Existing Scripts

1. **Import the scene manager:**
```python
from projects_root.utils.dynamic_scene_manager import DynamicSceneManager
```

2. **Initialize with your robot setup:**
```python
scene_manager = DynamicSceneManager(
    n_robots=len(robots),
    robot_positions=[robot.base_position for robot in robots],
    stage=my_world.stage,
    scene_type="bin",  # or "table"
    center_position=[0.0, 0.0, 0.2],
    target_area_radius=0.4,
    target_height_range=(0.3, 0.7),
    update_interval=(5.0, 10.0),
    marker_type="sphere"
)
```

3. **Define target update callback:**
```python
def update_robot_targets(updated_targets: dict):
    for robot_id, new_target_world in updated_targets.items():
        # Convert to robot frame and update target
        robot_pos = robot_positions[robot_id][:3]
        target_robot_frame = new_target_world - robot_pos
        robots[robot_id].update_target(target_robot_frame)

scene_manager.add_target_update_callback(update_robot_targets)
```

4. **Start the dynamic scene:**
```python
scene_manager.start_scene()
```

5. **Use in simulation loop:**
```python
while simulation_app.is_running():
    # Get current target and color for any robot
    target_pos, target_color = scene_manager.get_target_for_robot(robot_id)
    
    # Your existing simulation code...
    my_world.step(render=True)
```

6. **Clean up when done:**
```python
scene_manager.stop_scene()
```

## Architecture

### Core Components

1. **`DynamicTargetManager`**: Handles target positions and automatic updates
2. **`RobotColorMarker`**: Creates visual markers on robot bases
3. **`CentralObstacleManager`**: Generates obstacle configurations for bins/tables
4. **`DynamicSceneManager`**: Main coordinator that combines all components

### File Structure

```
projects_root/
├── utils/
│   └── dynamic_scene_manager.py     # Core scene management system
├── examples/
│   ├── mpc_multirobot_dynamic_targets.py    # Full working example
│   ├── dynamic_scene_usage_example.py       # Integration examples  
│   └── README_dynamic_scene.md              # This documentation
└── configs/
    └── dynamic_scene_obstacles.yml          # Obstacle configurations
```

## Configuration Options

### Scene Types

**Bin Mode (`scene_type="bin"`)**:
- Creates a rectangular bin with walls
- Targets can be placed inside or around the bin
- Robots must navigate around the walls
- Good for pick-and-place scenarios

**Table Mode (`scene_type="table"`)**:
- Creates a flat table surface
- Targets placed around the table edges
- Simpler obstacle for basic navigation
- Good for surface manipulation tasks

### Target Behavior

```python
# Target update frequency
update_interval=(5.0, 10.0)  # Random interval between 5-10 seconds

# Target placement area
target_area_radius=0.4       # Maximum distance from center
target_height_range=(0.3, 0.7)  # Min/max target height

# Number of targets that change each update
# (automatically random between 1 and n_robots//2)
```

### Visual Markers

```python
marker_type="sphere"  # Spherical markers (default)
marker_type="cube"    # Cubic markers
```

Robot markers are automatically colored to match their assigned target colors:
- Robot 0: Red target → Red marker
- Robot 1: Green target → Green marker  
- Robot 2: Blue target → Blue marker
- Robot 3: Yellow target → Yellow marker
- etc.

## Advanced Usage

### Manual Target Updates

```python
# Force immediate target update
updated = scene_manager.manual_target_update(num_targets=2)

# Get current target for specific robot
target_pos, target_color = scene_manager.get_target_for_robot(robot_id)

# Get all current targets
all_targets = scene_manager.target_manager.get_all_targets()
```

### Custom Obstacle Configurations

```python
from projects_root.utils.dynamic_scene_manager import CentralObstacleManager

# Create custom bin
bin_config = CentralObstacleManager.create_bin_config(
    center_pos=[0.0, 0.0, 0.1],
    bin_size=[0.6, 0.6, 0.4],  # Larger bin
    wall_thickness=0.03
)

# Create custom table
table_config = CentralObstacleManager.create_table_config(
    center_pos=[0.0, 0.0, 0.0],
    table_size=[0.8, 0.8, 0.08]  # Larger table
)
```

### Multiple Callbacks

```python
def callback1(targets):
    print("Callback 1: Targets updated!")
    
def callback2(targets):
    print("Callback 2: Logging target changes...")

scene_manager.add_target_update_callback(callback1)
scene_manager.add_target_update_callback(callback2)
```

## Troubleshooting

### Common Issues

1. **Robots not updating targets**:
   - Check that your callback function is properly registered
   - Ensure `update_target()` method exists on your robot objects
   - Verify coordinate frame conversions (world → robot frame)

2. **Visual markers not appearing**:
   - Ensure Isaac Sim stage is properly initialized before creating scene manager
   - Check that robot positions are valid numpy arrays
   - Verify USD path permissions

3. **Obstacles not loaded**:
   - Check obstacle configuration file path
   - Ensure `Obstacle` class is imported correctly
   - Verify world models are initialized before adding obstacles

### Debug Tips

```python
# Enable verbose logging
scene_manager.target_manager._callbacks.append(
    lambda targets: print(f"Debug: {len(targets)} targets updated")
)

# Check obstacle configurations
print("Loaded obstacles:")
for obs in scene_manager.obstacle_configs:
    print(f"  {obs['name']}: {obs['pose'][:3]}")

# Monitor target positions  
for i in range(n_robots):
    pos, color = scene_manager.get_target_for_robot(i)
    print(f"Robot {i}: target at {pos}, color {color}")
```

## Performance Notes

- Target updates run in background threads (non-blocking)
- Visual markers are created once at initialization
- Obstacle configurations are generated programmatically (no file I/O during runtime)
- Coordinate transformations use numpy for efficiency

## Examples

See the following files for complete examples:
- `mpc_multirobot_dynamic_targets.py` - Full Isaac Sim integration
- `dynamic_scene_usage_example.py` - Integration patterns and code snippets  
- `dynamic_scene_obstacles.yml` - Example obstacle configurations

## Contributing

To extend the system:
1. Add new obstacle types in `CentralObstacleManager`
2. Implement custom target update patterns in `DynamicTargetManager` 
3. Create new marker types in `RobotColorMarker`
4. Add scene-specific logic in `DynamicSceneManager` 
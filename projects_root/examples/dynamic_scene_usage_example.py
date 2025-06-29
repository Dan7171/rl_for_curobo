"""
Dynamic Scene Manager Usage Example

This demonstrates how to integrate the dynamic scene manager 
into existing multi-robot scripts.

Usage:
    # Import the scene manager
    from projects_root.utils.dynamic_scene_manager import DynamicSceneManager
    
    # Initialize in your main function
    scene_manager = DynamicSceneManager(...)
    
    # Add target update callback
    scene_manager.add_target_update_callback(your_callback_function)
    
    # Start the dynamic scene
    scene_manager.start_scene()
    
    # In simulation loop: get current targets
    target_pos, target_color = scene_manager.get_target_for_robot(robot_id)
    
    # Stop when done
    scene_manager.stop_scene()
"""

import numpy as np
from typing import List, Dict


def example_integration():
    """
    Example showing how to integrate dynamic scene manager
    into an existing multi-robot script.
    """
    
    # Example robot positions (replace with your actual robot positions)
    robot_positions = [
        np.array([-0.6, -0.3, 0]),  # Robot 0 position
        np.array([-0.6, 0.3, 0]),   # Robot 1 position
        np.array([0.6, -0.3, 0]),   # Robot 2 position
        np.array([0.6, 0.3, 0])     # Robot 3 position
    ]
    
    n_robots = len(robot_positions)
    
    # This would normally be your Isaac Sim stage
    stage = None  # Replace with: my_world.stage
    
    print(f"Example: Setting up dynamic scene for {n_robots} robots")
    
    # STEP 1: Initialize the scene manager
    # from projects_root.utils.dynamic_scene_manager import DynamicSceneManager
    
    # Uncomment when integrating into actual Isaac Sim script:
    # scene_manager = DynamicSceneManager(
    #     n_robots=n_robots,
    #     robot_positions=robot_positions,
    #     stage=stage,
    #     scene_type="bin",  # or "table"
    #     center_position=[0.0, 0.0, 0.2],
    #     target_area_radius=0.4,
    #     target_height_range=(0.3, 0.7),
    #     update_interval=(5.0, 10.0),
    #     marker_type="sphere"  # or "cube"
    # )
    
    # STEP 2: Define your target update callback
    def handle_target_updates(updated_targets: Dict[int, np.ndarray]):
        """
        This function is called whenever targets are updated.
        
        Args:
            updated_targets: Dict mapping robot_id -> new_target_position
        """
        print(f"Targets updated for robots: {list(updated_targets.keys())}")
        
        for robot_id, new_target_world in updated_targets.items():
            print(f"  Robot {robot_id}: new target at {new_target_world}")
            
            # Here you would update your robot's target:
            # 1. Convert world coordinates to robot frame if needed
            # 2. Call your robot's target update method
            # Example:
            # robot_pos = robot_positions[robot_id]
            # target_robot_frame = new_target_world - robot_pos
            # your_robots[robot_id].update_target(target_robot_frame)
    
    # STEP 3: Add the callback and start the scene
    # scene_manager.add_target_update_callback(handle_target_updates)
    # scene_manager.start_scene()
    
    # STEP 4: In your main simulation loop, you can:
    # - Get current target for any robot:
    #   target_pos, target_color = scene_manager.get_target_for_robot(robot_id)
    # - Manually trigger target updates:
    #   scene_manager.manual_target_update(num_targets=2)
    # - Get obstacle configurations for collision checking:
    #   obstacles = scene_manager.obstacle_configs
    
    print("Example setup complete!")
    
    # STEP 5: Stop the scene when simulation ends
    # scene_manager.stop_scene()


def example_obstacle_configs():
    """
    Example showing different obstacle configurations
    """
    from projects_root.utils.dynamic_scene_manager import CentralObstacleManager
    
    # Create bin configuration
    bin_obstacles = CentralObstacleManager.create_bin_config(
        center_pos=[0.0, 0.0, 0.2],
        bin_size=[0.4, 0.4, 0.3],
        wall_thickness=0.02
    )
    
    print("Bin configuration:")
    for obs in bin_obstacles:
        print(f"  {obs['name']}: {obs['pose'][:3]} with dims {obs['dims']}")
    
    # Create table configuration  
    table_obstacles = CentralObstacleManager.create_table_config(
        center_pos=[0.0, 0.0, 0.0],
        table_size=[0.6, 0.6, 0.05]
    )
    
    print("Table configuration:")
    for obs in table_obstacles:
        print(f"  {obs['name']}: {obs['pose'][:3]} with dims {obs['dims']}")


def example_customization():
    """
    Example showing how to customize the dynamic scene behavior
    """
    
    print("Customization options:")
    print("1. Scene Type:")
    print("   - 'bin': Creates a central bin with walls")
    print("   - 'table': Creates a flat table surface")
    
    print("2. Target Update Intervals:")
    print("   - update_interval=(5.0, 10.0): Update every 5-10 seconds")
    print("   - update_interval=(2.0, 3.0): More frequent updates")
    
    print("3. Target Area:")
    print("   - target_area_radius=0.4: Targets within 0.4m of center")
    print("   - target_height_range=(0.3, 0.7): Target height between 0.3-0.7m")
    
    print("4. Robot Markers:")
    print("   - marker_type='sphere': Spherical markers on robots")
    print("   - marker_type='cube': Cubic markers on robots")
    
    print("5. Manual Control:")
    print("   - Call scene_manager.manual_target_update() to force updates")
    print("   - Call scene_manager.stop_scene() to stop automatic updates")


if __name__ == "__main__":
    print("=== Dynamic Scene Manager Usage Examples ===")
    print()
    
    print("1. Basic Integration Example:")
    example_integration()
    print()
    
    print("2. Obstacle Configuration Examples:")
    example_obstacle_configs()
    print()
    
    print("3. Customization Options:")
    example_customization()
    print()
    
    print("=== Integration Steps Summary ===")
    print("1. Import: from projects_root.utils.dynamic_scene_manager import DynamicSceneManager")
    print("2. Initialize scene manager with robot positions and stage")
    print("3. Define target update callback function")
    print("4. Add callback and start scene")
    print("5. Use scene_manager.get_target_for_robot() in simulation loop")
    print("6. Stop scene when done")
    print()
    print("See mpc_multirobot_dynamic_targets.py for full working example!") 
"""
Main simulation runner for MoveIt2 Isaac Sim integration.
This script orchestrates the entire system with flexible robot configurations.
"""

import sys
import os
import time
import threading
import argparse
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))
from config_loader import ConfigLoader, RobotConfig
from moveit2_utils import MoveIt2Planner, create_pose_from_list, EndEffectorTarget

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# ROS2 message types
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

# Optional RViz integration
try:
    from visualization_msgs.msg import Marker, MarkerArray
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class SimulationRunner(Node):
    """
    Main simulation runner that orchestrates MoveIt2 planning with Isaac Sim.
    """
    
    def __init__(self, robot_config: RobotConfig, use_rviz: bool = False):
        """
        Initialize the simulation runner.
        
        Args:
            robot_config: Robot configuration object
            use_rviz: Whether to use RViz for goal setting
        """
        super().__init__('simulation_runner')
        
        self.robot_config = robot_config
        self.use_rviz = use_rviz
        self.callback_group = ReentrantCallbackGroup()
        
        # Simulation state
        self.simulation_step = 0
        self.isaac_sim_process = None
        self.isaac_sim_launcher = None
        self.running = False
        
        # Planning state
        self.planner = None
        self.planning_active = False
        self.last_replan_time = 0.0
        
        # Target management
        self.target_positions = {}
        self.target_change_interval = robot_config.get_target_change_interval()
        self.target_tolerance = robot_config.get_target_pose_tolerance()
        
        # RViz integration
        self.rviz_process = None
        if use_rviz:
            self._setup_rviz_integration()
            
        # Visualization
        if VISUALIZATION_AVAILABLE:
            self._setup_visualization()
            
        self.get_logger().info(f"Simulation runner initialized for robot: {robot_config.name}")
        
    def _setup_rviz_integration(self):
        """Setup RViz integration for goal setting."""
        # TODO: Implement RViz goal setting integration
        # This would involve subscribing to interactive markers or goal poses from RViz
        self.get_logger().info("RViz integration enabled (TODO: implement interactive markers)")
        
    def _setup_visualization(self):
        """Setup visualization publishers."""
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/target_markers',
            10
        )
        
        # Timer for publishing markers
        self.marker_timer = self.create_timer(
            0.1,  # 10 Hz
            self._publish_target_markers,
            callback_group=self.callback_group
        )
        
    def start_isaac_sim(self) -> bool:
        """
        Start Isaac Sim with the robot configuration.
        
        Returns:
            True if started successfully
        """
        try:
            self.get_logger().info("Starting Isaac Sim...")
            
            # Get the path to our start_sim.py script
            start_sim_script = Path(__file__).parent / "start_sim.py"
            
            # Build command
            cmd = [
                os.environ.get("omni_python", "omni_python"),
                str(start_sim_script),
                "--robot", "default_robot",  # Use the same robot config
                "--config", str(self.robot_config.config_file_path) if hasattr(self.robot_config, 'config_file_path') else None
            ]
            
            # Remove None values
            cmd = [c for c in cmd if c is not None]
            
            # Start Isaac Sim in a separate process
            self.isaac_sim_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give Isaac Sim time to start
            time.sleep(10)
            
            # Check if process is still running
            if self.isaac_sim_process.poll() is None:
                self.get_logger().info("Isaac Sim started successfully")
                return True
            else:
                self.get_logger().error("Isaac Sim failed to start")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error starting Isaac Sim: {e}")
            return False
            
    def start_moveit2_planner(self) -> bool:
        """
        Start the MoveIt2 planner.
        
        Returns:
            True if started successfully
        """
        try:
            self.get_logger().info("Starting MoveIt2 planner...")
            
            # Create planner
            self.planner = MoveIt2Planner(self.robot_config, 'moveit2_planner_internal')
            
            # Wait for joint states
            start_time = time.time()
            timeout = 30.0
            
            while self.planner.current_joint_states is None and (time.time() - start_time) < timeout:
                rclpy.spin_once(self.planner, timeout_sec=1.0)
                
            if self.planner.current_joint_states is None:
                self.get_logger().error("Failed to receive joint states from Isaac Sim")
                return False
                
            self.get_logger().info("MoveIt2 planner started successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error starting MoveIt2 planner: {e}")
            return False
            
    def start_rviz(self) -> bool:
        """
        Start RViz if enabled.
        
        Returns:
            True if started successfully
        """
        if not self.use_rviz:
            return True
            
        try:
            self.get_logger().info("Starting RViz...")
            
            # TODO: Launch RViz with appropriate config
            # For now, just print a message
            self.get_logger().info("RViz integration not yet implemented")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error starting RViz: {e}")
            return False
            
    def initialize_targets(self):
        """Initialize target positions for all end effectors."""
        end_effectors = self.robot_config.get_end_effectors()
        
        for ee_config in end_effectors:
            ee_name = ee_config['name']
            default_pose = ee_config.get('default_target_pose', {})
            position = default_pose.get('position', [0.5, 0.0, 0.5])
            
            self.target_positions[ee_name] = position
            
            # Set target in planner
            if self.planner:
                pose = create_pose_from_list(position)
                self.planner.set_target_pose(ee_name, pose)
                
        self.get_logger().info(f"Initialized {len(self.target_positions)} target positions")
        
    def update_target_position(self, ee_name: str, new_position: List[float]) -> bool:
        """
        Update target position for an end effector.
        
        Args:
            ee_name: End effector name
            new_position: New target position [x, y, z]
            
        Returns:
            True if updated successfully
        """
        try:
            # Store new position
            self.target_positions[ee_name] = new_position
            
            # Update planner target
            if self.planner:
                pose = create_pose_from_list(new_position)
                self.planner.set_target_pose(ee_name, pose)
                
            # Update Isaac Sim visualization (if available)
            if self.isaac_sim_launcher:
                target_name = f"target_{ee_name}"
                self.isaac_sim_launcher.update_target_position(target_name, new_position)
                
            self.get_logger().info(f"Updated target for {ee_name}: {new_position}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error updating target for {ee_name}: {e}")
            return False
            
    def generate_random_target(self, ee_name: str) -> List[float]:
        """
        Generate a random target position for an end effector.
        
        Args:
            ee_name: End effector name
            
        Returns:
            Random target position [x, y, z]
        """
        # Define workspace bounds (can be made configurable)
        x_range = [0.2, 0.8]
        y_range = [-0.5, 0.5]
        z_range = [0.2, 0.8]
        
        new_position = [
            random.uniform(x_range[0], x_range[1]),
            random.uniform(y_range[0], y_range[1]),
            random.uniform(z_range[0], z_range[1])
        ]
        
        return new_position
        
    def check_and_replan(self) -> bool:
        """
        Check if replanning is needed and execute it.
        
        Returns:
            True if replanning was performed
        """
        if not self.planner or not self.planner.is_planning_available():
            return False
            
        # Check if any targets have changed significantly
        replan_needed = False
        changed_targets = []
        
        for ee_name in self.target_positions.keys():
            if self.planner.has_target_changed(ee_name, self.target_tolerance):
                replan_needed = True
                changed_targets.append(ee_name)
                
        if not replan_needed:
            return False
            
        # Plan for changed targets
        self.get_logger().info(f"Replanning for targets: {changed_targets}")
        
        try:
            results = self.planner.plan_to_targets(changed_targets)
            
            if results:
                self.get_logger().info(f"Replanning successful for {len(results)} targets")
                
                # Execute plans
                for result in results.values():
                    success = self.planner.execute_plan(result, use_isaac_direct=True)
                    if success:
                        self.get_logger().info(f"Executing plan for {result['end_effector']}")
                    else:
                        self.get_logger().error(f"Failed to execute plan for {result['end_effector']}")
                        
                self.last_replan_time = time.time()
                return True
                
            else:
                self.get_logger().error("Replanning failed")
                return False
                
        except Exception as e:
            self.get_logger().error(f"Error during replanning: {e}")
            return False
            
    def simulation_step_callback(self):
        """Callback for each simulation step."""
        self.simulation_step += 1
        
        # Check if we should change targets
        if self.simulation_step % self.target_change_interval == 0:
            self._change_random_target()
            
        # Check if replanning is needed
        if not self.use_rviz:  # Only auto-replan if not using RViz
            self.check_and_replan()
            
    def _change_random_target(self):
        """Change a random target position."""
        if not self.target_positions:
            return
            
        # Select random end effector
        ee_names = list(self.target_positions.keys())
        ee_name = random.choice(ee_names)
        
        # Generate new random position
        new_position = self.generate_random_target(ee_name)
        
        # Update target
        self.update_target_position(ee_name, new_position)
        
        self.get_logger().info(f"Changed target for {ee_name} to {new_position}")
        
    def _publish_target_markers(self):
        """Publish visualization markers for targets."""
        if not VISUALIZATION_AVAILABLE or not self.target_positions:
            return
            
        marker_array = MarkerArray()
        
        for i, (ee_name, position) in enumerate(self.target_positions.items()):
            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "targets"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position.x = position[0]
            marker.pose.position.y = position[1]
            marker.pose.position.z = position[2]
            marker.pose.orientation.w = 1.0
            
            # Set size
            size = self.robot_config.get_target_visualization_size()
            marker.scale.x = size
            marker.scale.y = size
            marker.scale.z = size
            
            # Set color (different for each target)
            colors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
            color = colors[i % len(colors)]
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
            
        self.marker_pub.publish(marker_array)
        
    def run(self) -> bool:
        """
        Run the complete simulation.
        
        Returns:
            True if simulation ran successfully
        """
        try:
            self.get_logger().info("Starting complete simulation...")
            
            # Start Isaac Sim
            if not self.start_isaac_sim():
                self.get_logger().error("Failed to start Isaac Sim")
                return False
                
            # Start MoveIt2 planner
            if not self.start_moveit2_planner():
                self.get_logger().error("Failed to start MoveIt2 planner")
                return False
                
            # Start RViz if enabled
            if not self.start_rviz():
                self.get_logger().error("Failed to start RViz")
                return False
                
            # Initialize targets
            self.initialize_targets()
            
            # Go to home position
            if self.planner:
                self.planner.go_to_home_position()
                time.sleep(2.0)
                
            # Start simulation loop
            self.running = True
            self.get_logger().info("Simulation running. Press Ctrl+C to stop.")
            
            # Main simulation loop
            while self.running:
                try:
                    # Spin ROS2 nodes
                    rclpy.spin_once(self, timeout_sec=0.01)
                    if self.planner:
                        rclpy.spin_once(self.planner, timeout_sec=0.01)
                        
                    # Simulation step
                    self.simulation_step_callback()
                    
                    # Small delay to control loop rate
                    time.sleep(0.01)
                    
                except KeyboardInterrupt:
                    self.get_logger().info("Stopping simulation...")
                    self.running = False
                    break
                    
            return True
            
        except Exception as e:
            self.get_logger().error(f"Simulation error: {e}")
            return False
            
        finally:
            self.cleanup()
            
    def cleanup(self):
        """Clean up resources."""
        self.get_logger().info("Cleaning up...")
        
        self.running = False
        
        # Stop planner
        if self.planner:
            self.planner.cleanup()
            
        # Stop Isaac Sim
        if self.isaac_sim_process:
            self.isaac_sim_process.terminate()
            self.isaac_sim_process.wait()
            
        # Stop RViz
        if self.rviz_process:
            self.rviz_process.terminate()
            self.rviz_process.wait()
            
        self.get_logger().info("Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MoveIt2 Isaac Sim Integration Runner")
    parser.add_argument("--robot", "-r", default="default_robot",
                       help="Robot configuration name")
    parser.add_argument("--config", "-c", default=None,
                       help="Path to robot configuration file")
    parser.add_argument("--planner", "-p", default=None,
                       help="Motion planner name (overrides config)")
    parser.add_argument("--rviz", action="store_true",
                       help="Use RViz for goal setting")
    parser.add_argument("--headless", action="store_true",
                       help="Run Isaac Sim in headless mode")
    parser.add_argument("--real-hardware", action="store_true",
                       help="Send commands to real hardware (TODO)")
    
    args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init()
    
    try:
        # Load robot configuration
        print(f"Loading robot configuration: {args.robot}")
        config_loader = ConfigLoader(args.config)
        robot_config = config_loader.get_robot_config(args.robot)
        
        # Validate configuration
        if not config_loader.validate_config(args.robot):
            print("Error: Invalid robot configuration")
            return 1
            
        # Override planner if specified
        if args.planner:
            # This would require modifying the config object
            print(f"Using planner: {args.planner}")
            
        # Real hardware mode
        if args.real_hardware:
            print("Real hardware mode not yet implemented")
            return 1
            
        # Create and run simulation
        runner = SimulationRunner(robot_config, use_rviz=args.rviz)
        
        success = runner.run()
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main()) 
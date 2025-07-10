"""
MoveIt2 utilities for flexible robot planning.
Provides planning functionality for any robot configuration.
"""

import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add the current directory to the path so we can import config_loader
sys.path.append(str(Path(__file__).parent))
from config_loader import ConfigLoader, RobotConfig

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# ROS2 message types
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration

# MoveIt2 imports
try:
    import moveit_commander
    MOVEIT_COMMANDER_AVAILABLE = True
except ImportError:
    print("Warning: moveit_commander not available. Using alternative MoveIt2 interface.")
    MOVEIT_COMMANDER_AVAILABLE = False

try:
    from moveit_msgs.msg import RobotTrajectory, MoveItErrorCodes
    from moveit_msgs.srv import GetPlanningScene
    import moveit_msgs.msg
    MOVEIT_AVAILABLE = True
except ImportError:
    print("Warning: MoveIt2 not found. Some functionality will be limited.")
    MOVEIT_AVAILABLE = False

# TF2 imports
try:
    import tf2_ros
    import tf2_geometry_msgs
    TF2_AVAILABLE = True
except ImportError:
    print("Warning: TF2 not found. Some functionality will be limited.")
    TF2_AVAILABLE = False


class EndEffectorTarget:
    """Represents a target pose for an end effector."""
    
    def __init__(self, name: str, pose: Pose, reference_frame: str = "base_link"):
        self.name = name
        self.pose = pose
        self.reference_frame = reference_frame
        self.timestamp = time.time()
        
    def distance_to(self, other_pose: Pose) -> float:
        """Calculate distance to another pose."""
        dx = self.pose.position.x - other_pose.position.x
        dy = self.pose.position.y - other_pose.position.y
        dz = self.pose.position.z - other_pose.position.z
        return np.sqrt(dx*dx + dy*dy + dz*dz)
        
    def __str__(self):
        return f"Target({self.name}): pos=({self.pose.position.x:.3f}, {self.pose.position.y:.3f}, {self.pose.position.z:.3f})"


class MoveIt2Planner(Node):
    """
    MoveIt2 planning interface that works with any robot configuration.
    """
    
    def __init__(self, robot_config: RobotConfig, node_name: str = "moveit2_planner"):
        """
        Initialize the MoveIt2 planner.
        
        Args:
            robot_config: Robot configuration object
            node_name: ROS2 node name
        """
        super().__init__(node_name)
        
        if not MOVEIT_AVAILABLE:
            raise RuntimeError("MoveIt2 not available. Please install moveit2 packages.")
            
        if not MOVEIT_COMMANDER_AVAILABLE:
            self.get_logger().warn("moveit_commander not available. Some functionality may be limited.")
            
        self.robot_config = robot_config
        self.callback_group = ReentrantCallbackGroup()
        
        # Current state
        self.current_joint_states = None
        self.current_targets = {}
        self.previous_targets = {}
        
        # Planning state
        self.planning_in_progress = False
        self.last_plan_time = 0.0
        
        # Initialize MoveIt2
        self._initialize_moveit()
        
        # Setup ROS2 publishers and subscribers
        self._setup_ros_interface()
        
        # Setup TF2 if available
        if TF2_AVAILABLE:
            self._setup_tf2()
            
        self.get_logger().info(f"MoveIt2 planner initialized for robot: {robot_config.name}")
        
    def _initialize_moveit(self):
        """Initialize MoveIt2 commander and planning interface."""
        try:
            if not MOVEIT_COMMANDER_AVAILABLE:
                self.get_logger().error("moveit_commander not available. Cannot initialize MoveIt2 planning.")
                raise RuntimeError("moveit_commander required for planning")
                
            # Initialize moveit_commander
            moveit_commander.roscpp_initialize(sys.argv)
            
            # Create robot commander
            self.robot_commander = moveit_commander.RobotCommander()
            
            # Create scene interface
            self.scene = moveit_commander.PlanningSceneInterface()
            
            # Create move group interfaces for each end effector
            self.move_groups = {}
            self.end_effector_links = {}
            
            for ee_config in self.robot_config.get_end_effectors():
                group_name = ee_config.get('planning_group', 'manipulator')
                ee_name = ee_config['name']
                
                if group_name not in self.move_groups:
                    move_group = moveit_commander.MoveGroupCommander(group_name)
                    
                    # Configure planner
                    planner_name = self.robot_config.get_default_planner()
                    move_group.set_planner_id(planner_name)
                    
                    # Set planning parameters
                    planning_config = self.robot_config.get_planning_config()
                    move_group.set_planning_time(planning_config.get('planning_time', 5.0))
                    move_group.set_max_velocity_scaling_factor(planning_config.get('max_velocity_scaling_factor', 0.1))
                    move_group.set_max_acceleration_scaling_factor(planning_config.get('max_acceleration_scaling_factor', 0.1))
                    
                    self.move_groups[group_name] = move_group
                    
                # Store end effector link mapping
                self.end_effector_links[ee_name] = {
                    'link': ee_config['link_name'],
                    'group': group_name,
                    'reference_frame': ee_config.get('reference_frame', 'base_link')
                }
                
            # Initialize current targets with default poses
            self._initialize_default_targets()
            
            self.get_logger().info("MoveIt2 commander initialized successfully")
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize MoveIt2: {e}")
            raise
            
    def _initialize_default_targets(self):
        """Initialize targets with default poses from configuration."""
        for ee_config in self.robot_config.get_end_effectors():
            ee_name = ee_config['name']
            default_pose_config = ee_config.get('default_target_pose', {})
            
            # Create pose from config
            pose = Pose()
            pos = default_pose_config.get('position', [0.5, 0.0, 0.5])
            ori = default_pose_config.get('orientation', [0.0, 0.0, 0.0, 1.0])
            
            pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
            pose.orientation = Quaternion(x=ori[0], y=ori[1], z=ori[2], w=ori[3])
            
            reference_frame = ee_config.get('reference_frame', 'base_link')
            target = EndEffectorTarget(ee_name, pose, reference_frame)
            
            self.current_targets[ee_name] = target
            self.previous_targets[ee_name] = target
            
    def _setup_ros_interface(self):
        """Setup ROS2 publishers and subscribers."""
        topics = self.robot_config.get_ros2_topics()
        
        # Joint state subscriber (from Isaac Sim)
        self.joint_state_sub = self.create_subscription(
            JointState,
            topics.get('isaac_joint_states', '/isaac_joint_states'),
            self.joint_state_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Joint command publisher (to Isaac Sim)
        self.isaac_joint_cmd_pub = self.create_publisher(
            JointState,
            topics.get('isaac_joint_commands', '/isaac_joint_commands'),
            10
        )
        
        # Joint trajectory publisher (to MoveIt2 controller)
        self.joint_traj_pub = self.create_publisher(
            JointTrajectory,
            topics.get('moveit_joint_trajectory', '/joint_trajectory_controller/joint_trajectory'),
            10
        )
        
        self.get_logger().info("ROS2 interface setup complete")
        
    def _setup_tf2(self):
        """Setup TF2 buffer and listener."""
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
    def joint_state_callback(self, msg: JointState):
        """Callback for joint state updates."""
        self.current_joint_states = msg
        
    def set_target_pose(self, end_effector_name: str, pose: Pose, reference_frame: str = None):
        """
        Set target pose for an end effector.
        
        Args:
            end_effector_name: Name of the end effector
            pose: Target pose
            reference_frame: Reference frame (optional)
        """
        if end_effector_name not in self.end_effector_links:
            self.get_logger().error(f"Unknown end effector: {end_effector_name}")
            return False
            
        if reference_frame is None:
            reference_frame = self.end_effector_links[end_effector_name]['reference_frame']
            
        # Store previous target
        if end_effector_name in self.current_targets:
            self.previous_targets[end_effector_name] = self.current_targets[end_effector_name]
            
        # Set new target
        target = EndEffectorTarget(end_effector_name, pose, reference_frame)
        self.current_targets[end_effector_name] = target
        
        self.get_logger().info(f"Target set for {end_effector_name}: {target}")
        return True
        
    def get_target_pose(self, end_effector_name: str) -> Optional[EndEffectorTarget]:
        """Get current target pose for an end effector."""
        return self.current_targets.get(end_effector_name)
        
    def has_target_changed(self, end_effector_name: str, tolerance: float = None) -> bool:
        """
        Check if target has changed significantly.
        
        Args:
            end_effector_name: Name of the end effector
            tolerance: Distance tolerance (uses config default if None)
            
        Returns:
            True if target has changed beyond tolerance
        """
        if tolerance is None:
            tolerance = self.robot_config.get_target_pose_tolerance()
            
        if end_effector_name not in self.current_targets or end_effector_name not in self.previous_targets:
            return True
            
        current = self.current_targets[end_effector_name]
        previous = self.previous_targets[end_effector_name]
        
        distance = current.distance_to(previous.pose)
        return distance > tolerance
        
    def plan_to_targets(self, end_effector_names: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Plan trajectories to current targets.
        
        Args:
            end_effector_names: List of end effector names to plan for (all if None)
            
        Returns:
            Dictionary with planning results or None if planning failed
        """
        if self.planning_in_progress:
            self.get_logger().warn("Planning already in progress")
            return None
            
        if end_effector_names is None:
            end_effector_names = list(self.current_targets.keys())
            
        self.planning_in_progress = True
        self.last_plan_time = time.time()
        
        try:
            # Group end effectors by their planning groups
            group_targets = {}
            for ee_name in end_effector_names:
                if ee_name not in self.current_targets:
                    self.get_logger().error(f"No target set for end effector: {ee_name}")
                    continue
                    
                target = self.current_targets[ee_name]
                group_name = self.end_effector_links[ee_name]['group']
                
                if group_name not in group_targets:
                    group_targets[group_name] = []
                group_targets[group_name].append((ee_name, target))
                
            # Plan for each group
            results = {}
            for group_name, targets in group_targets.items():
                if group_name not in self.move_groups:
                    self.get_logger().error(f"No move group for: {group_name}")
                    continue
                    
                move_group = self.move_groups[group_name]
                
                # For now, handle single target per group
                # TODO: Implement multi-target planning
                if len(targets) == 1:
                    ee_name, target = targets[0]
                    result = self._plan_single_target(move_group, ee_name, target)
                    if result:
                        results[ee_name] = result
                else:
                    self.get_logger().warn(f"Multi-target planning not yet implemented for group: {group_name}")
                    
            return results if results else None
            
        except Exception as e:
            self.get_logger().error(f"Planning failed: {e}")
            return None
            
        finally:
            self.planning_in_progress = False
            
    def _plan_single_target(self, move_group, ee_name: str, target: EndEffectorTarget) -> Optional[Dict[str, Any]]:
        """Plan to a single target."""
        try:
            # Set target pose
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = target.reference_frame
            pose_stamped.header.stamp = self.get_clock().now().to_msg()
            pose_stamped.pose = target.pose
            
            move_group.set_pose_target(pose_stamped)
            
            # Plan
            self.get_logger().info(f"Planning for {ee_name}...")
            success, plan, planning_time, error_code = move_group.plan()
            
            if success:
                self.get_logger().info(f"Planning successful for {ee_name} (time: {planning_time:.2f}s)")
                return {
                    'end_effector': ee_name,
                    'plan': plan,
                    'planning_time': planning_time,
                    'target': target
                }
            else:
                self.get_logger().error(f"Planning failed for {ee_name}: {error_code}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"Error planning for {ee_name}: {e}")
            return None
            
    def execute_plan(self, plan_result: Dict[str, Any], use_isaac_direct: bool = True) -> bool:
        """
        Execute a planned trajectory.
        
        Args:
            plan_result: Result from plan_to_targets
            use_isaac_direct: If True, send commands directly to Isaac Sim
            
        Returns:
            True if execution started successfully
        """
        try:
            plan = plan_result['plan']
            ee_name = plan_result['end_effector']
            
            if use_isaac_direct:
                return self._execute_plan_isaac_direct(plan, ee_name)
            else:
                return self._execute_plan_moveit(plan, ee_name)
                
        except Exception as e:
            self.get_logger().error(f"Error executing plan: {e}")
            return False
            
    def _execute_plan_isaac_direct(self, plan: RobotTrajectory, ee_name: str) -> bool:
        """Execute plan by sending commands directly to Isaac Sim."""
        try:
            trajectory = plan.joint_trajectory
            
            # Send each trajectory point
            for i, point in enumerate(trajectory.points):
                # Create joint state message
                joint_msg = JointState()
                joint_msg.header.stamp = self.get_clock().now().to_msg()
                joint_msg.name = trajectory.joint_names
                joint_msg.position = list(point.positions)
                joint_msg.velocity = list(point.velocities) if point.velocities else [0.0] * len(point.positions)
                joint_msg.effort = list(point.efforts) if point.efforts else [0.0] * len(point.positions)
                
                # Send command
                self.isaac_joint_cmd_pub.publish(joint_msg)
                
                # Wait for next point
                if i < len(trajectory.points) - 1:
                    duration = point.time_from_start
                    sleep_time = duration.sec + duration.nanosec / 1e9
                    if i == 0:
                        sleep_time = 0.1  # Small delay for first point
                    time.sleep(sleep_time)
                    
            self.get_logger().info(f"Trajectory executed for {ee_name}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error executing trajectory directly: {e}")
            return False
            
    def _execute_plan_moveit(self, plan: RobotTrajectory, ee_name: str) -> bool:
        """Execute plan through MoveIt2 controller."""
        try:
            # Send trajectory to joint trajectory controller
            self.joint_traj_pub.publish(plan.joint_trajectory)
            self.get_logger().info(f"Trajectory sent to controller for {ee_name}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error sending trajectory to controller: {e}")
            return False
            
    def go_to_home_position(self) -> bool:
        """Move robot to home position."""
        try:
            home_position = self.robot_config.get_home_position()
            joint_names = self.robot_config.get_joint_names()
            
            if len(home_position) != len(joint_names):
                self.get_logger().error("Home position length doesn't match joint count")
                return False
                
            # Send home position directly to Isaac Sim
            joint_msg = JointState()
            joint_msg.header.stamp = self.get_clock().now().to_msg()
            joint_msg.name = joint_names
            joint_msg.position = home_position
            joint_msg.velocity = [0.0] * len(joint_names)
            joint_msg.effort = [0.0] * len(joint_names)
            
            self.isaac_joint_cmd_pub.publish(joint_msg)
            
            self.get_logger().info("Moving to home position")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error going to home position: {e}")
            return False
            
    def get_current_pose(self, end_effector_name: str) -> Optional[Pose]:
        """Get current pose of an end effector."""
        if end_effector_name not in self.end_effector_links:
            return None
            
        group_name = self.end_effector_links[end_effector_name]['group']
        if group_name not in self.move_groups:
            return None
            
        try:
            move_group = self.move_groups[group_name]
            current_pose = move_group.get_current_pose()
            return current_pose.pose
            
        except Exception as e:
            self.get_logger().error(f"Error getting current pose for {end_effector_name}: {e}")
            return None
            
    def is_planning_available(self) -> bool:
        """Check if planning is available."""
        return MOVEIT_AVAILABLE and not self.planning_in_progress
        
    def get_planning_time(self) -> float:
        """Get time since last planning attempt."""
        return time.time() - self.last_plan_time
        
    def cleanup(self):
        """Cleanup resources."""
        if MOVEIT_AVAILABLE and MOVEIT_COMMANDER_AVAILABLE:
            try:
                moveit_commander.roscpp_shutdown()
            except:
                pass
                
        self.get_logger().info("MoveIt2 planner cleanup complete")


def create_pose_from_list(position: List[float], orientation: List[float] = None) -> Pose:
    """
    Create a Pose message from position and orientation lists.
    
    Args:
        position: [x, y, z] position
        orientation: [x, y, z, w] quaternion (optional, defaults to identity)
        
    Returns:
        Pose message
    """
    pose = Pose()
    pose.position = Point(x=position[0], y=position[1], z=position[2])
    
    if orientation is None:
        orientation = [0.0, 0.0, 0.0, 1.0]
    pose.orientation = Quaternion(x=orientation[0], y=orientation[1], z=orientation[2], w=orientation[3])
    
    return pose


def main():
    """Test function for MoveIt2 planner."""
    rclpy.init()
    
    try:
        # Load default robot configuration
        config_loader = ConfigLoader()
        robot_config = config_loader.get_robot_config("default_robot")
        
        # Create planner
        planner = MoveIt2Planner(robot_config)
        
        # Test basic functionality
        print("Testing MoveIt2 planner...")
        
        # Wait for joint states
        print("Waiting for joint states...")
        while planner.current_joint_states is None:
            rclpy.spin_once(planner, timeout_sec=1.0)
            
        print("Joint states received")
        
        # Test target setting
        end_effectors = list(planner.current_targets.keys())
        if end_effectors:
            ee_name = end_effectors[0]
            test_pose = create_pose_from_list([0.6, 0.1, 0.4])
            planner.set_target_pose(ee_name, test_pose)
            
            print(f"Target set for {ee_name}")
            
            # Test planning
            print("Planning...")
            results = planner.plan_to_targets([ee_name])
            
            if results:
                print(f"Planning successful: {len(results)} result(s)")
                
                # Test execution
                for result in results.values():
                    print(f"Executing plan for {result['end_effector']}...")
                    planner.execute_plan(result, use_isaac_direct=True)
                    break
            else:
                print("Planning failed")
                
        # Keep node running
        print("Press Ctrl+C to stop")
        try:
            rclpy.spin(planner)
        except KeyboardInterrupt:
            pass
            
    except Exception as e:
        print(f"Error: {e}")
        
    finally:
        if 'planner' in locals():
            planner.cleanup()
        rclpy.shutdown()


if __name__ == "__main__":
    main() 
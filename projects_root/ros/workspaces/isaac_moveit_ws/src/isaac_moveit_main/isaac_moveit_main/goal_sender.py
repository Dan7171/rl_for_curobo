#!/usr/bin/env python3
"""
UR5 MoveIt Goal Pose Sender

This script sends goal poses to MoveIt for a UR5 robot arm. It can:
- Read current joint states from /joint_states topic
- Compute forward kinematics using UR5 DH parameters
- Send different types of goal poses (fixed, random, left, right, high, low)
- Visualize goals in RViz
- Select different planning algorithms
- Test mode for debugging joint states

Author: Assistant
Date: 2024
"""

import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
import random
import math
import numpy as np

from rclpy.action import ActionClient

# class SRDF:
#     def __init__(self, srdf_file):
#         self.srdf_file = srdf_file
#         self.joint_names = self.get_joint_names()
# s
class MoveGroupClient(Node):
    """
    A ROS2 node that sends goal poses to MoveIt for UR5 robot arm control.
    
    This class handles:
    - Joint state monitoring
    - Forward kinematics computation
    - Goal pose generation
    - MoveIt action communication
    - RViz visualization
    """
    
    def __init__(self):
        """Initialize the MoveGroupClient node with all necessary subscribers and publishers."""
        super().__init__('move_group_client')
        
        # Action client for MoveIt
        client_owning_node = self
        action_type = MoveGroup
        action_name = '/move_action'
        self._client = ActionClient(client_owning_node, action_type,action_name)
        
        # # Subscribe to joint states to get current robot state
        # self._joint_states_sub = self.create_subscription(
        #     JointState, 
        #     '/joint_states', 
        #     self._joint_states_callback, 
        #     10
        # )
     
        
        # # Try to subscribe to end-effector pose topic if available
        # self._ee_pose_sub = None
        # self._current_ee_pose = None
        # try:
        #     self._ee_pose_sub = self.create_subscription(
        #         PoseStamped,
        #         '/tf_ee_link',  # Common topic name for end-effector pose
        #         self._ee_pose_callback,
        #         10
        #     )
        #     self.get_logger().info('Subscribed to /tf_ee_link for end-effector pose')
        # except:
        #     self.get_logger().info('No /tf_ee_link topic found, will compute FK manually')
        
        # Publishers for RViz visualization
        # self._goal_marker_pub = self.create_publisher(Marker, '/goal_pose_marker', 10)
        # self._goal_pose_pub = self.create_publisher(PoseStamped, '/move_group/display_goal_pose', 10)
        
        # Store current joint states
        # self._current_joint_states = None
        # self._joint_states_received = False
        
        # Wait for action server to be ready
        self._client.wait_for_server()
        self.get_logger().info('Action server is ready')

    # def _joint_states_callback(self, msg):
    #     """
    #     Callback to receive current joint states from robot.
        
    #     Only logs significant changes to reduce noise in the console.
        
    #     Args:
    #         msg: JointState message containing current joint positions and velocities
    #     """
    #     self._current_joint_states = msg
        # self._joint_states_received = True
        
        # # Only log if joint states changed significantly (reduce noise)
        # if not hasattr(self, '_last_logged_joint_states'):
        #     self._last_logged_joint_states = None

        # if not hasattr(self, 'joint_names'):
        #     self.joint_names:list[str] = msg.name
        
        # if not hasattr(self, '_last_logged_joint_states'):
        # if self._last_logged_joint_states is not None:
        #     # Check if any joint changed significantly (>1cm threshold)
        #     significant_change = False
        #     for i, (name, pos) in enumerate(zip(msg.name, msg.position)):
        #         if i < len(self._last_logged_joint_states.position):
        #             if abs(pos - self._last_logged_joint_states.position[i]) > 0.01:
        #                 significant_change = True
        #                 break
            
        #     if not significant_change:
        #         return  # Skip logging if no significant change
        
        # Log only significant changes

        # self.get_logger().info(f'Joint states updated: {[f"{name}: {pos:.4f}" for name, pos in zip(msg.name, msg.position)]}')

    # def _ee_pose_callback(self, msg):
    #     """
    #     Callback to receive end-effector pose from TF topic.
        
    #     Args:
    #         msg: PoseStamped message containing end-effector pose
    #     """
    #     self._current_ee_pose = msg.pose
    #     self.get_logger().info(f'EE pose from TF: pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})')

    # def publish_goal_to_rviz(self, target_pose, goal_type):
    #     """
    #     Publish goal pose to RViz for visualization.
        
    #     Creates a colored arrow marker and publishes it to RViz topics.
    #     Different goal types get different colors for easy identification.
        
    #     Args:
    #         target_pose: Pose message containing the goal position and orientation
    #         goal_type: String indicating the type of goal (affects marker color)
    #     """
    #     # Create a marker for the goal pose
    #     marker = Marker()
    #     marker.header.frame_id = "world"
    #     marker.header.stamp = self.get_clock().now().to_msg()
    #     marker.ns = "goal_pose"
    #     marker.id = 0
    #     marker.type = Marker.ARROW
    #     marker.action = Marker.ADD
        
    #     # Set marker position and orientation
    #     marker.pose = target_pose
    #     marker.scale.x = 0.1  # Arrow length
    #     marker.scale.y = 0.02  # Arrow width
    #     marker.scale.z = 0.02  # Arrow height
        
    #     # Set color based on goal type
    #     if goal_type == 'fixed':
    #         marker.color.r = 1.0  # Red
    #         marker.color.g = 0.0
    #         marker.color.b = 0.0
    #     elif goal_type == 'random':
    #         marker.color.r = 0.0
    #         marker.color.g = 1.0  # Green
    #         marker.color.b = 0.0
    #     elif goal_type == 'left':
    #         marker.color.r = 0.0
    #         marker.color.g = 0.0
    #         marker.color.b = 1.0  # Blue
    #     elif goal_type == 'right':
    #         marker.color.r = 1.0
    #         marker.color.g = 1.0  # Yellow
    #         marker.color.b = 0.0
    #     else:
    #         marker.color.r = 1.0
    #         marker.color.g = 0.0
    #         marker.color.b = 1.0  # Magenta
        
    #     marker.color.a = 0.8  # Transparency
        
    #     # Publish marker
    #     self._goal_marker_pub.publish(marker)
        
    #     # Also publish as PoseStamped for MoveIt RViz plugin
    #     pose_stamped = PoseStamped()
    #     pose_stamped.header.frame_id = "world"
    #     pose_stamped.header.stamp = self.get_clock().now().to_msg()
    #     pose_stamped.pose = target_pose
        
    #     self._goal_pose_pub.publish(pose_stamped)
        
    #     self.get_logger().info(f'Published goal pose to RViz: {goal_type} goal at ({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f})')




    def get_goal_pose(self, goal_type='fixed'):
        """
        Get different goal poses based on type.
        
        Args:
            goal_type: String specifying the type of goal
                - 'fixed': Original fixed goal (0.5, 0.0, 0.5)
                - 'random': Random goal within safe workspace
                - 'left': Goal to the left side
                - 'right': Goal to the right side
                - 'high': High goal position
                - 'low': Low goal position
                
        Returns:
            Pose: Geometry_msgs/Pose message with target position and orientation
        """
        if goal_type == 'fixed':
            # Original fixed goal
            pose = Pose()
            pose.position.x = 0.5
            pose.position.y = 0.0
            pose.position.z = 0.5
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            return pose
        elif goal_type == 'random':
            # Random goal within safe workspace
            pose = Pose()
            pose.position.x = random.uniform(0.3, 0.7)
            pose.position.y = random.uniform(-0.3, 0.3)
            pose.position.z = random.uniform(0.3, 0.7)
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            return pose
        elif goal_type == 'left':
            # Goal to the left
            pose = Pose()
            pose.position.x = 0.4
            pose.position.y = 0.3
            pose.position.z = 0.5
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            return pose
        elif goal_type == 'right':
            # Goal to the right
            pose = Pose()
            pose.position.x = 0.4
            pose.position.y = -0.3
            pose.position.z = 0.5
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            return pose
        elif goal_type == 'high':
            # High goal
            pose = Pose()
            pose.position.x = 0.5
            pose.position.y = 0.0
            pose.position.z = 0.7
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            return pose
        elif goal_type == 'low':
            # Low goal
            pose = Pose()
            pose.position.x = 0.5
            pose.position.y = 0.0
            pose.position.z = 0.3
            pose.orientation.w = 1.0
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = 0.0
            return pose
        else:
            # Default to fixed goal
            return self.get_goal_pose('fixed')

    def get_available_planners(self):
        """
        Get list of available planners for the arm group.
        
        Returns:
            list: List of common MoveIt planner names
        """
        # Common MoveIt planners
        common_planners = [
            "RRTConnect",
            "RRT",
            "RRTstar", 
            "PRM",
            "PRMstar",
            "BKPIECE",
            "KPIECE",
            "BiTRRT",
            "FMT",
            "STRIDE",
            "CHOMP",
            "STOMP",
            "OMPL"
        ]
        
        print("=== Available Planners ===")
        for i, planner in enumerate(common_planners):
            print(f"{i}. {planner}")
        return common_planners

    def send_goal(self, goal_type='fixed', test_mode=False, planner_id="RRTConnect"):
        """
        Send goal pose to MoveIt for planning and execution.
        
        This is the main method that:
        1. Tests forward kinematics
        2. Waits for joint states
        3. Creates MoveIt goal message
        4. Sends goal to MoveIt
        5. Handles the response
        
        Args:
            goal_type: Type of goal pose ('fixed', 'random', 'left', 'right', 'high', 'low')
            test_mode: If True, only print joint states without sending goals
            planner_id: Name of the planning algorithm to use
        """
        # Test forward kinematics first
        # self.test_forward_kinematics()
        
        # Wait for current joint states
        # if not self.wait_for_joint_states():
        #     self.get_logger().error('Failed to get current joint states')
        #     return

        # # If in test mode, just print joint states and exit
        # if test_mode:
        #     if self._current_joint_states is None:
        #         self.get_logger().error('No joint states available in test mode')
        #         return
                
        #     self.get_logger().info('=== TEST MODE: Current Joint States ===')
        #     arm_joint_names = [
        #         'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        #         'wrist_2_joint', 'wrist_3_joint', 'wrist_1_joint'
        #     ]
            
        #     for joint_name in arm_joint_names:
        #         try:
        #             idx = self._current_joint_states.name.index(joint_name)
        #             pos = self._current_joint_states.position[idx] # radians
        #             pos_deg = pos * 180 / math.pi # degrees
        #             vel = self._current_joint_states.velocity[idx] if idx < len(self._current_joint_states.velocity) else 0.0 # radians/s
        #             vel_deg = vel * 180 / math.pi # degrees/s
        #             self.get_logger().info(f'{joint_name}: position (deg)={pos_deg:.4f} deg, velocity (deg/s)={vel_deg:.4f} deg/s')
        #         except (ValueError, IndexError):
        #             self.get_logger().warn(f'{joint_name}: not found in joint states')
            
        #     # Also print all available joint states
        #     self.get_logger().info('=== All Available Joint States ===')
        #     for i, name in enumerate(self._current_joint_states.name):
        #         pos = self._current_joint_states.position[i]
        #         vel = self._current_joint_states.velocity[i] if i < len(self._current_joint_states.velocity) else 0.0
        #         self.get_logger().info(f'{name}: position={pos:.4f}, velocity={vel:.4f}')
            
        #     return

        # Create MoveIt goal message
        goal_msg = MoveGroup.Goal()

        # Set planning group
        goal_msg.request.group_name = 'arm'

        # Set planner ID
        goal_msg.request.planner_id = planner_id
        self.get_logger().info(f'Using planner: {planner_id}')

        # Set planning time
        goal_msg.request.allowed_planning_time = 5.0

        # Get current joint state from /joint_states topic
        if self._current_joint_states is None:
            self.get_logger().error('No joint states available')
            return

        # Define joint names for the arm (excluding gripper)
        arm_joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'wrist_1_joint'
        ]

        # Extract current positions for arm joints only
        current_positions = []
        for joint_name in arm_joint_names:
            try:
                idx = self._current_joint_states.name.index(joint_name)
                current_positions.append(self._current_joint_states.position[idx])
            except ValueError:
                self.get_logger().error(f'Joint {joint_name} not found in joint states')
                return

        self.get_logger().info(f'Current arm joint positions: {[f"{name}: {pos:.4f}" for name, pos in zip(arm_joint_names, current_positions)]}')
        
        # Set start state to current robot state
        goal_msg.request.start_state.joint_state.name = arm_joint_names
        goal_msg.request.start_state.joint_state.position = current_positions
        goal_msg.request.start_state.is_diff = False #  If set to true, it indicates that the robot's start state is different from the current state

        # ===== GOAL POSITION DEFINITION =====
        # Get the target end-effector pose based on goal_type
        target_pose = self.get_goal_pose(goal_type)

        self.get_logger().info(f'Goal type: {goal_type}')
        self.get_logger().info(f'Goal end-effector pose: position=({target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f})')
        self.get_logger().info(f'Goal orientation: w={target_pose.orientation.w:.3f}, x={target_pose.orientation.x:.3f}, y={target_pose.orientation.y:.3f}, z={target_pose.orientation.z:.3f}')

        # Publish goal to RViz for visualization
        self.publish_goal_to_rviz(target_pose, goal_type)

        # ===== POSITION CONSTRAINT =====
        # Create a position constraint that defines where the end-effector should be
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = 'world'
        position_constraint.link_name = 'ee_link'  # End-effector link name

        # Define a small box around the target position (tolerance)
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]  # 1cm tolerance box

        position_constraint.constraint_region.primitives.append(box)
        position_constraint.constraint_region.primitive_poses.append(target_pose)
        position_constraint.weight = 1.0

        # ===== ORIENTATION CONSTRAINT =====
        # Create an orientation constraint to specify the desired end-effector orientation
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = 'world'
        orientation_constraint.link_name = 'ee_link'
        orientation_constraint.orientation = target_pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1  # ~5.7 degrees
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

        # Add both constraints to the goal
        constraints = Constraints()
        constraints.position_constraints.append(position_constraint)
        constraints.orientation_constraints.append(orientation_constraint)
        goal_msg.request.goal_constraints.append(constraints)

        # Planning options
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.planning_scene_diff.is_diff = True

        # Send and handle result
        self.get_logger().info(f'Sending goal to MoveIt using {planner_id} planner...')
        future = self._client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        self.get_logger().info(f"Planning result error code: {result.error_code.val}")
        if result.error_code.val == 1:
            self.get_logger().info('Planning successful')
        else:
            self.get_logger().error('Planning failed')


def main():
    """
    Main function to run the MoveGroupClient.
    
    This function:
    1. Initializes ROS2
    2. Creates the MoveGroupClient node
    3. Shows available planners
    4. Sends goals based on configuration
    5. Cleans up and shuts down
    """
    rclpy.init()
    node = MoveGroupClient()
    while True:
        select_mode = int(input("Select mode (1: test, 2: send goal): "))
        if select_mode in [1, 2]:
            break
        else:
            print("Invalid mode. Please select 1 or 2.")
    
    # Test mode: only print joint states, don't send goals
    test_mode = select_mode == 1  # Set to True to only print joint states
    if test_mode:
        node.send_goal('fixed', test_mode=True)

    else:
        # Show available planners
        input("Press Enter to show available planners")
        planner_names = node.get_available_planners()
        try:
            planner_index = int(input(f"Select planner (0, {len(planner_names) - 1}) or press enter to use default:0"))
        except ValueError:
            planner_index = 0
            print(f"Planner not selected. Using default planner: {planner_names[planner_index]}")
            
        if not planner_index in range(len(planner_names)):
            planner_index = 0
            print(f"Invalid planner index. Using default planner: {planner_names[planner_index]}")
        planner_name = planner_names[planner_index]
        print(f"Using planner: {planner_name}")
        
        # You can change the goal_type here to get different goals:
        # Options: 'fixed', 'random', 'left', 'right', 'high', 'low'
        goal_type = 'fixed'  # Change this to get different goals each time
        node.send_goal(goal_type, test_mode=False, planner_id=planner_name)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

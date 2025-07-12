import rclpy
from rclpy.node import Node
from moveit_msgs.action import MoveGroup
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint
from builtin_interfaces.msg import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
import random
import math
import numpy as np

from rclpy.action import ActionClient


class MoveGroupClient(Node):
    def __init__(self):
        super().__init__('move_group_client')
        self._client = ActionClient(self, MoveGroup, '/move_action')
        
        # Subscribe to joint states to get current robot state
        self._joint_states_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self._joint_states_callback, 
            10
        )
        
        # Try to subscribe to end-effector pose topic if available
        self._ee_pose_sub = None
        self._current_ee_pose = None
        try:
            self._ee_pose_sub = self.create_subscription(
                PoseStamped,
                '/tf_ee_link',  # Common topic name for end-effector pose
                self._ee_pose_callback,
                10
            )
            self.get_logger().info('Subscribed to /tf_ee_link for end-effector pose')
        except:
            self.get_logger().info('No /tf_ee_link topic found, will compute FK manually')
        
        # Store current joint states
        self._current_joint_states = None
        self._joint_states_received = False
        
        # Wait for action server
        self._client.wait_for_server()
        self.get_logger().info('Action server is ready')

    def _joint_states_callback(self, msg):
        """Callback to receive current joint states from robot"""
        self._current_joint_states = msg
        self._joint_states_received = True
        
        # Only log if joint states changed significantly (reduce noise)
        if not hasattr(self, '_last_logged_joint_states'):
            self._last_logged_joint_states = None
            
        if self._last_logged_joint_states is not None:
            # Check if any joint changed significantly
            significant_change = False
            for i, (name, pos) in enumerate(zip(msg.name, msg.position)):
                if i < len(self._last_logged_joint_states.position):
                    if abs(pos - self._last_logged_joint_states.position[i]) > 0.01:  # 1cm threshold
                        significant_change = True
                        break
            
            if not significant_change:
                return  # Skip logging if no significant change
        
        # Log only significant changes
        self.get_logger().info(f'Joint states updated: {[f"{name}: {pos:.4f}" for name, pos in zip(msg.name, msg.position)]}')
        self._last_logged_joint_states = msg

    def _ee_pose_callback(self, msg):
        """Callback to receive end-effector pose from TF"""
        self._current_ee_pose = msg.pose
        self.get_logger().info(f'EE pose from TF: pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, {msg.pose.position.z:.3f})')

    def wait_for_joint_states(self, timeout=5.0):
        """Wait for joint states to be received"""
        self.get_logger().info('Waiting for joint states...')
        start_time = self.get_clock().now()
        while not self._joint_states_received:
            if (self.get_clock().now() - start_time).nanoseconds > timeout * 1e9:
                self.get_logger().error('Timeout waiting for joint states')
                return False
            rclpy.spin_once(self, timeout_sec=0.1)
        return True

    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion (w, x, y, z)"""
        # Ensure R is a 3x3 matrix
        if R.shape != (3, 3):
            return [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
        
        # Method from http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = np.trace(R)
        
        if trace > 0:
            S = math.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
        
        # Normalize quaternion
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm > 0:
            w /= norm
            x /= norm
            y /= norm
            z /= norm
        
        return [w, x, y, z]

    def ur5_forward_kinematics(self, joint_angles):
        """
        Compute UR5 forward kinematics using DH parameters
        Based on UR5 DH parameters from the URDF:
        a = [0.00000, -0.42500, -0.39225,  0.00000,  0.00000,  0.0000]
        d = [0.089159,  0.00000,  0.00000,  0.10915,  0.09465,  0.0823]
        alpha = [ 1.570796327, 0, 0, 1.570796327, -1.570796327, 0 ]
        q_home_offset = [0, -1.570796327, 0, -1.570796327, 0, 0]
        """
        # UR5 DH parameters
        a = [0.00000, -0.42500, -0.39225, 0.00000, 0.00000, 0.0000]
        d = [0.089159, 0.00000, 0.00000, 0.10915, 0.09465, 0.0823]
        alpha = [1.570796327, 0, 0, 1.570796327, -1.570796327, 0]
        q_home_offset = [0, -1.570796327, 0, -1.570796327, 0, 0]
        
        # Apply home offset to joint angles
        q = [joint_angles[i] + q_home_offset[i] for i in range(6)]
        
        # Initialize transformation matrix
        T = np.eye(4)
        
        # Compute forward kinematics for each joint
        for i in range(6):
            # DH transformation matrix
            ct = math.cos(q[i])
            st = math.sin(q[i])
            ca = math.cos(alpha[i])
            sa = math.sin(alpha[i])
            
            # DH transformation matrix
            Ti = np.array([
                [ct, -st*ca, st*sa, a[i]*ct],
                [st, ct*ca, -ct*sa, a[i]*st],
                [0, sa, ca, d[i]],
                [0, 0, 0, 1]
            ])
            
            # Multiply with current transformation
            T = T @ Ti
        
        # Extract position and orientation
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        
        # Convert rotation matrix to quaternion
        orientation = self.rotation_matrix_to_quaternion(rotation_matrix)
        
        return position, orientation

    def get_current_end_effector_pose(self):
        """
        Get current end-effector pose either from TF topic or by computing FK
        Returns (position, orientation) or None if not available
        """
        # First try to get from TF topic
        if self._current_ee_pose is not None:
            pos = (self._current_ee_pose.position.x, 
                   self._current_ee_pose.position.y, 
                   self._current_ee_pose.position.z)
            quat = (self._current_ee_pose.orientation.w,
                   self._current_ee_pose.orientation.x,
                   self._current_ee_pose.orientation.y,
                   self._current_ee_pose.orientation.z)
            self.get_logger().info(f'Using EE pose from TF topic')
            return pos, quat
        
        # If no TF available, compute FK from joint states
        if self._current_joint_states is None:
            return None
            
        # Get joint positions for UR5 arm (excluding gripper)
        arm_joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_2_joint', 'wrist_3_joint', 'wrist_1_joint'
        ]
        
        try:
            joint_positions = []
            for joint_name in arm_joint_names:
                idx = self._current_joint_states.name.index(joint_name)
                joint_positions.append(self._current_joint_states.position[idx])
            
            # Compute forward kinematics
            position, orientation = self.ur5_forward_kinematics(joint_positions)
            self.get_logger().info(f'Computed EE pose using FK')
            return position, orientation
            
        except (ValueError, IndexError) as e:
            self.get_logger().error(f'Error computing FK: {e}')
            return None

    def is_at_goal(self, target_pose, position_tolerance=0.1):
        """
        Check if the end-effector is already at the goal pose
        Returns True if within tolerance, False otherwise
        """
        # Get current end-effector position
        current_ee = self.get_current_end_effector_pose()
        if current_ee is None:
            self.get_logger().warn('Could not get current end-effector pose')
            return False
            
        current_pos, current_quat = current_ee
        target_pos = (target_pose.position.x, target_pose.position.y, target_pose.position.z)
        
        # Calculate distance to target
        distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(current_pos, target_pos)))
        
        self.get_logger().info(f'Current EE position: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})')
        self.get_logger().info(f'Target EE position: ({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})')
        self.get_logger().info(f'Distance to target: {distance:.3f} m (tolerance: {position_tolerance:.3f} m)')
        
        if distance < position_tolerance:
            self.get_logger().info(f'End-effector is already at goal position (distance: {distance:.3f}m < {position_tolerance}m)')
            return True
            
        self.get_logger().info(f'End-effector is NOT at goal position (distance: {distance:.3f}m >= {position_tolerance}m)')
        return False

    def get_goal_pose(self, goal_type='fixed'):
        """
        Get different goal poses based on type
        goal_type options: 'fixed', 'random', 'left', 'right', 'high', 'low'
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

    def test_forward_kinematics(self):
        """Test forward kinematics with known joint angles"""
        # Test with home position (all joints at 0)
        home_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pos, quat = self.ur5_forward_kinematics(home_joints)
        self.get_logger().info(f'Home position FK test: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})')
        
        # Test with some non-zero angles
        test_joints = [0.5, -1.0, 0.5, 0.0, 0.0, 0.0]
        pos, quat = self.ur5_forward_kinematics(test_joints)
        self.get_logger().info(f'Test position FK: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f})')

    def send_goal(self, goal_type='fixed', test_mode=False):
        # Test forward kinematics first
        self.test_forward_kinematics()
        
        # Wait for current joint states
        if not self.wait_for_joint_states():
            self.get_logger().error('Failed to get current joint states')
            return

        # If in test mode, just print joint states and exit
        if test_mode:
            self.get_logger().info('=== TEST MODE: Current Joint States ===')
            arm_joint_names = [
                'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                'wrist_2_joint', 'wrist_3_joint', 'wrist_1_joint'
            ]
            
            for joint_name in arm_joint_names:
                try:
                    idx = self._current_joint_states.name.index(joint_name)
                    pos = self._current_joint_states.position[idx] # radians
                    pos_deg = pos * 180 / math.pi # degrees
                    vel = self._current_joint_states.velocity[idx] if idx < len(self._current_joint_states.velocity) else 0.0 # radians/s
                    vel_deg = vel * 180 / math.pi # degrees/s
                    self.get_logger().info(f'{joint_name}: position (deg)={pos_deg:.4f} deg, velocity (deg/s)={vel_deg:.4f} deg/s')
                except (ValueError, IndexError):
                    self.get_logger().warn(f'{joint_name}: not found in joint states')
            
            # Also print all available joint states
            self.get_logger().info('=== All Available Joint States ===')
            for i, name in enumerate(self._current_joint_states.name):
                pos = self._current_joint_states.position[i]
                vel = self._current_joint_states.velocity[i] if i < len(self._current_joint_states.velocity) else 0.0
                self.get_logger().info(f'{name}: position={pos:.4f}, velocity={vel:.4f}')
            
            return

        goal_msg = MoveGroup.Goal()

        # Set planning group
        goal_msg.request.group_name = 'arm'

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
        self.get_logger().info('Sending goal to MoveIt...')
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
    rclpy.init()
    node = MoveGroupClient()
    
    # Test mode: only print joint states, don't send goals
    test_mode = False  # Set to False to send actual goals
    
    if test_mode:
        node.send_goal('fixed', test_mode=True)
    else:
        # You can change the goal_type here to get different goals:
        # Options: 'fixed', 'random', 'left', 'right', 'high', 'low'
        goal_type = 'fixed'  # Change this to get different goals each time
        node.send_goal(goal_type, test_mode=False)
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

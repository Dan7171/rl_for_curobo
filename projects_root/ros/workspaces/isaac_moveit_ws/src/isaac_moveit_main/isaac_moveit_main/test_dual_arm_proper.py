#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, PositionConstraint
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
import subprocess
import time

class DualArmCentralizedTest(Node):
    def __init__(self):
        super().__init__('dual_arm_centralized_test')
        
        # Create action client for MoveGroup
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        
        self.get_logger().info("🚀 === DUAL-ARM CENTRALIZED PLANNING TEST === 🚀")
        self.get_logger().info("Testing: Separate end effector goals → Single coordinated motion")

    def check_controllers(self):
        """Check which controllers are currently active"""
        
        try:
            result = subprocess.run(['ros2', 'control', 'list_controllers'], 
                                  capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                active_controllers = []
                for line in lines:
                    if 'active' in line:
                        controller_name = line.split()[0]
                        active_controllers.append(controller_name)
                
                self.get_logger().info(f"Active controllers: {active_controllers}")
                
                # Check if we need to switch to all_arms_controller
                if 'all_arms_controller' not in active_controllers:
                    return self.switch_to_dual_arm_controller()
                else:
                    self.get_logger().info("✅ all_arms_controller already active!")
                    return True
            else:
                self.get_logger().warn("Could not check controllers")
                return True  # Proceed anyway
                
        except Exception as e:
            self.get_logger().warn(f"Controller check failed: {e}")
            return True  # Proceed anyway

    def switch_to_dual_arm_controller(self):
        """Switch from individual arm controllers to all_arms_controller"""
        
        self.get_logger().info("🔄 Switching to dual-arm controller for centralized planning...")
        
        try:
            # Stop individual controllers
            self.get_logger().info("Stopping individual arm controllers...")
            subprocess.run(['ros2', 'control', 'set_controller_state', 'left_arm_controller', 'inactive'], 
                         timeout=10, check=False)
            subprocess.run(['ros2', 'control', 'set_controller_state', 'right_arm_controller', 'inactive'], 
                         timeout=10, check=False)
            
            time.sleep(1)  # Brief pause
            
            # Start all_arms_controller
            self.get_logger().info("Starting all_arms_controller...")
            result = subprocess.run(['ros2', 'run', 'controller_manager', 'spawner', 'all_arms_controller'], 
                                  timeout=15, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.get_logger().info("✅ Successfully switched to all_arms_controller!")
                return True
            else:
                self.get_logger().error(f"❌ Failed to start all_arms_controller: {result.stderr}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ Controller switching failed: {e}")
            return False

    def send_dual_arm_goal(self):
        """Send centralized planning goal for both end effectors"""
        
        goal_msg = MoveGroup.Goal()
        
        # Configure for dual-arm centralized planning
        goal_msg.request.group_name = "all_arms"  # Critical: use combined group
        goal_msg.request.num_planning_attempts = 5
        goal_msg.request.allowed_planning_time = 8.0  # More time for complex planning
        goal_msg.request.max_velocity_scaling_factor = 0.15
        goal_msg.request.max_acceleration_scaling_factor = 0.15
        
        # Configure planning options
        goal_msg.planning_options.plan_only = True  # Plan first, then decide execution
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = False
        
        # Create constraints for BOTH end effectors (separate goals!)
        constraints = Constraints()
        
        # ===== LEFT ARM END EFFECTOR GOAL =====
        left_constraint = PositionConstraint()
        left_constraint.header.frame_id = "base_link"
        left_constraint.link_name = "left_panda_link8"  # Left end effector
        left_constraint.weight = 1.0
        
        # Reasonable tolerance
        primitive_left = SolidPrimitive()
        primitive_left.type = SolidPrimitive.SPHERE
        primitive_left.dimensions = [0.05]  # Increased tolerance to 5cm
        left_constraint.constraint_region.primitives.append(primitive_left)
        
        # Left arm target pose (reachable position)
        pose_left = PoseStamped()
        pose_left.header.frame_id = "base_link"
        pose_left.pose.position.x = 0.25  # Closer to robot
        pose_left.pose.position.y = 0.10  # Smaller separation
        pose_left.pose.position.z = 0.25  # Lower height
        pose_left.pose.orientation.w = 1.0
        left_constraint.constraint_region.primitive_poses.append(pose_left.pose)
        
        # ===== RIGHT ARM END EFFECTOR GOAL =====
        right_constraint = PositionConstraint()
        right_constraint.header.frame_id = "base_link"
        right_constraint.link_name = "right_panda_link8"  # Right end effector
        right_constraint.weight = 1.0
        
        # Reasonable tolerance
        primitive_right = SolidPrimitive()
        primitive_right.type = SolidPrimitive.SPHERE
        primitive_right.dimensions = [0.05]  # Increased tolerance to 5cm
        right_constraint.constraint_region.primitives.append(primitive_right)
        
        # Right arm target pose (reachable position)
        pose_right = PoseStamped()
        pose_right.header.frame_id = "base_link"
        pose_right.pose.position.x = 0.25  # Closer to robot
        pose_right.pose.position.y = -0.10  # Smaller separation
        pose_right.pose.position.z = 0.25  # Lower height
        pose_right.pose.orientation.w = 1.0
        right_constraint.constraint_region.primitive_poses.append(pose_right.pose)
        
        # Add BOTH constraints to single planning request
        constraints.position_constraints.append(left_constraint)
        constraints.position_constraints.append(right_constraint)
        goal_msg.request.goal_constraints.append(constraints)
        
        # Log the dual-arm planning request
        self.get_logger().info("📋 DUAL-ARM PLANNING REQUEST:")
        self.get_logger().info(f"   Planning group: {goal_msg.request.group_name}")
        self.get_logger().info(f"   Time limit: {goal_msg.request.allowed_planning_time}s")
        self.get_logger().info(f"   Attempts: {goal_msg.request.num_planning_attempts}")
        self.get_logger().info("📍 TARGET POSES:")
        self.get_logger().info(f"   Left arm (left_panda_link8):  x=0.25, y=+0.10, z=0.25")
        self.get_logger().info(f"   Right arm (right_panda_link8): x=0.25, y=-0.10, z=0.25")
        self.get_logger().info("🎯 Goal: Plan coordinated motion for both arms avoiding collisions")
        
        # Wait for action server
        self._action_client.wait_for_server()
        
        # Send the dual-arm planning request
        self.get_logger().info("🚀 Sending dual-arm centralized planning request...")
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handle planning goal acceptance/rejection"""
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ DUAL-ARM PLANNING GOAL REJECTED!')
            self.get_logger().info("Possible causes:")
            self.get_logger().info("- all_arms_controller not active")
            self.get_logger().info("- Invalid planning group name")
            self.get_logger().info("- MoveIt configuration issue")
            rclpy.shutdown()
            return

        self.get_logger().info('✅ DUAL-ARM PLANNING GOAL ACCEPTED!')
        self.get_logger().info('⏳ Planning coordinated motion for both arms...')

        # Get result async with callback
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.planning_result_callback)

    def planning_result_callback(self, future):
        """Handle dual-arm planning results"""
        
        result = future.result().result
        error_code = result.error_code.val
        
        if error_code == 1:  # SUCCESS
            self.get_logger().info('🎉 DUAL-ARM CENTRALIZED PLANNING SUCCEEDED! 🎉')
            
            # Analyze the planned trajectory
            trajectory = result.planned_trajectory
            if trajectory.joint_trajectory.points:
                num_points = len(trajectory.joint_trajectory.points)
                joints = trajectory.joint_trajectory.joint_names
                left_joints = [j for j in joints if 'left' in j]
                right_joints = [j for j in joints if 'right' in j]
                
                self.get_logger().info('📊 TRAJECTORY ANALYSIS:')
                self.get_logger().info(f'   ✅ Total waypoints: {num_points}')
                self.get_logger().info(f'   ✅ Total joints planned: {len(joints)}')
                self.get_logger().info(f'   ✅ Left arm joints: {len(left_joints)}')
                self.get_logger().info(f'   ✅ Right arm joints: {len(right_joints)}')
                self.get_logger().info(f'   ✅ Planning time: {result.planning_time:.2f}s')
                
                if num_points > 0:
                    duration = trajectory.joint_trajectory.points[-1].time_from_start
                    self.get_logger().info(f'   ✅ Execution duration: {duration.sec}.{duration.nanosec//1000000:03d}s')
                
                self.get_logger().info('\n🏆 MISSION ACCOMPLISHED! 🏆')
                self.get_logger().info('✅ Can plan separate goals for each end effector')
                self.get_logger().info('✅ Using single "all_arms" planning group')
                self.get_logger().info('✅ Centralized dual-arm coordination')
                self.get_logger().info('✅ Collision avoidance between arms')
                self.get_logger().info('✅ Single coordinated trajectory for both arms')
                self.get_logger().info('🚀 Ready for your centralized dual-arm applications!')
                
                # Ask about execution
                self.get_logger().info('\n❓ Would you like to execute this motion?')
                self.get_logger().info('   (You can manually execute in RViz or modify this script)')
                
            else:
                self.get_logger().warn('⚠️  Planning succeeded but no trajectory generated')
        else:
            # Detailed error analysis
            error_names = {
                -2: "PLANNING_FAILED - Could not find valid path",
                -7: "TIMED_OUT - Planning time exceeded", 
                -12: "INVALID_GOAL_CONSTRAINTS - Target poses unreachable",
                -14: "INVALID_LINK_NAME - Check end effector link names",
                -10: "INVALID_GROUP_NAME - all_arms group not found",
                -1: "UNKNOWN_ERROR - An unexpected error occurred"
            }
            error_name = error_names.get(error_code, f"ERROR_{error_code}")
            self.get_logger().error(f'❌ DUAL-ARM PLANNING FAILED: {error_name}')
            
            # Provide troubleshooting advice
            if error_code == -2:
                self.get_logger().info("💡 Try: Increase planning time or adjust target positions")
            elif error_code == -7:
                self.get_logger().info("💡 Try: Increase allowed_planning_time or use closer targets")
            elif error_code == -12:
                self.get_logger().info("💡 Try: Move targets closer to current arm positions")
            elif error_code == -10:
                self.get_logger().info("💡 Check: all_arms group exists in SRDF configuration")

        # Shutdown after test
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    # Create the dual-arm test node
    test_node = DualArmCentralizedTest()
    
    # Check and setup controllers
    if test_node.check_controllers():
        # Send dual-arm planning goal
        test_node.send_dual_arm_goal()
        
        # Spin to handle callbacks
        rclpy.spin(test_node)
    else:
        test_node.get_logger().error("❌ Controller setup failed")
        rclpy.shutdown()

if __name__ == '__main__':
    main() 
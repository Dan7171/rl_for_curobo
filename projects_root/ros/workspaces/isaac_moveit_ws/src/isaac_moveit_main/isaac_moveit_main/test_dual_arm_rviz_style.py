#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest, PlanningOptions, WorkspaceParameters,
    Constraints, PositionConstraint, OrientationConstraint
)
from geometry_msgs.msg import Pose, PoseStamped
from shape_msgs.msg import SolidPrimitive
import time
from rclpy.executors import MultiThreadedExecutor
import threading


class DualArmRVizStyleTest(Node):
    def __init__(self):
        super().__init__('dual_arm_rviz_style_test')
        
        # Create action client for MoveGroup
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        
        self.get_logger().info("🚀 === DUAL-ARM RVIZ-STYLE TEST === 🚀")
        self.get_logger().info("Mimicking RViz workflow: Individual EE goals → All arms planning")
        self.get_logger().info("💡 Assumes all controllers (left_arm, right_arm, all_arms) are active")

    def create_position_constraint(self, link_name, target_pose, tolerance=0.05):
        """Create a position constraint for an end effector"""
        constraint = PositionConstraint()
        constraint.header.frame_id = "world"
        constraint.link_name = link_name
        
        # Target position
        constraint.target_point_offset.x = target_pose.position.x
        constraint.target_point_offset.y = target_pose.position.y
        constraint.target_point_offset.z = target_pose.position.z
        
        # Tolerance sphere
        constraint.constraint_region.primitives = [SolidPrimitive()]
        constraint.constraint_region.primitives[0].type = SolidPrimitive.SPHERE
        constraint.constraint_region.primitives[0].dimensions = [tolerance]
        
        # Constraint region pose (centered at target)
        constraint.constraint_region.primitive_poses = [Pose()]
        constraint.constraint_region.primitive_poses[0].position.x = target_pose.position.x
        constraint.constraint_region.primitive_poses[0].position.y = target_pose.position.y
        constraint.constraint_region.primitive_poses[0].position.z = target_pose.position.z
        constraint.constraint_region.primitive_poses[0].orientation.w = 1.0
        
        constraint.weight = 1.0
        return constraint

    def run_test(self):
        try:
            # Wait for action server
            self.get_logger().info("⏳ Waiting for MoveGroup action server...")
            if not self._action_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("❌ MoveGroup action server not available")
                return False

            # Step 1: Define end effector goals (like positioning gizmos in RViz)
            self.get_logger().info("📍 STEP 1: Defining individual end effector goals...")
            
            # Left arm goal (similar to positioning gizmo in RViz)
            left_pose = Pose()
            left_pose.position.x = 0.35
            left_pose.position.y = 0.15  # +Y for left arm
            left_pose.position.z = 0.35
            left_pose.orientation.w = 1.0
            
            # Right arm goal (similar to positioning gizmo in RViz)
            right_pose = Pose()
            right_pose.position.x = 0.35
            right_pose.position.y = -0.15  # -Y for right arm 
            right_pose.position.z = 0.35
            right_pose.orientation.w = 1.0
            
            self.get_logger().info(f"   Left arm goal: x={left_pose.position.x}, y={left_pose.position.y}, z={left_pose.position.z}")
            self.get_logger().info(f"   Right arm goal: x={right_pose.position.x}, y={right_pose.position.y}, z={right_pose.position.z}")
            
            # Step 2: Create planning request for all_arms group (like switching to all_arms in RViz)
            self.get_logger().info("🎯 STEP 2: Creating all_arms planning request...")
            
            # Create motion plan request
            goal_msg = MoveGroup.Goal()
            goal_msg.request = MotionPlanRequest()
            goal_msg.request.group_name = "all_arms"
            goal_msg.request.num_planning_attempts = 3
            goal_msg.request.allowed_planning_time = 10.0
            goal_msg.request.planner_id = "RRTConnect"  # Works well for multi-arm
            
            # Workspace parameters
            goal_msg.request.workspace_parameters = WorkspaceParameters()
            goal_msg.request.workspace_parameters.header.frame_id = "world"
            goal_msg.request.workspace_parameters.min_corner.x = -1.0
            goal_msg.request.workspace_parameters.min_corner.y = -1.0
            goal_msg.request.workspace_parameters.min_corner.z = -1.0
            goal_msg.request.workspace_parameters.max_corner.x = 1.0
            goal_msg.request.workspace_parameters.max_corner.y = 1.0
            goal_msg.request.workspace_parameters.max_corner.z = 1.0
            
            # Create constraints for both end effectors
            constraints = Constraints()
            constraints.name = "dual_arm_end_effector_goals"
            
            # Add position constraints for both arms
            left_constraint = self.create_position_constraint("left_panda_link8", left_pose, tolerance=0.05)
            right_constraint = self.create_position_constraint("right_panda_link8", right_pose, tolerance=0.05)
            
            constraints.position_constraints = [left_constraint, right_constraint]
            goal_msg.request.goal_constraints = [constraints]
            
            # Planning options
            goal_msg.planning_options = PlanningOptions()
            goal_msg.planning_options.plan_only = False  # Plan and execute
            goal_msg.planning_options.look_around = False
            goal_msg.planning_options.look_around_attempts = 0
            goal_msg.planning_options.max_safe_execution_cost = 0.0
            goal_msg.planning_options.replan = False
            goal_msg.planning_options.replan_attempts = 0
            goal_msg.planning_options.replan_delay = 0.0
            
            self.get_logger().info("✅ Successfully created dual-arm planning request")
            
            # Step 3: Send planning request (like clicking "Plan & Execute" in RViz)
            self.get_logger().info("🚀 STEP 3: Sending dual-arm planning request...")
            self.get_logger().info("   This should generate a coordinated motion for both arms")
            self.get_logger().info("   Using existing active controllers (no switching needed)")
            
            # Send goal
            future = self._action_client.send_goal_async(goal_msg)
            
            # Wait for goal acceptance
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if not future.done():
                self.get_logger().error("❌ Goal sending timed out")
                return False
                
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("❌ Goal was rejected")
                return False
                
            self.get_logger().info("✅ DUAL-ARM PLANNING GOAL ACCEPTED!")
            self.get_logger().info("⏳ Planning and executing coordinated motion...")
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
            
            if not result_future.done():
                self.get_logger().error("❌ Planning/execution timed out")
                return False
                
            result = result_future.result()
            
            if result.result.error_code.val == 1:  # SUCCESS
                self.get_logger().info("✅ PLANNING AND EXECUTION SUCCESS!")
                self.get_logger().info("🎉 Dual-arm coordinated motion completed!")
                self.get_logger().info(f"   Planning time: {result.result.planning_time:.2f}s")
                return True
            else:
                self.get_logger().error(f"❌ PLANNING/EXECUTION FAILED: ERROR_{result.result.error_code.val}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception during test: {e}")
            return False


def main():
    rclpy.init()
    
    # Create executor and node
    executor = MultiThreadedExecutor()
    test_node = DualArmRVizStyleTest()
    executor.add_node(test_node)
    
    # Start executor in background thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    # Give time for initialization
    time.sleep(2.0)
    
    try:
        # Run the test
        success = test_node.run_test()
        
        if success:
            test_node.get_logger().info("🎊 === TEST COMPLETED SUCCESSFULLY === 🎊")
        else:
            test_node.get_logger().error("💥 === TEST FAILED === 💥")
            
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        # Cleanup
        test_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
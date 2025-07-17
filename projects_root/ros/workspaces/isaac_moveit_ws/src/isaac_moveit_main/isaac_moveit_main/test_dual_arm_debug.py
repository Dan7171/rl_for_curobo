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


class DualArmDebugTest(Node):
    def __init__(self):
        super().__init__('dual_arm_debug_test')
        
        # Create action client for MoveGroup
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        
        self.get_logger().info("🔍 === DUAL-ARM DEBUG TEST === 🔍")

    def test_planning_only(self, group_name, target_poses, frame_id="panda_link0"):
        """Test planning only (no execution) with given parameters"""
        try:
            self.get_logger().info(f"🧪 Testing planning for group: {group_name}")
            self.get_logger().info(f"   Frame ID: {frame_id}")
            
            # Wait for action server
            if not self._action_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("❌ MoveGroup action server not available")
                return False

            # Create motion plan request
            goal_msg = MoveGroup.Goal()
            goal_msg.request = MotionPlanRequest()
            goal_msg.request.group_name = group_name
            goal_msg.request.num_planning_attempts = 1
            goal_msg.request.allowed_planning_time = 5.0
            goal_msg.request.planner_id = "RRTConnect"
            
            # Workspace parameters
            goal_msg.request.workspace_parameters = WorkspaceParameters()
            goal_msg.request.workspace_parameters.header.frame_id = frame_id
            goal_msg.request.workspace_parameters.min_corner.x = -2.0
            goal_msg.request.workspace_parameters.min_corner.y = -2.0
            goal_msg.request.workspace_parameters.min_corner.z = -2.0
            goal_msg.request.workspace_parameters.max_corner.x = 2.0
            goal_msg.request.workspace_parameters.max_corner.y = 2.0
            goal_msg.request.workspace_parameters.max_corner.z = 2.0
            
            # Create constraints
            constraints = Constraints()
            constraints.name = f"{group_name}_goals"
            
            for link_name, pose in target_poses.items():
                constraint = PositionConstraint()
                constraint.header.frame_id = frame_id
                constraint.link_name = link_name
                
                # Target position
                constraint.target_point_offset.x = pose.position.x
                constraint.target_point_offset.y = pose.position.y
                constraint.target_point_offset.z = pose.position.z
                
                # Large tolerance sphere for debugging
                constraint.constraint_region.primitives = [SolidPrimitive()]
                constraint.constraint_region.primitives[0].type = SolidPrimitive.SPHERE
                constraint.constraint_region.primitives[0].dimensions = [0.1]  # 10cm tolerance
                
                # Constraint region pose
                constraint.constraint_region.primitive_poses = [Pose()]
                constraint.constraint_region.primitive_poses[0].position.x = pose.position.x
                constraint.constraint_region.primitive_poses[0].position.y = pose.position.y
                constraint.constraint_region.primitive_poses[0].position.z = pose.position.z
                constraint.constraint_region.primitive_poses[0].orientation.w = 1.0
                
                constraint.weight = 1.0
                constraints.position_constraints.append(constraint)
                
                self.get_logger().info(f"   {link_name}: x={pose.position.x}, y={pose.position.y}, z={pose.position.z}")
            
            goal_msg.request.goal_constraints = [constraints]
            
            # Planning options - PLAN ONLY
            goal_msg.planning_options = PlanningOptions()
            goal_msg.planning_options.plan_only = True  # Only plan, don't execute
            goal_msg.planning_options.look_around = False
            goal_msg.planning_options.look_around_attempts = 0
            goal_msg.planning_options.max_safe_execution_cost = 0.0
            goal_msg.planning_options.replan = False
            goal_msg.planning_options.replan_attempts = 0
            goal_msg.planning_options.replan_delay = 0.0
            
            # Send goal
            future = self._action_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if not future.done():
                self.get_logger().error("❌ Goal sending timed out")
                return False
                
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("❌ Goal was rejected")
                return False
                
            self.get_logger().info("✅ Goal accepted, planning...")
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=15.0)
            
            if not result_future.done():
                self.get_logger().error("❌ Planning timed out")
                return False
                
            result = result_future.result()
            
            if result.result.error_code.val == 1:  # SUCCESS
                self.get_logger().info(f"✅ PLANNING SUCCESS! Time: {result.result.planning_time:.2f}s")
                return True
            else:
                self.get_logger().error(f"❌ PLANNING FAILED: ERROR_{result.result.error_code.val}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception: {e}")
            return False

    def run_debug_tests(self):
        """Run a series of debug tests"""
        
        # Test 1: Individual arm planning (should work)
        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 1: Individual left arm planning")
        
        left_pose = Pose()
        left_pose.position.x = 0.3
        left_pose.position.y = 0.0
        left_pose.position.z = 0.3
        left_pose.orientation.w = 1.0
        
        success1 = self.test_planning_only("left_arm", {"left_panda_link8": left_pose})
        
        # Test 2: Individual right arm planning (should work)
        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 2: Individual right arm planning")
        
        right_pose = Pose()
        right_pose.position.x = 0.3
        right_pose.position.y = 0.0
        right_pose.position.z = 0.3
        right_pose.orientation.w = 1.0
        
        success2 = self.test_planning_only("right_arm", {"right_panda_link8": right_pose})
        
        # Test 3: Dual arm with panda_link0 frame
        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 3: Dual arm planning (panda_link0 frame)")
        
        dual_poses_close = {
            "left_panda_link8": Pose(),
            "right_panda_link8": Pose()
        }
        dual_poses_close["left_panda_link8"].position.x = 0.3
        dual_poses_close["left_panda_link8"].position.y = 0.1
        dual_poses_close["left_panda_link8"].position.z = 0.3
        dual_poses_close["left_panda_link8"].orientation.w = 1.0
        
        dual_poses_close["right_panda_link8"].position.x = 0.3
        dual_poses_close["right_panda_link8"].position.y = -0.1
        dual_poses_close["right_panda_link8"].position.z = 0.3
        dual_poses_close["right_panda_link8"].orientation.w = 1.0
        
        success3 = self.test_planning_only("all_arms", dual_poses_close, "panda_link0")
        
        # Test 4: Dual arm with world frame
        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 4: Dual arm planning (world frame)")
        
        success4 = self.test_planning_only("all_arms", dual_poses_close, "world")
        
        # Test 5: Very conservative dual arm targets
        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 5: Dual arm planning (very conservative targets)")
        
        dual_poses_conservative = {
            "left_panda_link8": Pose(),
            "right_panda_link8": Pose()
        }
        dual_poses_conservative["left_panda_link8"].position.x = 0.4
        dual_poses_conservative["left_panda_link8"].position.y = 0.2
        dual_poses_conservative["left_panda_link8"].position.z = 0.4
        dual_poses_conservative["left_panda_link8"].orientation.w = 1.0
        
        dual_poses_conservative["right_panda_link8"].position.x = 0.4
        dual_poses_conservative["right_panda_link8"].position.y = -0.2
        dual_poses_conservative["right_panda_link8"].position.z = 0.4
        dual_poses_conservative["right_panda_link8"].orientation.w = 1.0
        
        success5 = self.test_planning_only("all_arms", dual_poses_conservative, "panda_link0")
        
        # Summary
        self.get_logger().info("=" * 60)
        self.get_logger().info("🎯 TEST SUMMARY:")
        self.get_logger().info(f"   Individual left arm: {'✅ PASS' if success1 else '❌ FAIL'}")
        self.get_logger().info(f"   Individual right arm: {'✅ PASS' if success2 else '❌ FAIL'}")
        self.get_logger().info(f"   Dual arm (panda_link0): {'✅ PASS' if success3 else '❌ FAIL'}")
        self.get_logger().info(f"   Dual arm (world frame): {'✅ PASS' if success4 else '❌ FAIL'}")
        self.get_logger().info(f"   Dual arm (conservative): {'✅ PASS' if success5 else '❌ FAIL'}")
        
        return any([success3, success4, success5])


def main():
    rclpy.init()
    
    # Create executor and node
    executor = MultiThreadedExecutor()
    test_node = DualArmDebugTest()
    executor.add_node(test_node)
    
    # Start executor in background thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    # Give time for initialization
    time.sleep(2.0)
    
    try:
        # Run debug tests
        success = test_node.run_debug_tests()
        
        if success:
            test_node.get_logger().info("🎊 === AT LEAST ONE DUAL-ARM TEST PASSED === 🎊")
        else:
            test_node.get_logger().error("💥 === ALL DUAL-ARM TESTS FAILED === 💥")
            
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        # Cleanup
        test_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
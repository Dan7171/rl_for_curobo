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


class DualArmSeparatedTest(Node):
    def __init__(self):
        super().__init__('dual_arm_separated_test')
        
        # Create action client for MoveGroup
        self._action_client = ActionClient(self, MoveGroup, '/move_action')
        
        self.get_logger().info("🎯 === DUAL-ARM SEPARATED TEST === 🎯")
        self.get_logger().info("Testing with widely separated arm positions to avoid collisions")

    def test_planning_only(self, group_name, target_poses, frame_id="panda_link0"):
        """Test planning only with given parameters"""
        try:
            self.get_logger().info(f"🧪 Testing planning for group: {group_name}")
            
            # Wait for action server
            if not self._action_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error("❌ MoveGroup action server not available")
                return False

            # Create motion plan request
            goal_msg = MoveGroup.Goal()
            goal_msg.request = MotionPlanRequest()
            goal_msg.request.group_name = group_name
            goal_msg.request.num_planning_attempts = 3
            goal_msg.request.allowed_planning_time = 10.0
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
                
                # Large tolerance sphere
                constraint.constraint_region.primitives = [SolidPrimitive()]
                constraint.constraint_region.primitives[0].type = SolidPrimitive.SPHERE
                constraint.constraint_region.primitives[0].dimensions = [0.15]  # 15cm tolerance
                
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
            goal_msg.planning_options.plan_only = True
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
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=20.0)
            
            if not result_future.done():
                self.get_logger().error("❌ Planning timed out")
                return False
                
            result = result_future.result()
            
            if result.result.error_code.val == 1:  # SUCCESS
                self.get_logger().info(f"✅ PLANNING SUCCESS! Time: {result.result.planning_time:.2f}s")
                return True
            else:
                self.get_logger().error(f"❌ PLANNING FAILED: ERROR_{result.result.error_code.val}")
                # Let's also print the error message if available
                if hasattr(result.result, 'error_message') and result.result.error_message:
                    self.get_logger().error(f"   Error message: {result.result.error_message}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception: {e}")
            return False

    def run_separated_tests(self):
        """Test with increasingly separated arm positions"""
        
        test_cases = [
            {
                "name": "Wide separation (Y: ±40cm)",
                "left": {"x": 0.4, "y": 0.4, "z": 0.4},
                "right": {"x": 0.4, "y": -0.4, "z": 0.4}
            },
            {
                "name": "Very wide separation (Y: ±60cm)",
                "left": {"x": 0.5, "y": 0.6, "z": 0.4},
                "right": {"x": 0.5, "y": -0.6, "z": 0.4}
            },
            {
                "name": "Different X positions",
                "left": {"x": 0.6, "y": 0.3, "z": 0.4},
                "right": {"x": 0.2, "y": -0.3, "z": 0.4}
            },
            {
                "name": "Far apart diagonally",
                "left": {"x": 0.6, "y": 0.5, "z": 0.5},
                "right": {"x": 0.2, "y": -0.5, "z": 0.3}
            },
            {
                "name": "Maximum separation",
                "left": {"x": 0.7, "y": 0.7, "z": 0.6},
                "right": {"x": 0.1, "y": -0.7, "z": 0.2}
            }
        ]
        
        successful_tests = []
        
        for i, test_case in enumerate(test_cases, 1):
            self.get_logger().info("=" * 60)
            self.get_logger().info(f"TEST {i}: {test_case['name']}")
            
            # Create poses
            dual_poses = {
                "left_panda_link8": Pose(),
                "right_panda_link8": Pose()
            }
            
            # Left arm
            dual_poses["left_panda_link8"].position.x = test_case["left"]["x"]
            dual_poses["left_panda_link8"].position.y = test_case["left"]["y"]
            dual_poses["left_panda_link8"].position.z = test_case["left"]["z"]
            dual_poses["left_panda_link8"].orientation.w = 1.0
            
            # Right arm
            dual_poses["right_panda_link8"].position.x = test_case["right"]["x"]
            dual_poses["right_panda_link8"].position.y = test_case["right"]["y"]
            dual_poses["right_panda_link8"].position.z = test_case["right"]["z"]
            dual_poses["right_panda_link8"].orientation.w = 1.0
            
            # Calculate separation distance
            dx = test_case["left"]["x"] - test_case["right"]["x"]
            dy = test_case["left"]["y"] - test_case["right"]["y"]
            dz = test_case["left"]["z"] - test_case["right"]["z"]
            distance = (dx**2 + dy**2 + dz**2)**0.5
            
            self.get_logger().info(f"   End effector separation: {distance:.2f}m")
            
            success = self.test_planning_only("all_arms", dual_poses, "panda_link0")
            
            if success:
                successful_tests.append(test_case["name"])
                self.get_logger().info(f"✅ {test_case['name']} - SUCCESS!")
                # If we found a working case, also try to execute it
                self.get_logger().info("🎬 Testing execution for this successful case...")
                exec_success = self.test_with_execution("all_arms", dual_poses, "panda_link0")
                if exec_success:
                    self.get_logger().info("🎉 EXECUTION ALSO SUCCESSFUL!")
                    break  # We found a working case, stop here
            else:
                self.get_logger().error(f"❌ {test_case['name']} - FAILED")
        
        # Summary
        self.get_logger().info("=" * 60)
        self.get_logger().info("🎯 SEPARATION TEST SUMMARY:")
        if successful_tests:
            self.get_logger().info("✅ Successful test cases:")
            for test_name in successful_tests:
                self.get_logger().info(f"   - {test_name}")
        else:
            self.get_logger().error("❌ No test cases succeeded")
        
        return len(successful_tests) > 0

    def test_with_execution(self, group_name, target_poses, frame_id="panda_link0"):
        """Test planning AND execution"""
        try:
            # Create motion plan request (similar to planning-only but with execution)
            goal_msg = MoveGroup.Goal()
            goal_msg.request = MotionPlanRequest()
            goal_msg.request.group_name = group_name
            goal_msg.request.num_planning_attempts = 3
            goal_msg.request.allowed_planning_time = 10.0
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
            constraints.name = f"{group_name}_execution_goals"
            
            for link_name, pose in target_poses.items():
                constraint = PositionConstraint()
                constraint.header.frame_id = frame_id
                constraint.link_name = link_name
                
                # Target position
                constraint.target_point_offset.x = pose.position.x
                constraint.target_point_offset.y = pose.position.y
                constraint.target_point_offset.z = pose.position.z
                
                # Tolerance sphere
                constraint.constraint_region.primitives = [SolidPrimitive()]
                constraint.constraint_region.primitives[0].type = SolidPrimitive.SPHERE
                constraint.constraint_region.primitives[0].dimensions = [0.1]  # 10cm tolerance for execution
                
                # Constraint region pose
                constraint.constraint_region.primitive_poses = [Pose()]
                constraint.constraint_region.primitive_poses[0].position.x = pose.position.x
                constraint.constraint_region.primitive_poses[0].position.y = pose.position.y
                constraint.constraint_region.primitive_poses[0].position.z = pose.position.z
                constraint.constraint_region.primitive_poses[0].orientation.w = 1.0
                
                constraint.weight = 1.0
                constraints.position_constraints.append(constraint)
            
            goal_msg.request.goal_constraints = [constraints]
            
            # Planning options - PLAN AND EXECUTE
            goal_msg.planning_options = PlanningOptions()
            goal_msg.planning_options.plan_only = False  # Plan AND execute
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
                
            self.get_logger().info("✅ Goal accepted, planning and executing...")
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
            
            if not result_future.done():
                self.get_logger().error("❌ Planning/execution timed out")
                return False
                
            result = result_future.result()
            
            if result.result.error_code.val == 1:  # SUCCESS
                self.get_logger().info(f"✅ PLANNING AND EXECUTION SUCCESS! Time: {result.result.planning_time:.2f}s")
                return True
            else:
                self.get_logger().error(f"❌ PLANNING/EXECUTION FAILED: ERROR_{result.result.error_code.val}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception during execution test: {e}")
            return False


def main():
    rclpy.init()
    
    # Create executor and node
    executor = MultiThreadedExecutor()
    test_node = DualArmSeparatedTest()
    executor.add_node(test_node)
    
    # Start executor in background thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    # Give time for initialization
    time.sleep(2.0)
    
    try:
        # Run separated tests
        success = test_node.run_separated_tests()
        
        if success:
            test_node.get_logger().info("🎊 === FOUND WORKING DUAL-ARM CONFIGURATION === 🎊")
        else:
            test_node.get_logger().error("💥 === ALL SEPARATION TESTS FAILED === 💥")
            
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        # Cleanup
        test_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
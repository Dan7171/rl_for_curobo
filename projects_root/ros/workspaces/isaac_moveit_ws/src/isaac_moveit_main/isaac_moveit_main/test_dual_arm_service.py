#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPlanningScene, GetMotionPlan
from moveit_msgs.msg import (
    MotionPlanRequest, PlanningScene, RobotState, 
    WorkspaceParameters, Constraints, JointConstraint
)
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
import time
from rclpy.executors import MultiThreadedExecutor
import threading


class DualArmServiceTest(Node):
    def __init__(self):
        super().__init__('dual_arm_service_test')
        
        # Create service clients
        self.get_planning_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.get_motion_plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        
        self.get_logger().info("🔧 === DUAL-ARM SERVICE TEST === 🔧")
        self.get_logger().info("Using MoveIt services instead of action client")

    def wait_for_services(self):
        """Wait for MoveIt services to be available"""
        self.get_logger().info("⏳ Waiting for MoveIt services...")
        
        if not self.get_planning_scene_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ GetPlanningScene service not available")
            return False
            
        if not self.get_motion_plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ GetMotionPlan service not available") 
            return False
            
        self.get_logger().info("✅ All MoveIt services available")
        return True

    def get_current_robot_state(self):
        """Get the current robot state from planning scene"""
        try:
            request = GetPlanningScene.Request()
            request.components.components = request.components.ROBOT_STATE
            
            future = self.get_planning_scene_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if future.done():
                response = future.result()
                return response.scene.robot_state
            else:
                self.get_logger().error("❌ Failed to get current robot state")
                return None
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception getting robot state: {e}")
            return None

    def create_joint_space_goals(self):
        """Create joint space goals for dual arm (safer than Cartesian)"""
        # Conservative joint positions for both arms
        left_joints = [0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0]  # Left arm safe position
        right_joints = [0.0, 0.5, 0.0, -1.5, 0.0, 1.0, 0.0]  # Right arm safe position (mirrored)
        
        # Create joint constraints
        constraints = Constraints()
        constraints.name = "dual_arm_joint_goals"
        
        # Left arm constraints
        left_joint_names = [
            "left_panda_joint1", "left_panda_joint2", "left_panda_joint3", "left_panda_joint4",
            "left_panda_joint5", "left_panda_joint6", "left_panda_joint7"
        ]
        
        for i, (joint_name, target_value) in enumerate(zip(left_joint_names, left_joints)):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = target_value
            constraint.tolerance_above = 0.05
            constraint.tolerance_below = 0.05
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        
        # Right arm constraints
        right_joint_names = [
            "right_panda_joint1", "right_panda_joint2", "right_panda_joint3", "right_panda_joint4",
            "right_panda_joint5", "right_panda_joint6", "right_panda_joint7"
        ]
        
        for i, (joint_name, target_value) in enumerate(zip(right_joint_names, right_joints)):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = target_value
            constraint.tolerance_above = 0.05
            constraint.tolerance_below = 0.05
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        
        return constraints

    def test_joint_space_planning(self):
        """Test dual-arm planning using joint space goals"""
        try:
            self.get_logger().info("🧪 Testing dual-arm planning with joint space goals...")
            
            # Get current robot state
            current_state = self.get_current_robot_state()
            if current_state is None:
                return False
            
            # Create motion plan request
            request = GetMotionPlan.Request()
            request.motion_plan_request = MotionPlanRequest()
            request.motion_plan_request.group_name = "all_arms"
            request.motion_plan_request.num_planning_attempts = 3
            request.motion_plan_request.allowed_planning_time = 10.0
            request.motion_plan_request.planner_id = "RRTConnect"
            
            # Set start state
            request.motion_plan_request.start_state = current_state
            
            # Set workspace parameters
            request.motion_plan_request.workspace_parameters = WorkspaceParameters()
            request.motion_plan_request.workspace_parameters.header.frame_id = "panda_link0"
            request.motion_plan_request.workspace_parameters.min_corner.x = -2.0
            request.motion_plan_request.workspace_parameters.min_corner.y = -2.0
            request.motion_plan_request.workspace_parameters.min_corner.z = -2.0
            request.motion_plan_request.workspace_parameters.max_corner.x = 2.0
            request.motion_plan_request.workspace_parameters.max_corner.y = 2.0
            request.motion_plan_request.workspace_parameters.max_corner.z = 2.0
            
            # Set joint space goals
            goal_constraints = self.create_joint_space_goals()
            request.motion_plan_request.goal_constraints = [goal_constraints]
            
            self.get_logger().info("📋 Planning request details:")
            self.get_logger().info(f"   Group: {request.motion_plan_request.group_name}")
            self.get_logger().info(f"   Planner: {request.motion_plan_request.planner_id}")
            self.get_logger().info(f"   Joint constraints: {len(goal_constraints.joint_constraints)}")
            
            # Send planning request
            self.get_logger().info("🚀 Sending planning request...")
            future = self.get_motion_plan_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            
            if not future.done():
                self.get_logger().error("❌ Planning request timed out")
                return False
            
            response = future.result()
            
            if response.motion_plan_response.error_code.val == 1:  # SUCCESS
                self.get_logger().info("✅ JOINT SPACE PLANNING SUCCESS!")
                self.get_logger().info(f"   Planning time: {response.motion_plan_response.planning_time:.2f}s")
                self.get_logger().info(f"   Trajectory points: {len(response.motion_plan_response.trajectory.joint_trajectory.points)}")
                return True
            else:
                self.get_logger().error(f"❌ PLANNING FAILED: ERROR_{response.motion_plan_response.error_code.val}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception during joint space planning: {e}")
            return False

    def test_simple_joint_goals(self):
        """Test with very simple joint goals - just move to a known good configuration"""
        try:
            self.get_logger().info("🧪 Testing with simple joint goals (home position)...")
            
            # Get current robot state
            current_state = self.get_current_robot_state()
            if current_state is None:
                return False
            
            # Create motion plan request
            request = GetMotionPlan.Request()
            request.motion_plan_request = MotionPlanRequest()
            request.motion_plan_request.group_name = "all_arms"
            request.motion_plan_request.num_planning_attempts = 1
            request.motion_plan_request.allowed_planning_time = 5.0
            request.motion_plan_request.planner_id = "RRTConnect"
            
            # Set start state
            request.motion_plan_request.start_state = current_state
            
            # Simple goal: both arms to home position (all joints to 0)
            constraints = Constraints()
            constraints.name = "home_position"
            
            all_joint_names = [
                "left_panda_joint1", "left_panda_joint2", "left_panda_joint3", "left_panda_joint4",
                "left_panda_joint5", "left_panda_joint6", "left_panda_joint7",
                "right_panda_joint1", "right_panda_joint2", "right_panda_joint3", "right_panda_joint4", 
                "right_panda_joint5", "right_panda_joint6", "right_panda_joint7"
            ]
            
            for joint_name in all_joint_names:
                constraint = JointConstraint()
                constraint.joint_name = joint_name
                constraint.position = 0.0  # Home position
                constraint.tolerance_above = 0.1
                constraint.tolerance_below = 0.1
                constraint.weight = 1.0
                constraints.joint_constraints.append(constraint)
            
            request.motion_plan_request.goal_constraints = [constraints]
            
            self.get_logger().info("🚀 Sending simple home position planning request...")
            future = self.get_motion_plan_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            
            if not future.done():
                self.get_logger().error("❌ Simple planning request timed out")
                return False
            
            response = future.result()
            
            if response.motion_plan_response.error_code.val == 1:  # SUCCESS
                self.get_logger().info("✅ SIMPLE JOINT PLANNING SUCCESS!")
                self.get_logger().info(f"   Planning time: {response.motion_plan_response.planning_time:.2f}s")
                return True
            else:
                self.get_logger().error(f"❌ SIMPLE PLANNING FAILED: ERROR_{response.motion_plan_response.error_code.val}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception during simple planning: {e}")
            return False

    def run_service_tests(self):
        """Run all service-based tests"""
        if not self.wait_for_services():
            return False
        
        # Test 1: Simple home position
        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 1: Simple home position planning")
        success1 = self.test_simple_joint_goals()
        
        # Test 2: More complex joint space goals
        self.get_logger().info("=" * 60)
        self.get_logger().info("TEST 2: Joint space dual-arm planning")
        success2 = self.test_joint_space_planning()
        
        # Summary
        self.get_logger().info("=" * 60)
        self.get_logger().info("🎯 SERVICE TEST SUMMARY:")
        self.get_logger().info(f"   Simple home position: {'✅ PASS' if success1 else '❌ FAIL'}")
        self.get_logger().info(f"   Joint space dual-arm: {'✅ PASS' if success2 else '❌ FAIL'}")
        
        return success1 or success2


def main():
    rclpy.init()
    
    # Create executor and node
    executor = MultiThreadedExecutor()
    test_node = DualArmServiceTest()
    executor.add_node(test_node)
    
    # Start executor in background thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    # Give time for initialization
    time.sleep(2.0)
    
    try:
        # Run service tests
        success = test_node.run_service_tests()
        
        if success:
            test_node.get_logger().info("🎊 === AT LEAST ONE SERVICE TEST PASSED === 🎊")
        else:
            test_node.get_logger().error("💥 === ALL SERVICE TESTS FAILED === 💥")
            
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        # Cleanup
        test_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
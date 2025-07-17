#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.srv import GetPlanningScene, GetMotionPlan
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest, PlanningScene, RobotState, 
    WorkspaceParameters, Constraints, JointConstraint
)
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
import time
from rclpy.executors import MultiThreadedExecutor
import threading


class DualArmPlanExecuteTest(Node):
    def __init__(self):
        super().__init__('dual_arm_plan_execute_test')
        
        # Create service clients for planning
        self.get_planning_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.get_motion_plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        
        # Create action client for execution
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        
        self.get_logger().info("🚀 === DUAL-ARM PLAN & EXECUTE TEST === 🚀")
        self.get_logger().info("Using working service approach + trajectory execution")

    def wait_for_services(self):
        """Wait for MoveIt services and actions to be available"""
        self.get_logger().info("⏳ Waiting for MoveIt services and actions...")
        
        if not self.get_planning_scene_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ GetPlanningScene service not available")
            return False
            
        if not self.get_motion_plan_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("❌ GetMotionPlan service not available") 
            return False
            
        if not self.execute_trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error("❌ ExecuteTrajectory action not available")
            return False
            
        self.get_logger().info("✅ All MoveIt services and actions available")
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

    def create_safe_dual_arm_goals(self):
        """Create safe joint space goals for both arms"""
        # Safe positions that spread the arms apart
        left_joints = [0.3, -0.8, 0.0, -1.8, 0.0, 1.0, 0.0]   # Left arm position  
        right_joints = [-0.3, 0.8, 0.0, -1.8, 0.0, 1.0, 0.0]  # Right arm position (mirrored)
        
        # Create joint constraints
        constraints = Constraints()
        constraints.name = "safe_dual_arm_position"
        
        # Left arm constraints
        left_joint_names = [
            "left_panda_joint1", "left_panda_joint2", "left_panda_joint3", "left_panda_joint4",
            "left_panda_joint5", "left_panda_joint6", "left_panda_joint7"
        ]
        
        for joint_name, target_value in zip(left_joint_names, left_joints):
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
        
        for joint_name, target_value in zip(right_joint_names, right_joints):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = target_value
            constraint.tolerance_above = 0.05
            constraint.tolerance_below = 0.05
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        
        return constraints

    def plan_dual_arm_motion(self):
        """Plan dual-arm motion and return trajectory"""
        try:
            self.get_logger().info("📋 STEP 1: Planning dual-arm motion...")
            
            # Get current robot state
            current_state = self.get_current_robot_state()
            if current_state is None:
                return None
            
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
            goal_constraints = self.create_safe_dual_arm_goals()
            request.motion_plan_request.goal_constraints = [goal_constraints]
            
            self.get_logger().info(f"   Planning for {len(goal_constraints.joint_constraints)} joint constraints")
            
            # Send planning request
            self.get_logger().info("🚀 Sending planning request...")
            future = self.get_motion_plan_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)
            
            if not future.done():
                self.get_logger().error("❌ Planning request timed out")
                return None
            
            response = future.result()
            
            if response.motion_plan_response.error_code.val == 1:  # SUCCESS
                self.get_logger().info("✅ PLANNING SUCCESS!")
                self.get_logger().info(f"   Planning time: {response.motion_plan_response.planning_time:.2f}s")
                self.get_logger().info(f"   Trajectory points: {len(response.motion_plan_response.trajectory.joint_trajectory.points)}")
                return response.motion_plan_response.trajectory
            else:
                self.get_logger().error(f"❌ PLANNING FAILED: ERROR_{response.motion_plan_response.error_code.val}")
                return None
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception during planning: {e}")
            return None

    def execute_trajectory(self, trajectory):
        """Execute the planned trajectory"""
        try:
            self.get_logger().info("🎬 STEP 2: Executing dual-arm trajectory...")
            
            # Create execution request
            goal_msg = ExecuteTrajectory.Goal()
            goal_msg.trajectory = trajectory
            
            self.get_logger().info("🚀 Sending execution request...")
            
            # Send execution goal
            future = self.execute_trajectory_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            
            if not future.done():
                self.get_logger().error("❌ Execution goal sending timed out")
                return False
                
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("❌ Execution goal was rejected")
                return False
                
            self.get_logger().info("✅ EXECUTION GOAL ACCEPTED!")
            self.get_logger().info("⏳ Executing coordinated dual-arm motion...")
            
            # Wait for execution result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=30.0)
            
            if not result_future.done():
                self.get_logger().error("❌ Execution timed out")
                return False
                
            result = result_future.result()
            
            if result.result.error_code.val == 1:  # SUCCESS
                self.get_logger().info("✅ EXECUTION SUCCESS!")
                self.get_logger().info("🎉 Dual-arm coordinated motion completed!")
                return True
            else:
                self.get_logger().error(f"❌ EXECUTION FAILED: ERROR_{result.result.error_code.val}")
                return False
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception during execution: {e}")
            return False

    def run_plan_and_execute_test(self):
        """Run complete plan and execute test"""
        if not self.wait_for_services():
            return False
        
        self.get_logger().info("🎯 === DUAL-ARM PLAN & EXECUTE TEST === 🎯")
        self.get_logger().info("Goal: Plan and execute coordinated motion for both arms")
        
        # Step 1: Plan
        trajectory = self.plan_dual_arm_motion()
        if trajectory is None:
            self.get_logger().error("💥 Planning failed - cannot proceed to execution")
            return False
        
        # Step 2: Execute
        success = self.execute_trajectory(trajectory)
        
        if success:
            self.get_logger().info("🎊 === PLAN & EXECUTE SUCCESS === 🎊")
            self.get_logger().info("✨ Both arms moved in coordinated fashion!")
        else:
            self.get_logger().error("💥 === EXECUTION FAILED === 💥")
        
        return success


def main():
    rclpy.init()
    
    # Create executor and node
    executor = MultiThreadedExecutor()
    test_node = DualArmPlanExecuteTest()
    executor.add_node(test_node)
    
    # Start executor in background thread
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    
    # Give time for initialization
    time.sleep(2.0)
    
    try:
        # Run plan and execute test
        success = test_node.run_plan_and_execute_test()
        
        if success:
            test_node.get_logger().info("🌟 === DUAL-ARM SYSTEM WORKING === 🌟")
        else:
            test_node.get_logger().error("🔧 === NEEDS DEBUGGING === 🔧")
            
    except KeyboardInterrupt:
        test_node.get_logger().info("Test interrupted by user")
    finally:
        # Cleanup
        test_node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 
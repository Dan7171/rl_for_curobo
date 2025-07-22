
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
import numpy as np
import torch
import random

# CuRobo imports
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose as CuroboPose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


class DualArmPlanExecuteTest(Node):
    def __init__(self):
        super().__init__('dual_arm_plan_execute_test')
        
        # Create service clients for planning
        self.get_planning_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.get_motion_plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        
        # Create action client for execution
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        
        # Initialize CuRobo IK solver
        self.init_ik_solver()
        
        self.get_logger().info("🚀 === DUAL-ARM PLAN & EXECUTE TEST === 🚀")
        self.get_logger().info("Using CuRobo IK solver for cartesian goal processing")

    def init_ik_solver(self):
        """Initialize CuRobo IK solver for dual arm robot"""
        try:
            self.tensor_args = TensorDeviceType()
            
            # Load dual arm robot configuration
            robot_file = "franka_dual_arm.yml"
            robot_cfg = RobotConfig.from_dict(
                load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
            )
            
            # Create IK solver configuration
            ik_config = IKSolverConfig.load_from_robot_config(
                robot_cfg,
                world_model=None,  # No collision checking for now
                rotation_threshold=0.05,
                position_threshold=0.005,
                num_seeds=20,  # User requested up to 20 solutions
                self_collision_check=True,
                self_collision_opt=True,
                tensor_args=self.tensor_args,
                use_cuda_graph=False,  # Disable for flexibility
            )
            
            self.ik_solver = IKSolver(ik_config)
            
            # Joint names for dual arm robot
            self.left_joint_names = [
                "left_panda_joint1", "left_panda_joint2", "left_panda_joint3", "left_panda_joint4",
                "left_panda_joint5", "left_panda_joint6", "left_panda_joint7"
            ]
            
            self.right_joint_names = [
                "right_panda_joint1", "right_panda_joint2", "right_panda_joint3", "right_panda_joint4",
                "right_panda_joint5", "right_panda_joint6", "right_panda_joint7"
            ]
            
            # All joint names in order
            self.all_joint_names = self.left_joint_names + self.right_joint_names
            
            self.get_logger().info("✅ CuRobo IK solver initialized successfully")
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize IK solver: {e}")
            raise

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

    def cartesian_goals_to_curobo_poses(self, cartesian_goals):
        """
        Convert cartesian goals to CuRobo Pose format.
        
        Args:
            cartesian_goals: List of 2 goal poses, each as [x,y,z,qw,qx,qy,qz]
        
        Returns:
            Dictionary with link poses for CuRobo IK solver
        """
        if len(cartesian_goals) != 2:
            raise ValueError("Expected exactly 2 cartesian goals (left and right arm)")
        
        link_poses = {}
        
        # Left arm goal (index 0)
        left_goal = cartesian_goals[0]
        if len(left_goal) != 7:
            raise ValueError("Each goal must have 7 elements: [x,y,z,qw,qx,qy,qz]")
        
        left_pose = CuroboPose(
            position=self.tensor_args.to_device(torch.tensor(left_goal[:3], dtype=torch.float32)),
            quaternion=self.tensor_args.to_device(torch.tensor(left_goal[3:], dtype=torch.float32))
        )
        link_poses["left_panda_hand"] = left_pose
        
        # Right arm goal (index 1)
        right_goal = cartesian_goals[1]
        if len(right_goal) != 7:
            raise ValueError("Each goal must have 7 elements: [x,y,z,qw,qx,qy,qz]")
            
        right_pose = CuroboPose(
            position=self.tensor_args.to_device(torch.tensor(right_goal[:3], dtype=torch.float32)),
            quaternion=self.tensor_args.to_device(torch.tensor(right_goal[3:], dtype=torch.float32))
        )
        link_poses["right_panda_hand"] = right_pose
        
        return link_poses

    def solve_ik_for_cartesian_goals(self, cartesian_goals):
        """
        Solve IK for the given cartesian goals and return a sampled solution.
        
        Args:
            cartesian_goals: List of 2 goal poses, each as [x,y,z,qw,qx,qy,qz]
            
        Returns:
            Joint configuration as list of 14 joint values, or None if no solution found
        """
        try:
            self.get_logger().info("🔧 Converting cartesian goals to CuRobo format...")
            link_poses = self.cartesian_goals_to_curobo_poses(cartesian_goals)
            
            self.get_logger().info(f"   Left arm target:  {cartesian_goals[0][:3]}")
            self.get_logger().info(f"   Right arm target: {cartesian_goals[1][:3]}")
            
            # Get proper retract config from IK solver (like in ik_reachability.py)
            retract_config = self.ik_solver.get_retract_config().view(1, -1)
            self.get_logger().info(f"   Using CuRobo retract config with {retract_config.shape[1]} joints")
            self.get_logger().info(f"   CuRobo joint order: {self.ik_solver.joint_names}")
            
            # Get current joint state for comparison (but use retract for seeding)
            current_state = self.get_current_robot_state()
            if current_state is not None:
                self.get_logger().info(f"   ROS joint state has {len(current_state.joint_state.position)} joints")
                self.get_logger().info(f"   ROS joint names: {current_state.joint_state.name[:14]}")
            
            self.get_logger().info("🎯 Solving IK with CuRobo using proper retract config...")
            
            # Test basic FK first to verify setup
            fk_state = self.ik_solver.fk(retract_config)
            if fk_state is not None and fk_state.ee_pose is not None and fk_state.ee_pose.position is not None:
                self.get_logger().info(f"   FK test successful - EE pose shape: {fk_state.ee_pose.position.shape}")
            else:
                self.get_logger().warning("   FK test returned None or incomplete state")
            
            # Use primary end effector pose for main IK solve (left arm)
            primary_pose = link_poses["left_panda_hand"]
            
            # First try: solve for left arm only using retract config for seeding
            self.get_logger().info("   Step 1: Testing left arm IK...")
            ik_result_left = self.ik_solver.solve_single(
                goal_pose=primary_pose,
                retract_config=retract_config,
                seed_config=retract_config.unsqueeze(0),  # Shape: (1, 1, dof)
                return_seeds=5,  # Start with fewer seeds for debugging
                num_seeds=10,    # Use fewer seeds for debugging
            )
            
            left_success_count = torch.count_nonzero(ik_result_left.success).item()
            self.get_logger().info(f"   Left arm IK: {left_success_count} successful solutions")
            
            if not ik_result_left.success.any():
                self.get_logger().error("❌ Even single arm IK failed - poses may be unreachable")
                self.get_logger().error(f"   Target pose: pos={primary_pose.position.cpu().numpy()}, quat={primary_pose.quaternion.cpu().numpy()}")
                return None
            
            # Now try dual arm IK using link_poses
            self.get_logger().info("   Step 2: Attempting dual arm IK with link poses...")
            ik_result = self.ik_solver.solve_single(
                goal_pose=primary_pose,  # Primary pose (left arm)
                retract_config=retract_config,
                seed_config=retract_config.unsqueeze(0),  # Shape: (1, 1, dof) 
                return_seeds=20,  # Return up to 20 solutions
                num_seeds=20,     # Use 20 seeds for solving
                link_poses=link_poses  # This should handle dual arm constraints
            )
            
            successful_count = torch.count_nonzero(ik_result.success).item()
            self.get_logger().info(f"   Dual arm IK: {successful_count} successful solutions")
            
            if not ik_result.success.any():
                self.get_logger().error("❌ Dual arm IK solver found no valid solutions")
                # Fall back to using single arm solution extended to dual arm
                self.get_logger().info("   Fallback: Using single arm solution pattern")
                single_solution = ik_result_left.solution[0].cpu().numpy()  # Take first successful solution
                
                # Check the solution length and create proper dual arm solution
                self.get_logger().info(f"   Single arm solution length: {len(single_solution)}")
                if len(single_solution) == 14:
                    # Already dual arm solution
                    return single_solution.tolist()
                elif len(single_solution) == 7:
                    # Single arm, create conservative dual arm solution
                    # Use retract config for second arm
                    retract_joints = retract_config[0].cpu().numpy()
                    if len(retract_joints) >= 14:
                        dual_solution = list(single_solution) + list(retract_joints[7:14])
                        self.get_logger().info("   Created dual arm solution: left arm IK + right arm retract")
                        return dual_solution
                    else:
                        # Mirror the single arm solution
                        dual_solution = list(single_solution) + list(single_solution)
                        self.get_logger().info("   Created mirrored dual arm solution")
                        return dual_solution
                else:
                    self.get_logger().error(f"   Unexpected solution length: {len(single_solution)}")
                    return None
            
            # Get successful solutions from dual arm IK
            successful_indices = torch.where(ik_result.success)[0]
            num_solutions = len(successful_indices)
            
            self.get_logger().info(f"✅ IK solver found {num_solutions} valid dual arm solutions")
            
            # Sample one solution randomly from successful ones
            if num_solutions > 0:
                sampled_idx = successful_indices[random.randint(0, num_solutions - 1)]
                # ik_result.solution is a tensor with shape (return_seeds, dof)
                sampled_solution = ik_result.solution[sampled_idx].cpu().numpy()
                
                self.get_logger().info(f"🎲 Sampled solution {sampled_idx.item()} from {num_solutions} solutions")
                self.get_logger().info(f"   Solution length: {len(sampled_solution)}")
                
                # Verify solution is the right length for dual arm
                if len(sampled_solution) != 14:
                    self.get_logger().warning(f"   Expected 14 joints, got {len(sampled_solution)}")
                
                return sampled_solution.tolist()
            else:
                return None
                
        except Exception as e:
            self.get_logger().error(f"❌ Exception during IK solving: {e}")
            import traceback
            self.get_logger().error(f"   Traceback: {traceback.format_exc()}")
            return None

    def create_joint_constraints_from_target(self, target_joint_positions):
        """Create joint constraints for the target configuration"""
        constraints = Constraints()
        constraints.name = "dual_arm_joint_target"
        
        for joint_name, target_value in zip(self.all_joint_names, target_joint_positions):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = target_value
            constraint.tolerance_above = 0.05
            constraint.tolerance_below = 0.05
            constraint.weight = 1.0
            constraints.joint_constraints.append(constraint)
        
        return constraints

    def plan_to_joint_target(self, target_joint_positions):
        """Plan dual-arm motion to target joint configuration"""
        try:
            self.get_logger().info("📋 Planning motion to target joint configuration...")
            
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
            
            # Set joint space goals from IK solution
            goal_constraints = self.create_joint_constraints_from_target(target_joint_positions)
            request.motion_plan_request.goal_constraints = [goal_constraints]
            
            self.get_logger().info(f"   Planning to {len(goal_constraints.joint_constraints)} joint targets")
            
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
            self.get_logger().info("🎬 Executing dual-arm trajectory...")
            
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

    def run_cartesian_goal_test(self, cartesian_goals):
        """
        Run complete cartesian goal to execution pipeline
        
        Args:
            cartesian_goals: List of 2 goal poses, each as [x,y,z,qw,qx,qy,qz]
        """
        if not self.wait_for_services():
            return False
        
        self.get_logger().info("🎯 === DUAL-ARM CARTESIAN GOAL TEST === 🎯")
        self.get_logger().info("Pipeline: Cartesian Goals → IK Solutions → Sample → Plan → Execute")
        
        # Step 1: Solve IK for cartesian goals
        target_joint_positions = self.solve_ik_for_cartesian_goals(cartesian_goals)
        if target_joint_positions is None:
            self.get_logger().error("💥 IK solving failed - cannot proceed")
            return False
        
        # Step 2: Plan to target joint configuration
        trajectory = self.plan_to_joint_target(target_joint_positions)
        if trajectory is None:
            self.get_logger().error("💥 Planning failed - cannot proceed to execution")
            return False
        
        # Step 3: Execute trajectory
        success = self.execute_trajectory(trajectory)
        
        if success:
            self.get_logger().info("🎊 === CARTESIAN GOAL PIPELINE SUCCESS === 🎊")
            self.get_logger().info("✨ Both arms reached their cartesian targets!")
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
        # Example cartesian goals: [x, y, z, qw, qx, qy, qz]
        # Left arm goal (index 0), Right arm goal (index 1)
        # Using more conservative, reachable poses closer to robot base
        cartesian_goals = [
            [0.3, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],   # Left arm: closer and lower
            [0.3, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0]   # Right arm: closer and lower
        ]
        
        test_node.get_logger().info("🎯 Using conservative reachable cartesian goals:")
        test_node.get_logger().info(f"   Left arm:  {cartesian_goals[0]}")
        test_node.get_logger().info(f"   Right arm: {cartesian_goals[1]}")
        
        # Run cartesian goal test
        success = test_node.run_cartesian_goal_test(cartesian_goals)
        
        if success:
            test_node.get_logger().info("🌟 === DUAL-ARM CARTESIAN SYSTEM WORKING === 🌟")
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
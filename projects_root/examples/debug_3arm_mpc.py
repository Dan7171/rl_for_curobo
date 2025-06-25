#!/usr/bin/env python3

import sys
import os
import argparse
import torch
import numpy as np

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'curobo'))

from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.math import Pose
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

def main():
    # Test 3-arm MPC diagnostics
    
    # Robot config
    robot_cfg_file = "test_generated_configs/franka_3_arm.yml"
    robot_cfg = RobotConfig.from_dict(load_yaml(robot_cfg_file)["robot_cfg"], TensorDeviceType(device="cuda:0"))
    
    # Particle config  
    particle_cfg_file = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_triple_arm.yml"
    
    # Load configs
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        particle_cfg_file,
    )
    
    # Create MPC solver
    mpc_solver = MpcSolver(mpc_config)
    
    print("=== 3-Arm MPC Diagnostics ===")
    print(f"Robot DOF: {robot_cfg.kinematics.n_dof}")
    print(f"Link names: {robot_cfg.kinematics.link_names}")
    print(f"EE link: {robot_cfg.kinematics.ee_link}")
    
    # Test goal setup
    target_positions = torch.tensor([
        [0.2, 0.3, 0.5],  # Arm 0 target
        [0.5, 0.3, 0.5],  # Arm 1 target  
        [0.8, 0.3, 0.5],  # Arm 2 target
    ], device="cuda:0", dtype=torch.float32)
    
    target_quaternions = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # Identity quaternion for all arms
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], device="cuda:0", dtype=torch.float32)
    
    # Create multi-arm goal
    goal_pose = Pose(position=target_positions, quaternion=target_quaternions)
    
    print(f"\nGoal pose shape: pos={goal_pose.position.shape}, quat={goal_pose.quaternion.shape}")
    
    # Get current state (retract config)
    retract_config = torch.tensor(robot_cfg.kinematics.cspace.retract_config, device="cuda:0", dtype=torch.float32)
    current_state = retract_config
    print(f"Current state shape: {current_state.shape}")
    print(f"Current state (first 9 joints): {current_state[:9].cpu().numpy()}")
    
    # Set goal and solve
    result = mpc_solver.solve_single(goal_pose, current_state, max_attempts=1)
    
    print(f"\nMPC Result:")
    print(f"  Success: {result.success}")
    if hasattr(result, 'js_action') and result.js_action is not None:
        print(f"  Solution shape: {result.js_action.shape}")
        print(f"  Cost: {result.cost}")
        
        # Check if actions are non-zero
        action_magnitude = torch.norm(result.js_action, dim=-1)
        print(f"  Action magnitudes (first 5 timesteps): {action_magnitude[:5].cpu().numpy()}")
        
        # Check end-effector poses
        if hasattr(result, 'ee_pos_seq') and result.ee_pos_seq is not None:
            ee_pos = result.ee_pos_seq
            print(f"  EE positions shape: {ee_pos.shape}")
            if len(ee_pos.shape) == 4:  # Multi-arm format
                print(f"  Final EE positions:")
                for arm_idx in range(min(3, ee_pos.shape[2])):
                    final_pos = ee_pos[0, -1, arm_idx, :].cpu().numpy()
                    target_pos = target_positions[arm_idx].cpu().numpy()
                    distance = np.linalg.norm(final_pos - target_pos)
                    print(f"    Arm {arm_idx}: {final_pos} (target: {target_pos}, dist: {distance:.3f})")
    else:
        print("  MPC failed to find solution")
        if hasattr(result, 'debug'):
            print(f"  Debug info: {result.debug}")
    
    # Test cost function directly
    print(f"\n=== Cost Function Test ===")
    
    # Create a small perturbation toward the goal
    test_state = current_state.clone()
    # Move first 7 joints slightly (arm 0)
    test_state[:7] += 0.1 * torch.randn(7, device="cuda:0")
    
    # Test forward kinematics
    try:
        # Get kinematics
        kin_state = mpc_solver.rollout_fn.dynamics_model.robot_model.get_state(test_state.unsqueeze(0))
        print(f"Forward kinematics successful")
        print(f"EE position: {kin_state.ee_position}")
    except Exception as e:
        print(f"Forward kinematics failed: {e}")
    
    # Check pose cost specifically
    if hasattr(mpc_solver.rollout_fn, 'goal_cost'):
        pose_cost = mpc_solver.rollout_fn.goal_cost
        print(f"Pose cost type: {type(pose_cost)}")
        if hasattr(pose_cost, 'num_arms'):
            print(f"Pose cost num_arms: {pose_cost.num_arms}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script to verify plotting integration works with MpcPlanner.
"""

import sys
import time
import torch
import numpy as np
from typing import Dict

# Add the curobo path
sys.path.append('/home/dan/rl_for_curobo/curobo/src')

def test_mpc_plotting_integration():
    """Test MpcPlanner with centralized plotter integration."""
    
    print("=== Testing MPC + Centralized Plotter Integration ===")
    
    try:
        # Import required modules
        from curobo.wrap.reacher.mpc import MpcSolverConfig, MpcSolver
        from curobo.geom.types import WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.types.state import JointState
        from curobo.types.math import Pose
        from curobo.rollout.rollout_base import Goal
        from curobo.util_file import load_yaml, get_robot_configs_path, join_path
        from curobo.rollout.arm_reacher import get_global_plotter
        import threading
        
        print("✓ Imports successful")
        
        # Initialize centralized plotter
        plotter = get_global_plotter()
        plotter.enable_plotting(max_agents=1)
        print("✓ Centralized plotter enabled")
        
        # Start plot update thread
        stop_event = threading.Event()
        
        def plot_update_worker():
            while not stop_event.is_set():
                try:
                    plotter.update_plots()
                except Exception:
                    pass
                time.sleep(0.1)
        
        plot_thread = threading.Thread(target=plot_update_worker, daemon=True)
        plot_thread.start()
        print("✓ Plot update thread started")
        
        # Load robot configuration
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        print("✓ Robot config loaded")
        
        # Create simple world
        world_cfg = WorldConfig()
        tensor_args = TensorDeviceType()
        
        # Create MPC solver with plot_costs=True
        mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            use_cuda_graph=False,
            step_dt=0.02,
            plot_costs=True,  # This is key!
            store_rollouts=True,
        )
        
        mpc = MpcSolver(mpc_config)
        print("✓ MPC solver created")
        
        # Enable live plotting manually on the ArmReacher instance
        arm_reacher = mpc.rollout_fn
        if hasattr(arm_reacher, 'enable_live_plotting'):
            arm_reacher.enable_live_plotting(True)
            print("✓ Live plotting manually enabled")
        else:
            print("⚠ enable_live_plotting method not found")
        
        # Check if live plotting is enabled
        if hasattr(arm_reacher, '_enable_live_plotting'):
            print(f"✓ Live plotting enabled: {arm_reacher._enable_live_plotting}")
        else:
            print("⚠ _enable_live_plotting attribute not found")
        
        # Set up initial state and goal
        retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        joint_names = mpc.rollout_fn.joint_names
        
        current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
        
        # Create a simple goal (slightly perturbed from retract)
        goal_cfg = retract_cfg.clone()
        goal_cfg[0, 0] += 0.2  # Move first joint
        goal_cfg[0, 1] += 0.3  # Move second joint
        
        goal_state = JointState.from_position(goal_cfg, joint_names=joint_names)
        
        # Compute kinematics for goal pose
        goal_kin_state = mpc.rollout_fn.compute_kinematics(goal_state)
        goal_pose = Pose(goal_kin_state.ee_pos_seq, quaternion=goal_kin_state.ee_quat_seq)
        
        goal = Goal(
            current_state=current_state,
            goal_state=goal_state,
            goal_pose=goal_pose,
        )
        
        goal_buffer = mpc.setup_solve_single(goal, 1)
        print("✓ Goal set up")
        
        # Run optimization steps and check for plotting data
        print("Running MPC steps to generate plotting data...")
        
        for step in range(50):  # Run 50 steps
            try:
                # Update current state slightly
                current_state.position[0, 0] += 0.001 * step
                
                # Step the MPC
                result = mpc.step(current_state, max_attempts=1)
                
                if step % 10 == 0:
                    print(f"  Step {step}: MPC step completed")
                
                # Small delay to see updates
                time.sleep(0.05)
                
            except Exception as e:
                print(f"⚠ Error at step {step}: {e}")
                break
        
        print("✓ MPC steps completed")
        print("Check if plot window opened with cost data!")
        print("Press Ctrl+C to stop...")
        
        # Keep running to show plots
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping test...")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            if 'stop_event' in locals():
                stop_event.set()
            if 'plot_thread' in locals():
                plot_thread.join(timeout=1.0)
            if 'plotter' in locals():
                plotter.shutdown()
            print("✓ Cleanup completed")
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    test_mpc_plotting_integration() 
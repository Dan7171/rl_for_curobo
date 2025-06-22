#!/usr/bin/env python3
"""
Simple test script to validate MPC client-server communication.
This can be run without Isaac Sim to test the basic communication layer.
"""

import sys
import os
import time
import argparse

# Add path for MpcSolverAPI
from .mpc_solver_api import MpcSolverAPI

try:
    from curobo.wrap.reacher.mpc import MpcSolverConfig
    from curobo.geom.sdf.world import CollisionCheckerType
    from curobo.geom.types import WorldConfig
    from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
    from curobo.types.state import JointState
    from curobo.types.base import TensorDeviceType
    import torch
except ImportError as e:
    print(f"Import error: {e}")
    print("This test requires CuRobo and PyTorch. Run in proper environment.")
    sys.exit(1)


def test_basic_communication(server_ip: str, server_port: int):
    """Test basic client-server communication."""
    print("=== Testing MPC Client-Server Communication ===")
    
    try:
        # Load robot configuration  
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        
        # Create a minimal world config
        world_cfg = WorldConfig(cuboid=[], mesh=[])
        
        # Instead of creating MpcSolverConfig here, send raw parameters
        config_params = {
            'robot_cfg': robot_cfg,
            'world_cfg': world_cfg,
            'use_cuda_graph': False,
            'collision_checker_type': CollisionCheckerType.PRIMITIVE,
            'use_mppi': True,
            'use_lbfgs': False,
            'store_rollouts': False,
            'step_dt': 0.02,
            'tensor_args': TensorDeviceType(),
        }
        
        print(f"Connecting to server at {server_ip}:{server_port}...")
        
        # Test 1: Initialize MPC solver
        print("Test 1: Initializing remote MPC solver...")
        mpc_api = MpcSolverAPI(server_ip, server_port, config_params) 
        print("✓ MPC solver initialized successfully")
        
        # Test 2: Access nested attributes
        print("Test 2: Testing nested attribute access...")
        joint_names = mpc_api.rollout_fn.joint_names
        print(f"✓ Joint names: {joint_names}")
        
        # Test 3: Access tensor attributes with operations
        print("Test 3: Testing tensor operations...")
        retract_cfg = mpc_api.rollout_fn.dynamics_model.retract_config
        print(f"✓ Retract config shape: {retract_cfg.shape}")
        
        cloned_cfg = retract_cfg.clone()
        print(f"✓ Cloned config shape: {cloned_cfg.shape}")
        
        unsqueezed_cfg = cloned_cfg.unsqueeze(0)
        print(f"✓ Unsqueezed config shape: {unsqueezed_cfg.shape}")
        
        # Test 4: Method calls with complex objects
        print("Test 4: Testing method calls with JointState...")
        current_state = JointState.from_position(unsqueezed_cfg, joint_names=joint_names)
        
        # Test kinematics computation
        kinematics_state = mpc_api.rollout_fn.compute_kinematics(current_state)
        print(f"✓ Kinematics computed, pose shape: {kinematics_state.ee_pos_seq.shape}")
        
        print("\n=== All tests passed! ===")
        print("The MPC client-server communication is working correctly.")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Main test entry point."""
    parser = argparse.ArgumentParser(description="Test MPC Client-Server Communication")
    parser.add_argument("--server_ip", type=str, default="localhost", help="Server IP address")
    parser.add_argument("--server_port", type=int, default=10051, help="Server port")
    args = parser.parse_args()
    
    print("MPC Client-Server Communication Test")
    print("===================================")
    print(f"Server: {args.server_ip}:{args.server_port}")
    print()
    
    success = test_basic_communication(args.server_ip, args.server_port)
    
    if success:
        print("\n🎉 All tests passed! Ready for full MPC simulation.")
    else:
        print("\n❌ Tests failed. Check server connection and configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main() 
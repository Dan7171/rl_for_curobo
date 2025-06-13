#!/usr/bin/env python3

import sys
import os
sys.path.append('curobo/src')

import yaml
from curobo.wrap.reacher.mpc import MpcSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path

def test_num_particles_loading():
    """Test that _num_particles_rollout_full is properly loaded from YAML config."""
    
    print("Testing particle loading from YAML configs...")
    
    # Test particle_mpc0.yml
    print("\n=== Testing particle_mpc0.yml ===")
    particle_file_0 = 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/override_particle_files_multiple_robots/particle_mpc0.yml'
    
    # Load the YAML directly to see what we expect
    with open(particle_file_0, 'r') as f:
        particle_cfg_0 = yaml.safe_load(f)
    expected_particles_0 = particle_cfg_0['mppi']['num_particles']
    print(f"Expected particles from particle_mpc0.yml: {expected_particles_0}")
    
    # Load using MPC wrapper (this is the correct way)
    try:
        mpc_config_0 = MpcSolverConfig.load_from_robot_config(
            robot_cfg='franka.yml',
            world_model='collision_table.yml',
            override_particle_file=particle_file_0,
            use_cuda_graph=False,  # Disable for testing
        )
        
        # Check the rollout function's particle count
        rollout_fn = mpc_config_0.rollout_fn
        actual_particles_0 = rollout_fn._num_particles_rollout_full
        print(f"Actual particles loaded in rollout_fn: {actual_particles_0}")
        
        if actual_particles_0 == expected_particles_0:
            print("✅ SUCCESS: particle_mpc0.yml particles loaded correctly!")
        else:
            print(f"❌ FAILED: Expected {expected_particles_0}, got {actual_particles_0}")
            
    except Exception as e:
        print(f"❌ ERROR loading particle_mpc0.yml: {e}")
    
    # Test particle_mpc1.yml
    print("\n=== Testing particle_mpc1.yml ===")
    particle_file_1 = 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/override_particle_files_multiple_robots/particle_mpc1.yml'
    
    # Load the YAML directly to see what we expect
    with open(particle_file_1, 'r') as f:
        particle_cfg_1 = yaml.safe_load(f)
    expected_particles_1 = particle_cfg_1['mppi']['num_particles']
    print(f"Expected particles from particle_mpc1.yml: {expected_particles_1}")
    
    # Load using MPC wrapper (this is the correct way)
    try:
        mpc_config_1 = MpcSolverConfig.load_from_robot_config(
            robot_cfg='franka.yml',
            world_model='collision_table.yml',
            override_particle_file=particle_file_1,
            use_cuda_graph=False,  # Disable for testing
        )
        
        # Check the rollout function's particle count
        rollout_fn = mpc_config_1.rollout_fn
        actual_particles_1 = rollout_fn._num_particles_rollout_full
        print(f"Actual particles loaded in rollout_fn: {actual_particles_1}")
        
        if actual_particles_1 == expected_particles_1:
            print("✅ SUCCESS: particle_mpc1.yml particles loaded correctly!")
        else:
            print(f"❌ FAILED: Expected {expected_particles_1}, got {actual_particles_1}")
            
    except Exception as e:
        print(f"❌ ERROR loading particle_mpc1.yml: {e}")
    
    print("\n=== Summary ===")
    print("Both configs should show 400 particles if the loading is working correctly.")

if __name__ == "__main__":
    test_num_particles_loading() 
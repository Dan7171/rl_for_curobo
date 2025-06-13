#!/usr/bin/env python3

from curobo.wrap.reacher.mpc import MpcSolverConfig, MpcSolver
from curobo.rollout.cost.custom.arm_base.dynamic_obs_cost import DynamicObsCost, DynamicObsCostConfig
from curobo.types.base import TensorDeviceType

print("Testing particle count fix...")

# Test 1: Load basic MPC config
try:
    cfg = MpcSolverConfig.load_from_robot_config('franka.yml', 'collision_table.yml')
    print("✅ Basic MPC config loaded successfully")
    
    # Check the actual optimizer's rollout function
    solver = cfg.solver
    if hasattr(solver, 'optimizers') and len(solver.optimizers) > 0:
        optimizer = solver.optimizers[0]  # First optimizer is MPPI
        rollout_fn = optimizer.rollout_fn
        print(f"🔍 Optimizer type: {type(optimizer).__name__}")
        print(f"🔍 Rollout function type: {type(rollout_fn).__name__}")
        
        if hasattr(rollout_fn, 'num_particles_rollout'):
            particle_count = rollout_fn.num_particles_rollout
            print(f"🔍 Optimizer rollout function particle count: {particle_count}")
            
            if particle_count == 400:
                print("✅ SUCCESS: Particle count is correctly set to 400!")
            elif particle_count == -1:
                print("❌ ISSUE: Particle count still at default value -1")
            else:
                print(f"⚠️  Unexpected particle count: {particle_count}")
        else:
            print("❌ Rollout function doesn't have num_particles_rollout property")
    else:
        print("❌ No optimizers found in solver")
        
    # Also check the config's rollout_fn (this is the auxiliary one)
    aux_rollout_fn = cfg.rollout_fn
    if hasattr(aux_rollout_fn, 'num_particles_rollout'):
        aux_particle_count = aux_rollout_fn.num_particles_rollout
        print(f"🔍 Auxiliary rollout function particle count: {aux_particle_count}")
    
    # Test 2: Create custom cost directly
    tensor_args = TensorDeviceType()
    custom_config = DynamicObsCostConfig(
        weight=1.0,
        tensor_args=tensor_args,
        p_R=[0.0, 0.0, 0.0],
        p_T=[0.0, 0.0, 0.0],
        n_own_spheres=61,
        n_obs=61,
        manually_express_p_own_in_world_frame=False,
        safety_margin=0.1,
        num_particles_rollout=400,
        horizon_rollout=30
    )
    
    custom_cost = DynamicObsCost(custom_config)
    print("✅ Custom cost created successfully with explicit particle count")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("Test complete!") 
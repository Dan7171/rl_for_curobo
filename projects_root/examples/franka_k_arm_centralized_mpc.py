#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

"""
Franka K-Arm Centralized MPC Controller

This script provides centralized MPC control for multiple Franka arms (k=2, k=3, or more).
It builds upon the proven dual-arm framework to support arbitrary numbers of Franka arms.

Usage examples:
    # 2-arm system with standard dual-arm config
    python franka_k_arm_centralized_mpc.py --robot franka_dual_arm.yml --num_arms 2
    
    # 3-arm system with generated config  
    python franka_k_arm_centralized_mpc.py --robot test_generated_configs/franka_3_arm.yml --num_arms 3
    
    # 3-arm system with standard config (if available)
    python franka_k_arm_centralized_mpc.py --robot franka_triple_arm.yml --num_arms 3
"""

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import os

## import curobo:

parser = argparse.ArgumentParser(description="Franka K-Arm Centralized MPC Controller")

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)

parser.add_argument(
    "--robot", 
    type=str, 
    default="franka_dual_arm.yml", 
    help="robot configuration to load (supports franka_dual_arm.yml, franka_triple_arm.yml, or custom generated configs)"
)
parser.add_argument(
    "--num_arms", 
    type=int, 
    default=2, 
    help="number of Franka arms in the system (2, 3, or more)"
)
parser.add_argument(
    "--override_particle_file", 
    type=str, 
    default=None, 
    help="override particle MPC config file"
)
parser.add_argument(
    "--arm_spacing",
    type=float,
    default=0.8,
    help="spacing between arm targets in meters (default: 0.8m)"
)

args = parser.parse_args()

###########################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
 
# Standard Library
import carb
from helper import add_robot_to_scene, add_extensions
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects import cuboid, sphere

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

############################################################

EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")


def draw_points(rollouts: torch.Tensor):
    """Visualize MPC rollout trajectories."""
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
    draw.clear_points()
    cpu_rollouts = rollouts.cpu().numpy()
    b, h, _ = cpu_rollouts.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [
            (cpu_rollouts[i, j, 0], cpu_rollouts[i, j, 1], cpu_rollouts[i, j, 2]) for j in range(h)
        ]
        colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
    sizes = [10.0 for _ in range(b * h)]
    draw.draw_points(point_list, colors, sizes)


def create_arm_targets(num_arms: int, my_world: World, arm_spacing: float = 0.8):
    """Create target cubes for K Franka arms. Returns list of target objects."""
    targets = []
    
    # Generate distinct colors for each arm
    colors = []
    for i in range(num_arms):
        # Generate colors in HSV space for better distribution
        hue = (i * 360 / num_arms) % 360
        # Convert HSV to RGB (simplified)
        if hue < 120:
            r, g, b = 1.0, hue/120, 0.0
        elif hue < 240:
            r, g, b = (240-hue)/120, 1.0, 0.0
        else:
            r, g, b = 0.0, (360-hue)/120, 1.0
        colors.append([r, g, b])
    
    # Create target cubes positioned for each arm
    for i in range(num_arms):
        if num_arms == 2:
            # Dual arm: left and right positions (matching working dual-arm script)
            x_pos = 0.2 + (i * 0.6)  # 0.2, 0.8
            y_pos = 0.3             # Same Y for both
        elif num_arms == 3:
            # Triple arm: left, center, right
            x_pos = 0.2 + (i * 0.3)  # 0.2, 0.5, 0.8
            y_pos = 0.3
        else:
            # General case: distribute evenly
            x_pos = 0.2 + (i * 0.6 / (num_arms - 1)) if num_arms > 1 else 0.5
            y_pos = 0.3
            
        z_pos = 0.5  # Standard height for Franka workspace
        
        target = cuboid.VisualCuboid(
            f"/World/arm_{i}_target",
            position=np.array([x_pos, y_pos, z_pos]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array(colors[i]),
            size=0.05,
        )
        targets.append(target)
    
    return targets


def detect_config_type(robot_config_path: str) -> Tuple[str, int]:
    """
    Detect the type of robot configuration and number of arms.
    
    Returns:
        Tuple of (config_type, num_arms)
        config_type: 'dual_arm', 'triple_arm', 'generated', 'unknown'
    """
    config_name = os.path.basename(robot_config_path).lower()
    
    if 'dual' in config_name or config_name == 'franka_dual_arm.yml':
        return 'dual_arm', 2
    elif 'triple' in config_name or config_name == 'franka_triple_arm.yml':
        return 'triple_arm', 3
    elif 'franka_3_arm' in config_name:
        return 'generated', 3
    elif 'franka_2_arm' in config_name:
        return 'generated', 2
    else:
        # Try to infer from config content
        try:
            config = load_yaml(robot_config_path)
            joint_names = config["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
            # Count arm joints (each Franka arm has 7 main joints + 2 finger joints = 9 total)
            arm_count = len(joint_names) // 9
            return 'generated', arm_count
        except:
            return 'unknown', 2  # Default to dual arm


def get_particle_config_path(num_arms: int, override_file: Optional[str] = None) -> str:
    """Get the appropriate particle MPC configuration file path."""
    if override_file is not None:
        return override_file
        
    # Standard particle configs based on number of arms
    if num_arms == 2:
        return 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_dual_arm.yml'
    elif num_arms == 3:
        return 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_triple_arm.yml'
    elif num_arms == 4:
        return 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_quad_arm.yml'
    else:
        # For arbitrary K, generate config on the fly
        try:
            from projects_root.utils.multi_arm_config_generator import MultiArmConfigGenerator
            generator = MultiArmConfigGenerator("tmp_configs")
            config_path = generator.generate_particle_mpc_config(num_arms, f"franka_{num_arms}_arm")
            print(f"Generated particle MPC config for {num_arms} arms: {config_path}")
            return config_path
        except ImportError:
            print(f"Warning: Cannot generate particle config for {num_arms} arms, using dual-arm default")
            return 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_dual_arm.yml'


def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # Determine robot configuration path
    if os.path.isabs(args.robot):
        robot_config_path = args.robot
    elif args.robot.startswith('./') or args.robot.startswith('../') or os.path.exists(args.robot):
        robot_config_path = args.robot
    else:
        robot_config_path = join_path(get_robot_configs_path(), args.robot)
    
    print(f"Loading Franka {args.num_arms}-arm configuration: {robot_config_path}")

    # Create K arm targets
    arm_targets = create_arm_targets(args.num_arms, my_world, args.arm_spacing)

    setup_curobo_logger("warn")
    past_poses = [None] * args.num_arms  # Track past poses for each arm
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(robot_config_path)["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    init_curobo = False

    tensor_args = TensorDeviceType()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].pose[2] = -10.0

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # Get particle MPC configuration
    particle_config_path = get_particle_config_path(args.num_arms, args.override_particle_file)

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=False,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.02,
        override_particle_file=particle_config_path
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    
    # Debug: Check what the retract pose looks like for multi-arm
    print(f"=== Retract Pose Debug ===")
    print(f"retract_cfg shape: {retract_cfg.shape}")
    print(f"state.ee_pos_seq shape: {state.ee_pos_seq.shape}")
    print(f"state.ee_quat_seq shape: {state.ee_quat_seq.shape}")
    print(f"=== End Retract Pose Debug ===")
    
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    
    # Initialize with default target positions for K arms
    # Create initial multi-arm target positions (spread along x-axis)
    initial_positions = []
    initial_quaternions = []
    base_x = 0.2  # Starting x position
    target_y = 0.3  # Fixed y position
    target_z = 0.5  # Fixed z position
    
    for i in range(args.num_arms):
        x_pos = base_x + i * 0.3  # Space arms 0.3m apart in x
        initial_positions.append([x_pos, target_y, target_z])
        initial_quaternions.append([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    
    # Create initial multi-arm goal pose with explicit batch dimension
    initial_goal_positions = torch.tensor(initial_positions, device=tensor_args.device, dtype=tensor_args.dtype)
    initial_goal_quaternions = torch.tensor(initial_quaternions, device=tensor_args.device, dtype=tensor_args.dtype)
    
    # FIXED: Add batch dimension to ensure Goal.batch = 1 instead of num_arms
    # Format: [1, num_arms, 3] and [1, num_arms, 4] for single problem with multiple arms
    initial_goal_positions = initial_goal_positions.unsqueeze(0)  # [1, num_arms, 3]
    initial_goal_quaternions = initial_goal_quaternions.unsqueeze(0)  # [1, num_arms, 4]
    
    initial_goal_pose = Pose(position=initial_goal_positions, quaternion=initial_goal_quaternions)
    
    # Debug: Check initial goal pose shapes
    print(f"=== Initial Goal Pose Debug ===")
    print(f"initial_goal_positions shape: {initial_goal_positions.shape}")
    print(f"initial_goal_quaternions shape: {initial_goal_quaternions.shape}")
    print(f"=== End Initial Goal Pose Debug ===")
    
    # For K-arm, we use the multi-arm pose structure for goals
    # FIXED: Don't use retract_pose since it only has single end-effector data
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=initial_goal_pose,  # Use proper multi-arm target with correct tensor shapes
    )

    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)

    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None
    step = 0
    spheres = None  # For collision sphere visualization
    add_extensions(simulation_app, args.headless_mode)
    
    print(f"Initialized Franka {args.num_arms}-arm centralized MPC system")
    print(f"Using robot config: {robot_config_path}")
    print(f"Using particle file: {particle_config_path}")
    
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 10:
            robot._articulation_view.initialize()

            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step
        
        if step_index % 1000 == 0:
            print("Updating world")
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                ignore_substring=[
                    robot_prim_path,
                    "/World/defaultGroundPlane",
                    "/curobo",
                ] + [f"/World/arm_{i}_target" for i in range(args.num_arms)],  # Ignore all arm targets
                reference_prim_path=robot_prim_path,
            )
            obstacles.add_obstacle(world_cfg_table.cuboid[0])
            mpc.world_coll_checker.load_collision_model(obstacles)

        # Check if any arm target has moved (similar to dual-arm logic)
        arm_positions = []
        arm_orientations = []
        targets_changed = False
        
        for i, target in enumerate(arm_targets):
            current_pos, current_quat = target.get_world_pose()
            
            if past_poses[i] is None:
                past_poses[i] = current_pos + 1.0  # Force initial change
                
            # Check if this arm's target changed
            if np.linalg.norm(current_pos - past_poses[i]) > 1e-3:
                targets_changed = True
                past_poses[i] = current_pos
                
            arm_positions.append(tensor_args.to_device(current_pos))
            arm_orientations.append(tensor_args.to_device(current_quat))
        
        if targets_changed:
            # Create multi-arm goals: [1, num_arms, 3] for positions, [1, num_arms, 4] for quaternions
            # FIXED: Add batch dimension to ensure Goal.batch = 1 instead of num_arms
            multi_arm_positions = torch.stack(arm_positions, dim=0)  # [num_arms, 3]
            multi_arm_quaternions = torch.stack(arm_orientations, dim=0)  # [num_arms, 4]
            
            # Add batch dimension to match initial goal format
            multi_arm_positions = multi_arm_positions.unsqueeze(0)  # [1, num_arms, 3]
            multi_arm_quaternions = multi_arm_quaternions.unsqueeze(0)  # [1, num_arms, 4]
            
            # Create multi-arm pose goal
            ik_goal = Pose(
                position=multi_arm_positions,     # [1, num_arms, 3] tensor for all arms
                quaternion=multi_arm_quaternions  # [1, num_arms, 4] tensor for all arms
            )
            
            # Debug goal tensor shapes (reduced output)
            if step % 100 == 0:  # Only print occasionally
                print(f"--- Goal Tensor Debug ---")
                print(f"Goal position shape: {ik_goal.position.shape}")
                print(f"Goal quaternion shape: {ik_goal.quaternion.shape}")
                print(f"Goal position values: {ik_goal.position}")
                print(f"Goal quaternion values: {ik_goal.quaternion}")
                print("--- End Goal Debug ---")
            
            # IMPORTANT: Direct assignment instead of copy_() to preserve multi-arm structure
            goal_buffer.goal_pose = ik_goal
            mpc.update_goal(goal_buffer)
            
            # Debug: Force MPC to reset its internal state to avoid getting stuck
            if hasattr(mpc, 'reset') and step % 50 == 0:  # Reset every 50 steps when targets change
                print("Resetting MPC internal state to avoid local minima")
                mpc.reset()
            
            print(f"Updated {args.num_arms}-arm goals:")
            for i, pos in enumerate(arm_positions):
                print(f"  Arm {i} target: {past_poses[i]}")
            print(f"  Goal position shape: {goal_buffer.goal_pose.position.shape}")
            print(f"  Goal quaternion shape: {goal_buffer.goal_pose.quaternion.shape}")

        # Get robot current state:
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        js_names = robot.dof_names
        sim_js_names = robot.dof_names

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        
        # Collision sphere visualization
        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = mpc.kinematics.get_robot_as_spheres(cu_js.position)

            if spheres is None:
                spheres = []
                # create spheres:
                for si, s in enumerate(sph_list[0]):
                    sp = sphere.VisualSphere(
                        prim_path="/curobo/robot_sphere_" + str(si),
                        position=np.ravel(s.position),
                        radius=float(s.radius),
                        color=np.array([0, 0.8, 0.2]),
                    )
                    spheres.append(sp)
            else:
                for si, s in enumerate(sph_list[0]):
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))
        
        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names
            
        common_js_names = []
        current_state.copy_(cu_js)

        # Try with more attempts if we're getting stuck
        max_attempts = 5 if step % 100 == 0 else 2
        mpc_result = mpc.step(current_state, max_attempts=max_attempts)

        # Debug MPC result occasionally
        if step % 100 == 0:
            print(f"\n=== MPC Debug (step {step}) ===")
            print(f"MPC solve time: {mpc_result.solve_time:.4f}s")
            if hasattr(mpc_result, 'metrics') and mpc_result.metrics is not None:
                if hasattr(mpc_result.metrics, 'feasible'):
                    print(f"MPC feasible: {mpc_result.metrics.feasible}")
                if hasattr(mpc_result.metrics, 'cost'):
                    print(f"MPC cost: {mpc_result.metrics.cost}")
            if mpc_result.js_action is not None:
                print(f"Action shape: {mpc_result.js_action.position.shape}")
                print(f"Action velocity norm: {torch.norm(mpc_result.js_action.velocity).item():.6f}")
                print(f"Action position sample (first 6 joints): {mpc_result.js_action.position[0, :6].cpu().numpy()}")
                
                # Debug joint names and EE link names for multi-arm
                if args.num_arms >= 3:
                    print(f"Robot joint names ({len(mpc.rollout_fn.joint_names)}): {mpc.rollout_fn.joint_names[:12]}...")  # First 12 joints
                    if hasattr(mpc.rollout_fn, 'kinematics'):
                        print(f"EE link name: {mpc.rollout_fn.kinematics.ee_link}")
                        if hasattr(mpc.rollout_fn.kinematics, 'link_names'):
                            print(f"All link names: {mpc.rollout_fn.kinematics.link_names}")
                    
                    # Check current robot state and EE positions
                    state = mpc.rollout_fn.compute_kinematics(current_state)
                    if state.ee_pos_seq is not None:
                        print(f"EE position tensor shape: {state.ee_pos_seq.shape}")
                        # Handle different tensor dimensions safely
                        if len(state.ee_pos_seq.shape) == 3:
                            print(f"Current EE position: {state.ee_pos_seq[0, 0, :].cpu().numpy()}")
                        elif len(state.ee_pos_seq.shape) == 2:
                            print(f"Current EE position: {state.ee_pos_seq[0, :].cpu().numpy()}")
                        else:
                            print(f"Unexpected EE tensor shape, first values: {state.ee_pos_seq.flatten()[:6].cpu().numpy()}")
                    
                    if hasattr(state, 'link_pose') and state.link_pose is not None:
                        print(f"Available link poses: {list(state.link_pose.keys())}")
                        # Check if we have poses for expected arm links
                        expected_links = [f'arm_{i}_panda_hand' for i in range(args.num_arms)]
                        for i, link in enumerate(expected_links):
                            if link in state.link_pose:
                                pos_tensor = state.link_pose[link].position
                                print(f"  {link} position tensor shape: {pos_tensor.shape}")
                                if len(pos_tensor.shape) >= 3:
                                    pos = pos_tensor[0, 0, :].cpu().numpy()
                                    print(f"  {link} position: {pos}")
                                else:
                                    print(f"  {link} position: {pos_tensor.flatten()[:3].cpu().numpy()}")
                            else:
                                print(f"  {link}: MISSING!")
            else:
                print("No action generated!")
            print("=== End MPC Debug ===\n")

        succ = True
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state_full = cmd_state

        art_action = ArticulationAction(
            cmd_state.position.view(-1).cpu().numpy(),
            joint_indices=idx_list,
        )
        
        if step_index % 100 == 0:
            print(f"MPC feasible: {mpc_result.metrics.feasible.item()}, pose_error: {mpc_result.metrics.pose_error.item():.6f}")
            print(f"Applying action to {len(idx_list)} joints")
            print(f"Joint action sample: {art_action.joint_positions[:6]}")

        if succ:
            # Set desired joint angles obtained from MPC:
            for _ in range(1):
                articulation_controller.apply_action(art_action)
        else:
            if step_index % 100 == 0:
                print("WARNING: No action is being taken - MPC failed!")
            carb.log_warn("No action is being taken.")


############################################################

if __name__ == "__main__":
    main()
    simulation_app.close() 
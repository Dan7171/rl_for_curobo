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

parser = argparse.ArgumentParser(
    description="CuRobo MPC example with moving obstacle in Isaac Sim",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Default behavior (red cuboid moving at -0.1 m/s in x direction)
  omni_python mpc_example_with_moving_obstacle.py

  # Sphere obstacle moving diagonally with autoplay
  omni_python mpc_example_with_moving_obstacle.py --obstacle_type sphere --obstacle_velocity -0.1 0.1 0.0 --obstacle_size 0.15 --autoplay

  # Blue cuboid starting at specific position with physics
  omni_python mpc_example_with_moving_obstacle.py --obstacle_type cuboid --obstacle_initial_pose 1.0 0.5 0.3 --obstacle_color 0.0 0.0 1.0 --enable_physics --obstacle_mass 1.0 --obstacle_friction 0.5

  # Green sphere moving in y direction with custom size and physics
  omni_python mpc_example_with_moving_obstacle.py --obstacle_type sphere --obstacle_velocity 0.0 0.1 0.0 --obstacle_size 0.2 --obstacle_color 0.0 1.0 0.0 --enable_physics --obstacle_mass 2.0 --obstacle_restitution 0.8
"""
)

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="Run in headless mode. Options: [native, websocket]. Note: webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="Visualize robot collision spheres",
    default=False,
)
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot configuration file to load (e.g., franka.yml)",
)
parser.add_argument(
    "--obstacle_velocity",
    type=float,
    nargs=3,
    default=[-0.1, 0.0, 0.0],
    help="Velocity of the obstacle in x, y, z (m/s). Example: --obstacle_velocity -0.1 0.0 0.0",
)
parser.add_argument(
    "--obstacle_type",
    type=str,
    choices=["cuboid", "sphere"],
    default="cuboid",
    help="Type of obstacle to create (cuboid or sphere)",
)
parser.add_argument(
    "--obstacle_size",
    type=float,
    default=0.1,
    help="Size of the obstacle (diameter for sphere, side length for cuboid) in meters",
)
parser.add_argument(
    "--obstacle_initial_pose",
    type=float,
    nargs=3,
    default=[0.8, 0.0, 0.5],
    help="Initial position of the obstacle in x, y, z (meters). Example: --obstacle_initial_pose 0.8 0.0 0.5",
)
parser.add_argument(
    "--obstacle_color",
    type=float,
    nargs=3,
    default=[1.0, 0.0, 0.0],
    help="RGB color of the obstacle (values between 0 and 1). Example: --obstacle_color 1.0 0.0 0.0 for red",
)
parser.add_argument(
    "--autoplay",
    action="store_true",
    help="Start simulation automatically without requiring manual play button press",
    default=False,
)
parser.add_argument(
    "--enable_physics",
    action="store_true",
    help="Enable physical collision between obstacle and robot",
    default=False,
)
parser.add_argument(
    "--obstacle_mass",
    type=float,
    default=1.0,
    help="Mass of the obstacle in kilograms",
)
parser.add_argument(
    "--obstacle_friction",
    type=float,
    default=0.5,
    help="Friction coefficient of the obstacle (0-1)",
)
parser.add_argument(
    "--obstacle_restitution",
    type=float,
    default=0.8,
    help="Restitution (bounciness) coefficient of the obstacle (0-1)",
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
import os

# Third Party
import carb
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
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

def create_moving_obstacle(world, position, size=0.1, obstacle_type="cuboid", color=None, enable_physics=False, mass=1.0, friction=0.5, restitution=0.8):
    if color is None:
        color = np.array([0.0, 0.0, 0.1])  # Default blue color
    
    if obstacle_type == "cuboid":
        # Create a cube obstacle # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/search.html?q=DynamicSphere
        if enable_physics:
            from omni.isaac.core.objects import DynamicCuboid
            obstacle = world.scene.add(
                DynamicCuboid(
                    prim_path="/World/moving_obstacle",
                    name="moving_obstacle",
                    position=position,
                    size=size,
                    color=color,
                    mass=mass,
                    static_friction=friction,
                    dynamic_friction=friction,
                    restitution=restitution,
                )
            )
        else:
            obstacle = world.scene.add(
                cuboid.VisualCuboid(
                    prim_path="/World/moving_obstacle",
                    name="moving_obstacle",
                    position=position,
                    size=size,
                    color=color,
                )
            )
    elif obstacle_type == "sphere":
        # Create a sphere obstacle
        from omni.isaac.core.objects import sphere
        if enable_physics:
            from omni.isaac.core.objects import DynamicSphere
            obstacle = world.scene.add(
                DynamicSphere(
                    prim_path="/World/moving_obstacle",
                    name="moving_obstacle",
                    position=position,
                    radius=size/2,
                    color=color,
                    mass=mass,
                    # friction=friction,
                    # restitution=restitution,
                )
            )
        else:
            obstacle = world.scene.add(
                sphere.VisualSphere(
                    prim_path="/World/moving_obstacle",
                    name="moving_obstacle",
                    position=position,
                    radius=size/2,
                    color=color,
                )
            )
    return obstacle

def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform") # https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/UsdGeom.html , https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr/UsdGeom.html#pxr.UsdGeom.Xform
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    my_world.scene.add_default_ground_plane()

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

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

    # Create moving obstacle
    initial_pos = np.array(args.obstacle_initial_pose)  # Starting position from command line
    obstacle = create_moving_obstacle(
        my_world,
        initial_pos,
        args.obstacle_size,
        args.obstacle_type,
        np.array(args.obstacle_color),
        args.enable_physics,
        args.obstacle_mass,
        args.obstacle_friction,
        args.obstacle_restitution
    )
    
    # Define obstacle velocity (in m/s)
    obstacle_velocity = np.array(args.obstacle_velocity)
    
    if args.enable_physics:
        # Set initial velocity for physical obstacle
        obstacle.set_linear_velocity(obstacle_velocity)
    else:
        # For non-physical obstacle, we'll update position manually
        current_position = initial_pos
        dt = 1.0/60.0  # Simulation timestep

    # Add moving obstacle to world configuration
    if args.obstacle_type == "cuboid":
        moving_obstacle = Cuboid(
            name="moving_obstacle",
            pose=[initial_pos[0], initial_pos[1], initial_pos[2], 1.0, 0.0, 0.0, 0.0],
            dims=[args.obstacle_size, args.obstacle_size, args.obstacle_size],
        )
    else:  # sphere
        from curobo.geom.types import Sphere
        moving_obstacle = Sphere(
            name="moving_obstacle",
            pose=[initial_pos[0], initial_pos[1], initial_pos[2], 1.0, 0.0, 0.0, 0.0],
            radius=args.obstacle_size/2,
        )
    world_cfg.add_obstacle(moving_obstacle)

    init_curobo = False

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=True,
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
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=retract_pose,
    )

    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)

    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None
    step = 0
    add_extensions(simulation_app, args.headless_mode)
    
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
            if args.autoplay:
                my_world.play()
        draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            if args.autoplay:
                my_world.play()
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 2:
            my_world.reset()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step

        # Update obstacle position based on physics mode
        if not args.enable_physics:
            current_position = current_position + obstacle_velocity * dt
            obstacle.set_world_pose(current_position)
        else:
            current_position = obstacle.get_world_pose()[0]

        # Update obstacle position in world configuration
        moving_obstacle.pose = [current_position[0], current_position[1], current_position[2], 1.0, 0.0, 0.0, 0.0]
        mpc.world_coll_checker.load_collision_model(world_cfg)

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose()

        if past_pose is None:
            past_pose = cube_position + 1.0

        if np.linalg.norm(cube_position - past_pose) > 1e-3:
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            past_pose = cube_position

        # get robot current state:
        sim_js = robot.get_joints_state()
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

        mpc_result = mpc.step(current_state, max_attempts=2)

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
            cmd_state.position.cpu().numpy(),
            joint_indices=idx_list,
        )
        if step_index % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        if succ:
            # set desired joint angles obtained from IK:
            for _ in range(3):
                articulation_controller.apply_action(art_action)

        else:
            carb.log_warn("No action is being taken.")

if __name__ == "__main__":
    main()
    simulation_app.close() 
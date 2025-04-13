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
import time
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--external_asset_path",
    type=str,
    default=None,
    help="Path to external assets when loading an externally located robot",
)
parser.add_argument(
    "--external_robot_configs_path",
    type=str,
    default=None,
    help="Path to external robot config when loading an external robot",
)

parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--reactive",
    action="store_true",
    help="When True, runs in reactive mode",
    default=False,
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)

parser.add_argument(
    "--reach_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Reach partial pose",
    type=float,
    default=None,
)
parser.add_argument(
    "--hold_partial_pose",
    nargs=6,
    metavar=("qx", "qy", "qz", "x", "y", "z"),
    help="Hold partial pose while moving to goal",
    type=float,
    default=None,
)

parser.add_argument(
    "--autoplay",
    help="Start simulation automatically without requiring manual play button press",
    default="True",
    type=str,
    choices=["True", "False"],
)

args = parser.parse_args()

############################################################

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
from typing import Dict

# Third Party
import carb
import numpy as np
from projects_root.utils.helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)

############################################################


########### OV #################;;;;;

def is_new_global_plan_needed(sim_js, cube_position, cube_orientation, past_pose, past_orientation, target_pose, target_orientation):
    """
    This function checks if a new global plan is needed. Just a cosmetic change in the next code block from the original code. See below

    args:

        sim_js: current joint state of the robot

        # "target" - where the robot is trying to reach now
        target_pose: the current target position which robot is trying to reach now (but it doesent necessarily the updated location of the target cube)
        target_orientation: the current target position which robot is trying to reach now (but it doesent necessarily the updated location of the target cube)
    
        # "cube" - the updated target pose (maybe on the move, could be the target pose or a new target pose if it moved)
        cube_position: current time step position of the new (in amove maybe) target pose
        cube_orientation: current time step orientation of the new (in a move maybe) target pose
        
        # "past" - like cube, but one time step earlier        
        past_pose: previous time step position of the new (in amove maybe) target pose
        past_orientation: previous time step position of the new (in a move maybe) target pose
        
    REPLACED NEXT BLOXK IN OLDER CODE:
    "
    robot_static = False
    if (np.max(np.abs(sim_js.velocities)) < 0.2) or args.reactive:
        robot_static = True
    if (
        (
            np.linalg.norm(cube_position - target_pose) > 1e-3
            or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
        )
        and np.linalg.norm(past_pose - cube_position) == 0.0
        and np.linalg.norm(past_orientation - cube_orientation) == 0.0
        and robot_static
    ) 
    "

    """
    
    
    is_robot_static = np.max(np.abs(sim_js.velocities)) < 0.2
    allow_robot_to_replan = is_robot_static or args.reactive # robot is allowed to replan global plan if stopped (in the non-reactive mode) or anytime in the reactive mode
    is_target_pose_changed = np.linalg.norm(cube_position - target_pose) > 1e-3 or np.linalg.norm(cube_orientation - target_orientation) > 1e-3 # cube position is the updated target pose (which has moved). target_pose is the previous target poseis the
    is_target_is_static = np.linalg.norm(past_pose - cube_position) == 0.0 and np.linalg.norm(past_orientation - cube_orientation) == 0.0 # cube po
    
    return is_robot_static and is_target_pose_changed and is_target_is_static

def main():
    # create a curobo motion gen instance:
    num_targets = 0 # The number of "reachible" targets: number of the target poses which not only set to the robot to is trying to find a route to,but only the ones it succesed (to find a sucessfull global path, not yet executed in controller).
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose = None # previous (static pose) goal pose
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 100

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()
    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]

    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    articulation_controller = None

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    max_attempts = 4
    interpolation_dt = 0.05
    enable_finetune_trajopt = True
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt
        enable_finetune_trajopt = False
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        num_trajopt_seeds=12,
        num_graph_seeds=12,
        interpolation_dt=interpolation_dt,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        trim_steps=trim_steps,
    )
    motion_gen = MotionGen(motion_gen_config) # global planner
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)

    print("Curobo is Ready")

    add_extensions(simulation_app, args.headless_mode)

    plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=2,
        max_attempts=max_attempts,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5 if not args.reactive else 1.0,
    )

    usd_help.load_stage(my_world.stage)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")

    cmd_plan = None
    cmd_idx = 0
    my_world.scene.add_default_ground_plane()
    i = 0
    spheres = None
    past_cmd = None
    target_orientation = None
    past_orientation = None
    pose_metric = None

    

    while simulation_app.is_running():
        
        my_world.step(render=True) # Step the physics simulation while rendering. https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.core/docs/index.html?highlight=play#module-omni.isaac.core.world
        
        # wait until the play button is pressed (unless autoplay is true)
        if not my_world.is_playing():
            if args.autoplay: 
                my_world.play() # https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.core/docs/index.html?highlight=play#module-omni.isaac.core.world
            else:
                if i % 100 == 0:
                    print("**** Click Play to start simulation *****")
                i += 1
                continue

        # get the current step index
        step_index = my_world.current_time_step_index # starts from 1
        # print("step index debug ",step_index)

        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller() # https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.core/docs/index.html?highlight=play#module-omni.isaac.core.world
        
        if step_index < 2: # first iteration only 
            my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if step_index < 20: # TODO: why this is needed?
            continue

        if step_index == 50 or step_index % 1000 == 0.0: # TODO: why this is needed?
            print(f'step index: {step_index}')
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage( # reading obstacles state from stage
                # only_paths=[obstacles_path],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
            ).get_collision_check_world()
            print(f'len(obstacles.objects): {len(obstacles.objects)}')
            motion_gen.update_world(obstacles) # update the world representation in curobo with the new info about obstacles
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose() # reading updated target (goal pose) state from stage

        if past_pose is None: # first time step
            past_pose = cube_position
        if target_pose is None: # first time step
            target_pose = cube_position
        if target_orientation is None: # first time step
            target_orientation = cube_orientation
        if past_orientation is None: # first time step
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state() # reading current joint state from robot
        sim_js_names = robot.dof_names # reading current joint names from robot
        if np.any(np.isnan(sim_js.positions)): # check if any joint position is NaN
            log_error("isaac sim has returned NAN joint position values.")
        cu_js = JointState( # creating a joint state object for curobo
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),  # * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        if not args.reactive: # In reactive mode, we will not wait for a complete stopping of the robot before navigating to a new goal pose (if goal pose has changed). In the default mode on the other hand, we will wait for the robot to stop.
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0

        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        # visualize the robot as spheres
        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

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

        # check if a new global plan is needed, in case the cube (representing the target pose) has moved and the robot is able to move as well (depends in the reactive mode)
        if is_new_global_plan_needed(sim_js, cube_position, cube_orientation, past_pose, past_orientation, target_pose, target_orientation):
            
            print("Replanning a new global plan - goal pose has changed!")
            
            # Set EE teleop goals, use cube for simple non-vr init:
            ee_translation_goal = cube_position # cube position is the updated target pose (which has moved) 
            ee_orientation_teleop_goal = cube_orientation # cube orientation is the updated target orientation (which has moved)

            # compute curobo solution:
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            plan_config.pose_cost_metric = pose_metric
            start_state = cu_js.unsqueeze(0) # cu_js is the current joint state of the robot
            goal_pose = ik_goal # ik_goal is the updated target pose (which has moved)
            
            result: MotionGenResult = motion_gen.plan_single(start_state, goal_pose, plan_config) # https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGen.plan_single:~:text=GraphResult-,plan_single,-( , https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenResult:~:text=class-,MotionGenResult,-(
            succ = result.success.item()  # an attribute of this returned object that signifies whether a trajectory was successfully generated. success tensor with index referring to the batch index.
            
            if num_targets == 1: # it's 1 only immediately after the first time it found a successfull plan for the FIRST time (first target).
                if args.constrain_grasp_approach:
                    # cuRobo also can enable constrained motions for part of a trajectory.
                    # This is useful in pick and place tasks where traditionally the robot goes to an offset pose (pre-grasp pose) and then moves 
                    # to the grasp pose in a linear motion along 1 axis (e.g., z axis) while also constraining it’s orientation. We can formulate this two step process as a single trajectory optimization problem, with orientation and linear motion costs activated for the second portion of the timesteps. 
                    # https://curobo.org/advanced_examples/3_constrained_planning.html#:~:text=Grasp%20Approach%20Vector,behavior%20as%20below.
                    # Enables moving to a pregrasp and then locked orientation movement to final grasp.
                    # Since this is added as a cost, the trajectory will not reach the exact offset, instead it will try to take a blended path to the final grasp without stopping at the offset.
                    # https://curobo.org/_api/curobo.rollout.cost.pose_cost.html#curobo.rollout.cost.pose_cost.PoseCostMetric.create_grasp_approach_metric
                    pose_metric = PoseCostMetric.create_grasp_approach_metric() # 
                if args.reach_partial_pose is not None:
                    # This is probably a way to update the cost metric for reaching a partial pose reaching (not sure how, no documentation).
                    reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                    pose_metric = PoseCostMetric(
                        reach_partial_pose=True, reach_vec_weight=reach_vec
                    )
                if args.hold_partial_pose is not None:
                    # This is probably a way to update the cost metric for reaching a partial pose reaching (not sure how, no documentation).
                    hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                    pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
            
            if succ: 
                print(f"target counter - targets with a reachible plan = {num_targets}") # the number of the targets which are defined by curobo (after being static and ready to plan to) and have a successfull a plan for.
                num_targets += 1
                cmd_plan = result.get_interpolated_plan() # TODO: To clarify myself what get_interpolated_plan() is doing to the initial "result"  does. Also see https://curobo.org/get_started/2a_python_examples.html#:~:text=result%20%3D%20motion_gen.plan_single(start_state%2C%20goal_pose%2C%20MotionGenPlanConfig(max_attempts%3D1))%0Atraj%20%3D%20result.get_interpolated_plan()%20%20%23%20result.interpolation_dt%20has%20the%20dt%20between%20timesteps%0Aprint(%22Trajectory%20Generated%3A%20%22%2C%20result.success)
                cmd_plan = motion_gen.get_full_js(cmd_plan) # get the full joint state from the interpolated plan
                # get only joint names that are in both:
                idx_list = []
                common_js_names = []
                for x in sim_js_names:
                    if x in cmd_plan.joint_names:
                        idx_list.append(robot.get_dof_index(x))
                        common_js_names.append(x)
                # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

                cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

                cmd_idx = 0

            else:
                carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            target_pose = cube_position
            target_orientation = cube_orientation
            
        past_pose = cube_position
        past_orientation = cube_orientation
        if cmd_plan is not None:
            print(f"debug - plan found, step_index={step_index}, cmd_idx = {cmd_idx}, num_targets = {num_targets} ")
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy(),
                joint_indices=idx_list,
            )
            # set desired joint angles obtained from IK:
            articulation_controller.apply_action(art_action)
            cmd_idx += 1 # the index of the next command to execute in the plan
            for _ in range(2):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position): # NOTE: all cmd_plans (global plans) are at the same length from my observations (currently 61), no matter how many time steps (step_indexes) take to complete the plan.
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
    simulation_app.close()


if __name__ == "__main__":
    main()

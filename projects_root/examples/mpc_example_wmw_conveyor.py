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

import sys, os
# dynamic imports in container- temp solution instead of 'omni_python - pip install .' (in rl_for_curobo dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

## import curobo:

parser = argparse.ArgumentParser()

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

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
parser.add_argument(
    "--robot_base_frame",
    nargs=7,
    metavar=("x", "y", "z", "qw", "qx", "qy", "qz"),
    help="Robot base frame pose [x, y, z, qw, qx, qy, qz]",
    type=float,
    default=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
)

# Optional flag: use cuboid (OBB) approximation for non-cuboid obstacles.
parser.add_argument(
    "--use_obb_approx",
    action="store_true",
    help="Approximate analytic shapes (capsules/cylinders/spheres) with oriented bounding boxes for faster collision checks.",
    default=False,
)

# Optional: decimate complex meshes before loading to GPU.
parser.add_argument(
    "--max_mesh_faces",
    type=int,
    default=100, # 0 means no decimation, 50 considered small
    help="If > 0, simplify mesh obstacles to at most this many faces using trimesh QEM decimation.",
)

parser.add_argument(
    "--show_bnd_spheres",
    action="store_true",
    help="Render bounding sphere approximations of each obstacle in real-time.",
    default=False,
)

args = parser.parse_args()

###########################################################

# Third Party
from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics, make_world
from projects_root.utils.usd_utils import load_usd_to_stage
simulation_app = init_app()
stage = load_usd_to_stage("usd_collection/envs/cv_new.usd")
my_world = make_world(ground_plane=True, set_default_prim=True, to_Xform=True)

import os

# Third Party
import carb
import numpy as np
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.geom.types import Mesh  # type: ignore
from curobo.geom.sphere_fit import SphereFitType

############################################################


########### OV #################;;;;;


###########
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
########### frame prim #################;;;;;


# Standard Library
from typing import Optional
from typing import Any

# Third Party
from projects_root.utils.helper import add_extensions, add_robot_to_scene
from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics, make_world

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
from projects_root.utils.handy_utils import get_rollouts_in_world_frame
# Project utilities
from projects_root.utils.world_model_wrapper import WorldModelWrapper
from projects_root.utils.usd_pose_helper import get_stage_poses, list_relevant_prims
from projects_root.utils.handy_utils import save_curobo_world
from projects_root.utils.usd_utils import load_prims_from_usd

############################################################

# forward declaration for static checkers
cu_world_wrapper: Any  # will be assigned in main()

def draw_points(rollouts: torch.Tensor):
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


# ------------------------------------------------------------------
# Helper: Render geometry approximations (bounding spheres) for obstacles
# ------------------------------------------------------------------


obs_spheres = []  # Will be populated at runtime; keep at module scope for reuse


def render_geom_approx_to_spheres(collision_world,n_spheres=50):
    """Visualize an approximate geometry (collection of spheres) for each obstacle.

    Notes:
        • Uses SAMPLE_SURFACE sphere fitting with a per–obstacle radius equal to
          1 % of its smallest OBB extent.
        • Relies on global variables `robot_base_frame` and `cu_world_wrapper` that
          are established in the main routine.
        • Maintains a persistent `obs_spheres` list so VisualSphere prims are
          created only once and then updated every frame.

    Args:
        collision_world (WorldConfig): Current CuRobo collision world instance.
    """

    global obs_spheres, robot_base_frame, cu_world_wrapper

    if collision_world is None or len(collision_world.objects) == 0:
        return

    # Use the utility in WorldModelWrapper to get sphere list (world-frame)
    all_sph = WorldModelWrapper.make_geom_approx_to_spheres(
        collision_world,
        robot_base_frame.tolist(),
        n_spheres=50,
        fit_type=SphereFitType.SAMPLE_SURFACE,
        radius_scale=0.05,  # 5 % of smallest OBB side for visibility
    )

    if not all_sph:
        return

    # Create extra VisualSphere prims if needed (handle import gracefully during static analysis)
    try:
        from omni.isaac.core.objects import sphere  # type: ignore
    except ImportError:  # Fallback if omniverse modules are unavailable in the analysis env
        return

    # Get shared material from first sphere (if any)
    shared_mat_path = None
    if obs_spheres:
        try:
            first_rel = obs_spheres[0].prim.GetRelationship("material:binding")
            targets = first_rel.GetTargets()
            if targets:
                shared_mat_path = targets[0]
        except Exception:
            pass

    stage = None  # Will capture Omni stage after first sphere is created

    while len(obs_spheres) < len(all_sph):
        p, r = all_sph[len(obs_spheres)]

        # Create sphere – this will auto-generate a new material prim
        sp = sphere.VisualSphere(
            prim_path=f"/curobo/obs_sphere_{len(obs_spheres)}",
            position=np.ravel(p),
            radius=r,
            color=np.array([1.0, 0.6, 0.1]),
        )

        # On creation update stage reference
        if stage is None:
            stage = sp.prim.GetStage()

        try:
            rel = sp.prim.GetRelationship("material:binding")
            orig_targets = rel.GetTargets()
            new_mat_path = orig_targets[0] if orig_targets else None

            # Rebind to shared material if one exists
            if shared_mat_path is not None:
                rel.SetTargets([shared_mat_path])

                # Remove the auto-generated material prim to avoid duplicates
                if new_mat_path and stage.GetPrimAtPath(new_mat_path):
                    stage.RemovePrim(new_mat_path)
            else:
                # First sphere becomes the reference material
                if new_mat_path:
                    shared_mat_path = new_mat_path
        except Exception:
            pass

        obs_spheres.append(sp)

    # Update current prims
    for idx, (p, r) in enumerate(all_sph):
        # Explicitly update both position and orientation (identity quaternion) – some
        # Isaac Sim versions ignore translation-only updates when orientation is
        # omitted.
        obs_spheres[idx].set_world_pose(position=np.ravel(p), orientation=np.array([1.0, 0.0, 0.0, 0.0]))
        obs_spheres[idx].set_radius(r)

    # Hide surplus prims, if any
    for idx in range(len(all_sph), len(obs_spheres)):
        obs_spheres[idx].set_world_pose(position=np.array([0, 0, -10]), orientation=np.array([1.0, 0.0, 0.0, 0.0]))


def main(robot_base_frame, target_prim_subpath, obs_root_prim_path, world_prim_path):
    """
    Args:
        robot_base_frame: [x, y, z, qw, qx, qy, qz] pose of the robot's base frame in world frame. Originally as [0,0,0,1,0,0,0] 
        target_prim_subpath: prim path of the target object *w.r. to world prim path* .
        obs_root_prim_path: 
            path to where you should put all your isaac sim obstacles udner.
            prim path of the root prim in which the obstacles are. All obstacles should be put under %obs_root_prim_path%/obstacle name Drag obstacles under this prim path.
        world_prim_path: prim path of the world prim. ()
    """
    global sim_app, my_world
    target_prim_path = world_prim_path + target_prim_subpath
    # Get robot base frame from command line arguments
    # robot_base_frame = np.array(args.robot_base_frame)
    print(f"Robot base frame set to: {robot_base_frame}")
    
    # assuming obstacles are in objects_path:
    # my_world = World(stage_units_in_meters=1.0)
    activate_gpu_dynamics(my_world)
    stage = my_world.stage

    #xform = stage.DefinePrim(world_prim_path, "Xform")
    # stage.SetDefaultPrim(xform)
    
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # my_world.scene.add_default_ground_plane()

    
   
    
    cv_xforms_to_load = [ "/World/ConveyorTrack" + x for x in ['', '_01', '_02', '_03', '_04', '_05', '_06', '_07']]
    cu_obs_to_load = ['/World/cv_approx' + x for x in [str(i) for i in range(1, 18)]]
    cv_cube = ['/World/conveyor_cube']
    created_paths = load_prims_from_usd(
        "usd_collection/envs/cv_new.usd",
        prim_paths=cv_xforms_to_load + cu_obs_to_load + cv_cube, # ConveyorTrack, Cube
        dest_root="/World",
        stage=my_world.stage,
        
    )

    
    print(f"Created paths: {created_paths}")

    paths_to_ignore_in_curobo_world_model = cv_xforms_to_load + cv_cube
    print(f"Paths to ignore in curobo world model: {cv_xforms_to_load}")
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        target_prim_path,
        position=np.array([0.75, 0, 0.5]),
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

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world, position=robot_base_frame[:3], orientation=robot_base_frame[3:])

    articulation_controller = robot.get_articulation_controller()

    # world_cfg_table = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # )
    # world_cfg_table.cuboid[0].pose[2] -= 0.04
    # world_cfg1 = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # ).get_mesh_world()
    # world_cfg1.mesh[0].name += "_mesh"
    # world_cfg1.mesh[0].pose[2] = -10.5

    # world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # Initialize WorldModelWrapper for efficient obstacle updates
    world_cfg = WorldConfig() # initial world config, empty (could also contain obs...)
    cu_world_wrapper = WorldModelWrapper(
        world_config=world_cfg,
        X_robot_W=robot_base_frame,
        verbosity=4
    )

    init_curobo = False

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    
    # world_cfg_table = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # )
    # world_cfg1 = WorldConfig.from_dict(
    #     load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    # ).get_mesh_world()
    # world_cfg1.mesh[0].pose[2] = -10.0

    # world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=False,  # disable to isolate CUDA graph issues
        use_cuda_graph_metrics=False,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        # Use the mesh collision checker (includes primitive support automatically)
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
    
    # Initialize world model wrapper once - this replaces the repeated world recreation
    world_initialized = False
    # Cache of prim paths already known to the collision world
    known_prim_paths: set[str] = set()
    
    init_world = False
    cmd_state_full = None
    step = 0
    add_extensions(simulation_app, args.headless_mode)
    spheres = None  # For robot collision sphere visualization
    
    
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()

    
    
    
    
    while simulation_app.is_running():
        
        # uncomment to save world model
        # try:
        #     os.removedirs("tmp/debug_world_models")
        # except:
        #     pass
        # os.makedirs("tmp/debug_world_models", exist_ok=True)
        # if step % 500 == 0 and step > 0:
        #     print(f"Saving world model at step {step}")
        #     save_curobo_world(f"tmp/debug_world_models/world_model_{step}.obj",cu_world_wrapper.get_collision_world())
                
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(get_rollouts_in_world_frame(mpc.get_visual_rollouts(), robot_base_frame))

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 10:
            # my_world.reset()
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
        
        # Initialize world model wrapper once (replaces the expensive recreation every 1000 steps)
        if not world_initialized and step_index > 20:
            print("Initializing WorldModelWrapper (one-time setup)...")
            # 1) pull raw USD obstacles
            _raw_world = usd_help.get_obstacles_from_stage(
                only_paths=[world_prim_path],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    target_prim_path,
                    "/curobo",
                    *paths_to_ignore_in_curobo_world_model,
                ],
            )

            # 2) Optional mesh decimation
            # simplify_mesh_obstacles(_raw_world, args.max_mesh_faces)

            # 3) Optionally simplify to cuboids for speed
            if args.use_obb_approx:
                _init_cu_world = WorldConfig.create_obb_world(_raw_world)
            else:
                _init_cu_world = _raw_world.get_collision_check_world()

            # # Validate meshes before pushing to CUDA – helps track CUDA 700 errors
            # try:
            #     _validate_mesh_list(_init_cu_world.mesh, "Pre-CUDA")
            # except ValueError as e:
            #     import traceback, sys

            #     print("\n[MeshValidationError]", e)
            #     traceback.print_exc()
            #     sys.exit(1)

            cu_col_world_R: WorldConfig = cu_world_wrapper.initialize_from_cu_world(
                cu_world_R=_init_cu_world,
            )
            
            # # Extra validation after CuRobo converts meshes (they may be re-ordered)
            # try:
            #     _validate_mesh_list(cu_col_world_R.mesh, "Post-CuRobo")
            # except ValueError as e:
            #     import traceback, sys

            #     print("\n[MeshValidationError]", e)
            #     traceback.print_exc()
            #     sys.exit(1)
            
            # Update MPC world collision checker with the initialized world
            mpc.world_coll_checker.load_collision_model(cu_col_world_R)
            # Set the collision checker reference in the wrapper
            cu_world_wrapper.set_collision_checker(mpc.world_coll_checker)
            # Record the prims that are currently considered obstacles
            ignore_list = [
                robot_prim_path,
                target_prim_path,
                "/World/defaultGroundPlane",
                "/curobo",
                *paths_to_ignore_in_curobo_world_model,
            ]

            known_prim_paths = set(cu_world_wrapper.get_obstacle_names())

            world_initialized = True
            print("WorldModelWrapper initialized successfully!")
        
        
        # ------------------------------------------------------------------
        # Detect *new* prims cheaply; load geometry only when necessary
        # ------------------------------------------------------------------
        if world_initialized: #cu_world_wrapper.is_initialized():
            
            
            # ------------------------------------------------------------------
            # Fast pose update (no heavy geometry traversal)
            # ------------------------------------------------------------------
            ignore_list = [
                robot_prim_path,
                target_prim_path,
                "/World/defaultGroundPlane",
                "/curobo",
                *paths_to_ignore_in_curobo_world_model,
            ]
            # print(f"ignore_list: {ignore_list}")
            pose_dict = get_stage_poses(
                usd_helper=usd_help,
                only_paths=[obs_root_prim_path],
                reference_prim_path=world_prim_path,
                ignore_substring=ignore_list,
            )
            # print(f"pose_dict: {pose_dict}")
            cu_world_wrapper.update_from_pose_dict(pose_dict)

            
            current_paths = set(
                list_relevant_prims(usd_help, [obs_root_prim_path], ignore_list)
            )

            new_paths = current_paths - known_prim_paths
            if new_paths:
                print(f"[NEW OBSTACLES] {new_paths}")
                

                _new_raw = usd_help.get_obstacles_from_stage(
                    only_paths=list(new_paths),
                    reference_prim_path=robot_prim_path,
                    ignore_substring=ignore_list,
                )

                # Optional decimation for newly added meshes
                # simplify_mesh_obstacles(_new_raw, args.max_mesh_faces)

                if args.use_obb_approx:
                    new_world_cfg = WorldConfig.create_obb_world(_new_raw)
                else:
                    new_world_cfg = _new_raw.get_collision_check_world()

                if new_world_cfg.objects:  # add only if we got actual obstacles
                    cu_world_wrapper.add_new_obstacles_from_cu_world(
                        cu_world_R=new_world_cfg,
                        silent=False,
                    )
                    # Track real obstacle names so future pose updates work
                    for obj in new_world_cfg.objects:
                        known_prim_paths.add(obj.name)
        
        # position and orientation of target virtual cube:
        cube_position, cube_orientation = target.get_world_pose() # p_goal_W, q_goal_W
        if past_pose is None:
            past_pose = cube_position + 1.0

        if np.linalg.norm(cube_position - past_pose) > 1e-3:
            p_goal_R, q_goal_R = FrameUtils.world_to_F(
                robot_base_frame[:3],    # base frame position in world
                robot_base_frame[3:],    # base frame orientation in world  
                cube_position,         # position in world
                cube_orientation       # orientation in world
            )

            ik_goal = Pose(
                position=tensor_args.to_device(p_goal_R),
                quaternion=tensor_args.to_device(q_goal_R),
            )
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            past_pose = cube_position

        # if not changed don't call curobo:

        # get robot current state:
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
        if cmd_state_full is None:
            current_state.copy_(cu_js)
        else:
            current_state_partial = cmd_state_full.get_ordered_joint_state(
                mpc.rollout_fn.joint_names
            )
            current_state.copy_(current_state_partial)
            current_state.joint_names = current_state_partial.joint_names
            # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        common_js_names = []
        current_state.copy_(cu_js)

        mpc_result = mpc.step(current_state, max_attempts=2)
        # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

        succ = True  # ik_result.success.item()
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
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        # positions_goal = articulation_action.joint_positions
        if step_index % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        if succ:
            # set desired joint angles obtained from IK:
            for _ in range(1):
                articulation_controller.apply_action(art_action)

        else:
            carb.log_warn("No action is being taken.")

        # ------------------------------------------------------------------
        # OPTIONAL: visualize obstacle bounding spheres (simplified)
        # ------------------------------------------------------------------
        if args.show_bnd_spheres and world_initialized and step_index % 20 == 0:
            render_geom_approx_to_spheres(cu_world_wrapper.get_collision_world())


############################################################

# --------------------------------------------------------------
# OPTIONAL MESH SIMPLIFICATION
# --------------------------------------------------------------


def simplify_mesh_obstacles(world: WorldConfig, face_limit: int):
    """Simplify mesh obstacles using MeshLib SDK.
    
    Based on official MeshLib documentation: https://meshlib.io/feature/mesh-simplification/

    Args:
        world: WorldConfig potentially containing Mesh obstacles.
        face_limit: Upper bound on triangle count; meshes below this are left unmodified.
    """

    if face_limit <= 0:
        print("[MeshSimplify] Face limit <= 0, skipping simplification")
        return

    mesh_list = world.mesh if world.mesh is not None else []
    if not mesh_list:
        return

    print(f"[MeshSimplify] Processing {len(mesh_list)} meshes with face limit {face_limit}")

    # Try to import MeshLib
    try:
        import meshlib.mrmeshpy as mrmeshpy
        print(f"[MeshSimplify] Using MeshLib SDK")
    except ImportError:
        print("[MeshSimplify] MeshLib not available, skipping simplification")
        print("[MeshSimplify] Install: pip install meshlib")
        return

    # Process each mesh
    for m in mesh_list:
        if not isinstance(m, Mesh):
            continue
            
        verts, faces = m.get_mesh_data()
        if verts is None or faces is None:
            print(f"[MeshSimplify] {m.name}: No mesh data, skipping")
            continue
            
        face_cnt = len(faces)
        if face_cnt <= face_limit:
            print(f"[MeshSimplify] {m.name}: {face_cnt} faces (already below limit)")
            continue

        print(f"[MeshSimplify] {m.name}: {face_cnt} faces → attempting simplification")
        
        try:
            simplified_verts, simplified_faces = _simplify_with_meshlib(verts, faces, face_limit)
            
            if simplified_verts is not None and simplified_faces is not None:
                # Update the mesh
                m.vertices = simplified_verts.astype(np.float32).tolist()
                m.faces = simplified_faces.astype(np.int32).tolist()
                print(f"[MeshSimplify] {m.name}: {face_cnt} → {len(simplified_faces)} faces ✓")
                
                if len(simplified_faces) > face_limit:
                    print(f"[MeshSimplify] {m.name}: WARNING: Still above limit ({face_limit}), but best achievable")
            else:
                print(f"[MeshSimplify] {m.name}: Simplification failed, keeping original")
                
        except Exception as e:
            print(f"[MeshSimplify] {m.name}: Error during simplification: {e}")
            print(f"[MeshSimplify] {m.name}: Keeping original mesh")
        finally:
            print("finally debug") # remove this when done debugging
            exit()

def _simplify_with_meshlib(vertices, faces, face_limit):
    """Simplify mesh using MeshLib SDK based on official documentation."""
    import meshlib.mrmeshpy as mrmeshpy
    import numpy as np
    
    try:
        # Create mesh from vertices and faces
        # Convert to MeshLib format
        verts = []
        for v in vertices:
            verts.append(mrmeshpy.Vector3f(float(v[0]), float(v[1]), float(v[2])))
        
        faces_list = []
        for f in faces:
            # Create Triangle3i from the face indices
            face_array = mrmeshpy.Triangle3i([int(f[0]), int(f[1]), int(f[2])])
            faces_list.append(face_array)
        
        # Create mesh object
        mesh = mrmeshpy.Mesh()
        mesh.topology = mrmeshpy.MeshTopology(faces_list)
        mesh.points = verts
        
        # Validate mesh
        if not mesh.topology.isValid():
            print("[MeshLib] Invalid mesh topology")
            return None, None
        
        # Repack mesh optimally (recommended for performance)
        mesh.packOptimally()
        
        # Setup decimate parameters based on official documentation
        settings = mrmeshpy.DecimateSettings()
        
        # Set target face count
        original_face_count = len(faces)
        target_faces = min(face_limit, max(100, original_face_count // 10))
        settings.maxDeletedFaces = original_face_count - target_faces
        
        # Set maximum error (allow some error to reach target)
        settings.maxError = 0.1
        
        # Number of parts for parallel processing (improves performance)
        settings.subdivideParts = 64
        
        # Simplify mesh using official API
        result = mrmeshpy.decimateMesh(mesh, settings)
        
        if result and mesh.topology.isValid():
            final_face_count = mesh.topology.numValidFaces()
            
            if final_face_count < original_face_count and final_face_count > 0:
                # Extract simplified vertices and faces
                simplified_verts = []
                for i in range(len(mesh.points)):
                    p = mesh.points[i]
                    simplified_verts.append([p.x, p.y, p.z])
                
                simplified_faces = []
                for i in range(mesh.topology.faceSize()):
                    if mesh.topology.hasFace(mrmeshpy.FaceId(i)):
                        face_verts = mesh.topology.getTriVerts(mrmeshpy.FaceId(i))
                        simplified_faces.append([int(face_verts.a), int(face_verts.b), int(face_verts.c)])
                
                print(f"[MeshLib] Decimation: {original_face_count} → {final_face_count} faces")
                return np.array(simplified_verts), np.array(simplified_faces)
        
        return None, None
        
    except Exception as e:
        print(f"[MeshLib] Error: {e}")
        return None, None


def _simplify_with_pymeshlab(vertices, faces, face_limit):
    """Simplify mesh using PyMeshLab (most reliable)."""
    import pymeshlab
    import numpy as np
    
    # Create MeshSet and add mesh
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertices, faces)
    ms.add_mesh(mesh)
    
    # Use quadric edge collapse decimation - very reliable
    target_faces = min(face_limit, max(100, len(faces) // 10))  # Conservative target
    
    try:
        # Method 1: Try exact face count
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_faces)
        simplified = ms.current_mesh()
        
        if simplified.face_number() > 0 and simplified.face_number() < len(faces):
            print(f"[PyMeshLab] Exact decimation: {len(faces)} → {simplified.face_number()} faces")
            return simplified.vertex_matrix(), simplified.face_matrix()
            
    except Exception as e:
        print(f"[PyMeshLab] Exact decimation failed: {e}")
    
    try:
        # Method 2: Try percentage-based decimation
        percentage = min(0.9, face_limit / len(faces))
        ms.meshing_decimation_quadric_edge_collapse(targetperc=percentage)
        simplified = ms.current_mesh()
        
        if simplified.face_number() > 0 and simplified.face_number() < len(faces):
            print(f"[PyMeshLab] Percentage decimation: {len(faces)} → {simplified.face_number()} faces")
            return simplified.vertex_matrix(), simplified.face_matrix()
            
    except Exception as e:
        print(f"[PyMeshLab] Percentage decimation failed: {e}")
    
    try:
        # Method 3: Try aggressive decimation with higher target
        aggressive_target = min(face_limit * 2, len(faces) // 5)  # More aggressive
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=aggressive_target)
        simplified = ms.current_mesh()
        
        if simplified.face_number() > 0 and simplified.face_number() < len(faces):
            print(f"[PyMeshLab] Aggressive decimation: {len(faces)} → {simplified.face_number()} faces")
            return simplified.vertex_matrix(), simplified.face_matrix()
            
    except Exception as e:
        print(f"[PyMeshLab] Aggressive decimation failed: {e}")
    
    return None, None


def _simplify_with_open3d(vertices, faces, face_limit):
    """Simplify mesh using Open3D."""
    import open3d as o3d
    import numpy as np
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Use quadric decimation
    target_faces = min(face_limit, max(100, len(faces) // 10))
    
    try:
        simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_faces)
        return np.asarray(simplified.vertices), np.asarray(simplified.triangles)
    except Exception:
        # Fallback to vertex clustering
        try:
            voxel_size = mesh.get_axis_aligned_bounding_box().get_extent().max() / 50
            simplified = mesh.simplify_vertex_clustering(voxel_size)
            if len(simplified.triangles) < len(faces):
                return np.asarray(simplified.vertices), np.asarray(simplified.triangles)
        except Exception:
            pass
        return None, None


def _simplify_with_trimesh(vertices, faces, face_limit):
    """Simplify mesh using trimesh (fallback)."""
    import trimesh
    import numpy as np
    
    # Create trimesh object
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if len(tri_mesh.vertices) == 0 or len(tri_mesh.faces) == 0:
        return None, None
    
    # Try quadric decimation first
    if hasattr(tri_mesh, 'simplify_quadric_decimation'):
        target_faces = min(face_limit, max(100, len(faces) // 10))
        
        for aggression in [1, 5, 10]:  # Try different aggression levels
            try:
                simplified = tri_mesh.simplify_quadric_decimation(
                    face_count=target_faces, 
                    aggression=aggression
                )
                
                if (len(simplified.vertices) > 0 and 
                    len(simplified.faces) > 0 and
                    len(simplified.faces) < len(faces)):
                    return simplified.vertices, simplified.faces
                    
            except Exception:
                continue
    
    # Fallback to vertex clustering
    for radius in [0.01, 0.05, 0.1]:
        try:
            if hasattr(tri_mesh, 'simplify_vertex_clustering'):
                simplified = tri_mesh.simplify_vertex_clustering(radius=radius)
                if (len(simplified.vertices) > 0 and 
                    len(simplified.faces) > 0 and
                    len(simplified.faces) < len(faces)):
                    return simplified.vertices, simplified.faces
        except Exception:
            continue
    
    return None, None
def _validate_mesh_list(mesh_list, name_hint=""):
    """Sanity-check every CuRobo Mesh in *mesh_list*.

    Raises:
        ValueError if any mesh contains
            • wrong vertex/face shape
            • face index outside vertex range
            • NaN / Inf coordinates
            • degenerate triangles
    """

    import numpy as _np

    for mesh in mesh_list or []:
        v = _np.asarray(mesh.vertices)
        f = _np.asarray(mesh.faces)

        if v is None or f is None:
            raise ValueError(f"[{name_hint}] {mesh.name} – missing vertices or faces")
        if v.ndim != 2 or v.shape[1] != 3:
            raise ValueError(f"[{name_hint}] {mesh.name} – vertices have shape {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(f"[{name_hint}] {mesh.name} – faces have shape {f.shape}")
        if f.max() >= len(v):
            raise ValueError(
                f"[{name_hint}] {mesh.name} – face index out of range (max {f.max()} ≥ {len(v)})"
            )
        if _np.any(_np.isnan(v)) or _np.any(_np.isinf(v)):
            raise ValueError(f"[{name_hint}] {mesh.name} – vertices contain NaN/Inf")
        # Degenerate triangles: any two indices equal
        bad = _np.where((f[:, 0] == f[:, 1]) | (f[:, 0] == f[:, 2]) | (f[:, 1] == f[:, 2]))[0]
        if bad.size:
            raise ValueError(f"[{name_hint}] {mesh.name} – {bad.size} degenerate triangles")


if __name__ == "__main__":
    robot_base_frame = np.array([-0.75, -0.33, 0.0, 1.0, 0.0, 0.0, 0.0])
    world_prim_path = "/World"
    main(robot_base_frame, target_prim_subpath="/target",obs_root_prim_path=world_prim_path, world_prim_path=world_prim_path)
    simulation_app.close() 
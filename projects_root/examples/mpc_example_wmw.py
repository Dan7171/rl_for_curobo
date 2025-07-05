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

# Add current directory to Python path for local imports
import sys
import os

# dynamic imports in container- temp solution instead of 'omni_python - pip install .' (in rl_for_curobo dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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
    default=300, # 0 means no decimation, 50 considered small
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
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

# Third Party
# Enable the layers and stage windows in the UI
# Standard Library
import os
# import inspect, projects_root.utils.world_model_wrapper as wmw
# print("[IMPORT-PATH]", inspect.getfile(wmw.WorldModelWrapper))
# exit()
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
from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics

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


def render_geom_approx_to_spheres(collision_world):
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
    
    target_prim_path = world_prim_path + target_prim_subpath
    # Get robot base frame from command line arguments
    # robot_base_frame = np.array(args.robot_base_frame)
    print(f"Robot base frame set to: {robot_base_frame}")
    
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    activate_gpu_dynamics(my_world)
    stage = my_world.stage

    xform = stage.DefinePrim(world_prim_path, "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    target = cuboid.VisualCuboid(
        target_prim_path,
        position=np.array([1.5, 1, 0.5]),
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
        use_cuda_graph=True,
        use_cuda_graph_metrics=True,
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
                ],
            )

            # 2) Optional mesh decimation
            # simplify_mesh_obstacles(_raw_world, args.max_mesh_faces)

            # 3) Optionally simplify to cuboids for speed
            if args.use_obb_approx:
                _init_cu_world = WorldConfig.create_obb_world(_raw_world)
            else:
                _init_cu_world = _raw_world.get_collision_check_world()

            cu_col_world_R: WorldConfig = cu_world_wrapper.initialize_from_cu_world(
                cu_world_R=_init_cu_world,
            )
            
            # Update MPC world collision checker with the initialized world
            mpc.world_coll_checker.load_collision_model(cu_col_world_R)
            # Set the collision checker reference in the wrapper
            print("setting collision checker!!!!!!!!")
            print("debug 1")
            time.sleep(10)
            cu_world_wrapper.set_collision_checker(mpc.world_coll_checker)
            print("debug 2")
            time.sleep(10)
            # Record the prims that are currently considered obstacles
            ignore_list = [
                robot_prim_path,
                target_prim_path,
                "/World/defaultGroundPlane",
                "/curobo",
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
            ]

            pose_dict = get_stage_poses(
                usd_helper=usd_help,
                only_paths=[obs_root_prim_path],
                reference_prim_path=world_prim_path,
                ignore_substring=ignore_list,
            )
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
        if args.show_bnd_spheres and world_initialized and step_index % 5 == 0:
            render_geom_approx_to_spheres(cu_world_wrapper.get_collision_world())


############################################################

# --------------------------------------------------------------
# OPTIONAL MESH SIMPLIFICATION
# --------------------------------------------------------------


def simplify_mesh_obstacles(world: WorldConfig, face_limit: int):
    """Simplify mesh obstacles in-place to reduce GPU load.

    Args:
        world: WorldConfig potentially containing Mesh obstacles.
        face_limit: Upper bound on triangle count; meshes below this are left unmodified.
    """

    if face_limit <= 0:
        return

    mesh_list = world.mesh if world.mesh is not None else []
    for m in list(mesh_list):  # copy of list to allow mutation
        if isinstance(m, Mesh):
            verts, faces = m.get_mesh_data()
            face_cnt = len(faces) if faces is not None else 0
            if faces is not None and face_cnt > face_limit:
                try:
                    import trimesh  # local import
                    tri = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
                    decimate = getattr(tri, "simplify_quadratic_decimation", None)
                    if callable(decimate):
                        tri = decimate(face_limit)
                        m.vertices = tri.vertices
                        m.faces = tri.faces
                        print(f"[MeshDecimate] {m.name}: {face_cnt} → {len(tri.faces)} faces")
                    else:
                        print(f"[MeshDecimate] trimesh lacks decimator – converting {m.name} to OBB")
                        # Replace heavy mesh with its oriented bounding box cuboid
                        obb = m.get_cuboid()
                        if world.cuboid is None:
                            world.cuboid = []
                        world.cuboid.append(obb)
                        world.mesh.remove(m)
                        continue
                except ImportError:
                    print("[MeshDecimate] trimesh not installed – skipping mesh simplification")
                except Exception as e:
                    print(f"[MeshDecimate] Failed to decimate {m.name}: {e}")


if __name__ == "__main__":
    robot_base_frame = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    world_prim_path = "/World"
    main(robot_base_frame, target_prim_subpath="/target",obs_root_prim_path=world_prim_path, world_prim_path=world_prim_path)
    simulation_app.close() 
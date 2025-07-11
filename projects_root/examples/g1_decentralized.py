#
# G1 Dual Arm Decentralized MPC Controller
#
# This script controls both arms of the G1 robot using a decentralized MPC approach.
# It is based on the mpc_async_multirobot_wmv.py example.

# Force non-interactive matplotlib backend to avoid GUI operations from worker threads
import os
from typing import Dict, List, Optional, Tuple, Any
os.environ.setdefault("MPLBACKEND", "Agg")

# Simulation settings
SIMULATING = True 
REAL = False
REAL_TIME_EXPECTED_CTRL_DT = 0.03
ENABLE_GPU_DYNAMICS = True
OBS_PREDICTION = True
DEBUG = True
VISUALIZE_PLANS_AS_DOTS = True
VISUALIZE_MPC_ROLLOUTS = True
VISUALIZE_ROBOT_COL_SPHERES = False
HIGHLIGHT_OBS = False
HIGHLIGHT_OBS_H = 30
DEBUG_GPU_MEM = False
RENDER_DT = 0.03
PHYSICS_STEP_DT = 0.03
MPC_DT = 0.03
HEADLESS_ISAAC = False 

# Import modules needed for simulation
if SIMULATING:
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument( 
        "--show_spheres",
        action="store_true",
        help="Render collision spheres of the robot.",
        default=False,
    )
    args = parser.parse_args()
    
    # Import Isaac Sim
    try:
        import isaacsim
    except ImportError:
        pass
        
    from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics, make_world
    from projects_root.utils.usd_utils import load_usd_to_stage

    # Init Isaac Sim app
    simulation_app = init_app()
    from omni.isaac.core import World
    
    # Init Isaac Sim world
    my_world = make_world(ground_plane=True, set_default_prim=True, to_Xform=True)
    
    # Enable GPU dynamics if needed
    if ENABLE_GPU_DYNAMICS:
        activate_gpu_dynamics(my_world)
    
    # Set simulation dt
    my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT)
    
    from projects_root.utils.helper import add_extensions
    add_extensions(simulation_app, None if not HEADLESS_ISAAC else 'true')

# Standard library modules
import time
from threading import Thread, Event, Lock
import torch
import numpy as np

# CuRobo modules
from curobo.geom.types import Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.util.logger import setup_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import load_yaml

# Our modules
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import init_runtime_topics, get_topics
from projects_root.utils.transforms import transform_poses_batched
from projects_root.autonomous_arm import ArmMpc
from projects_root.utils.draw import draw_points
from projects_root.utils.colors import npColors

# Prevent cuda out of memory errors
a = torch.zeros(4, device="cuda:0")

def configure_g1_left_arm():
    """
    Configure the left arm of the G1 robot.
    
    Returns:
        Dict: Configuration dictionary for the left arm
    """
    # Load the base configuration for the G1 dual arm
    base_config_path = "curobo/src/curobo/content/configs/robot/g1_23dofs_dual_arm.yml"
    base_cfg = load_yaml(base_config_path)["robot_cfg"]
    
    # Create left arm configuration
    left_cfg = {
        "kinematics": {
            "urdf_path": base_cfg["kinematics"]["urdf_path"],
            "asset_root_path": base_cfg["kinematics"]["asset_root_path"],
            "base_link": base_cfg["kinematics"]["base_link"],
            "ee_link": base_cfg["kinematics"]["ee_link"][0],  # Left arm end effector
            
            # Extract only left arm collision spheres
            "collision_spheres": {
                k: v for k, v in base_cfg["kinematics"]["collision_spheres"].items()
                if k.startswith("left_") or k in ["torso_link", "pelvis"]
            },
            "collision_sphere_buffer": base_cfg["kinematics"]["collision_sphere_buffer"],
            
            # Self collision for left arm only
            "self_collision_ignore": {
                k: v for k, v in base_cfg["kinematics"].get("self_collision_ignore", {}).items()
                if k.startswith("left_")
            },
            "self_collision_buffer": {
                k: v for k, v in base_cfg["kinematics"].get("self_collision_buffer", {}).items()
                if k.startswith("left_") or k == "torso_link"
            },
            
            # Configure left arm joints
            "cspace": {
                "joint_names": base_cfg["kinematics"]["cspace"]["joint_names"][:5],  # First 5 joints for left arm
                "retract_config": base_cfg["kinematics"]["cspace"]["retract_config"][:5],
                "null_space_weight": base_cfg["kinematics"]["cspace"]["null_space_weight"][:5],
                "cspace_distance_weight": base_cfg["kinematics"]["cspace"]["cspace_distance_weight"][:5],
                "max_jerk": base_cfg["kinematics"]["cspace"]["max_jerk"],
                "max_acceleration": base_cfg["kinematics"]["cspace"]["max_acceleration"]
            }
        }
    }
    
    return {"robot_cfg": left_cfg}

def configure_g1_right_arm():
    """
    Configure the right arm of the G1 robot.
    
    Returns:
        Dict: Configuration dictionary for the right arm
    """
    # Load the base configuration for the G1 dual arm
    base_config_path = "curobo/src/curobo/content/configs/robot/g1_23dofs_dual_arm.yml"
    base_cfg = load_yaml(base_config_path)["robot_cfg"]
    
    # Create right arm configuration
    right_cfg = {
        "kinematics": {
            "urdf_path": base_cfg["kinematics"]["urdf_path"],
            "asset_root_path": base_cfg["kinematics"]["asset_root_path"],
            "base_link": base_cfg["kinematics"]["base_link"],
            "ee_link": base_cfg["kinematics"]["ee_link"][1],  # Right arm end effector
            
            # Extract only right arm collision spheres
            "collision_spheres": {
                k: v for k, v in base_cfg["kinematics"]["collision_spheres"].items()
                if k.startswith("right_") or k in ["torso_link", "pelvis"]
            },
            "collision_sphere_buffer": base_cfg["kinematics"]["collision_sphere_buffer"],
            
            # Self collision for right arm only
            "self_collision_ignore": {
                k: v for k, v in base_cfg["kinematics"].get("self_collision_ignore", {}).items()
                if k.startswith("right_")
            },
            "self_collision_buffer": {
                k: v for k, v in base_cfg["kinematics"].get("self_collision_buffer", {}).items()
                if k.startswith("right_") or k == "torso_link"
            },
            
            # Configure right arm joints
            "cspace": {
                "joint_names": base_cfg["kinematics"]["cspace"]["joint_names"][5:10],  # Last 5 joints for right arm
                "retract_config": base_cfg["kinematics"]["cspace"]["retract_config"][5:10],
                "null_space_weight": base_cfg["kinematics"]["cspace"]["null_space_weight"][5:10],
                "cspace_distance_weight": base_cfg["kinematics"]["cspace"]["cspace_distance_weight"][5:10],
                "max_jerk": base_cfg["kinematics"]["cspace"]["max_jerk"],
                "max_acceleration": base_cfg["kinematics"]["cspace"]["max_acceleration"]
            }
        }
    }
    
    return {"robot_cfg": right_cfg}

def calculate_robot_sphere_count(robot_cfg):
    """
    Calculate the number of collision spheres for a robot from its configuration.
    """
    # Get collision spheres configuration
    collision_spheres = robot_cfg["kinematics"]["collision_spheres"]
    
    # Count spheres by counting entries in each link's sphere list
    sphere_count = 0
    for link_name, spheres in collision_spheres.items():
        if isinstance(spheres, list):
            sphere_count += len(spheres)
    
    # Add extra collision spheres
    extra_spheres = robot_cfg["kinematics"].get("extra_collision_spheres", {})
    extra_sphere_count = 0
    for obj_name, count in extra_spheres.items():
        extra_sphere_count += count
        
    return sphere_count, extra_sphere_count

def init_ccheck_wcfg_in_sim(usd_help, robot_prim_path, target_prim_path, ignore_substrings):    
    """
    Make the initial collision check world configuration.
    This is the world configuration that will be used for collision checking.
    """
    # Get obstacles from simulation and convert to WorldConfig
    cu_world_R = usd_help.get_obstacles_from_stage( 
        only_paths=['/World'], # look for obs only under the world prim path
        reference_prim_path=robot_prim_path, # obstacles are expressed in robot frame! (not world frame)
        ignore_substring=ignore_substrings
    )

    # Convert raw WorldConfig to collision check world WorldConfig!
    cu_world_R = cu_world_R.get_collision_check_world()
    
    return cu_world_R

def ctrl_loop_robot(robot_idx: int,
                   stop_event: Event,
                   get_t_idx,
                   t_lock: Lock,
                   physx_lock: Lock,
                   plans_lock: Lock,
                   robots,
                   col_pred_with,
                   plans: List[Optional[Any]],
                   tensor_args,
                   cu_world_never_add: List[str],
                   cu_world_never_update: List[str],
                   usd_help: Optional[UsdHelper] = None
                   ):
    """
    Background loop that runs one MPC cycle whenever t_idx increments.
    """
    last_step = -1
    r: ArmMpc = robots[robot_idx]
    
    # Initialize collision check world configuration
    if SIMULATING:
        assert usd_help is not None
        r.reset_wmw(init_ccheck_wcfg_in_sim(usd_help, r.prim_path, r.target_prim_path, cu_world_never_add)) 
    if REAL:
        # Not implemented
        pass

    # Control loop
    while not stop_event.is_set():
        # Wait for new time step
        with t_lock:
            cur_step = get_t_idx()
        if cur_step == last_step:
            time.sleep(1e-7)
            continue
        last_step = cur_step

        # Publish data for other robots
        if OBS_PREDICTION:
            sim_js = None
            real_js = None
            if SIMULATING:
                sim_js = r.get_sim_joint_state(sync_new=False)
            r.publish(plans, plans_lock, sim_js, real_js)
            
        # Sense
        if SIMULATING:
            update_obs_callback = (r.update_obs_from_sim, {'usd_help': usd_help, 'ignore_list': cu_world_never_add + cu_world_never_update})
            update_target_callback = (r.update_target_from_sim, {})
            update_joint_state_callback = (r.update_cu_js_from_sim, {})
            
            if r.use_col_pred:
                r.sense(update_obs_callback, update_target_callback, update_joint_state_callback, physx_lock, 
                       plans, col_pred_with[robot_idx], cur_step, tensor_args, robot_idx, plans_lock)
            else:
                r.sense(update_obs_callback, update_target_callback, update_joint_state_callback, physx_lock)

        # Plan (our modified mpc planning, (torch-heavy, no PhysX))
        action = r.plan(max_attempts=2)
        
        # Command 
        r.command(action, num_times=1, physx_lock=physx_lock)

def is_dyn_obs_cost_in_cfg(mpc_cfg) -> bool:
    """
    Check if dynamic obstacle cost is enabled in the MPC config.
    """
    return "cost" in mpc_cfg and "custom" in mpc_cfg["cost"] and "arm_base" in mpc_cfg["cost"]["custom"] and "dynamic_obs_cost" in mpc_cfg["cost"]["custom"]["arm_base"]

def publish_robot_context(robot_idx, robot_context, robot_pose, n_obstacle_spheres, robot_sphere_count, 
                         mpc_cfg, col_pred_with_robot, mpc_config_paths, robot_config_paths, robot_sphere_counts_split):
    """
    Publish robot context ("topics") to the environment topics.
    """
    # Populate robot context
    robot_context["env_id"] = 0
    robot_context["robot_id"] = robot_idx
    robot_context["robot_pose"] = robot_pose
    robot_context["n_obstacle_spheres"] = n_obstacle_spheres
    robot_context["n_own_spheres"] = robot_sphere_count
    robot_context["horizon"] = mpc_cfg["model"]["horizon"]
    robot_context["n_rollouts"] = mpc_cfg["mppi"]["num_particles"]
    robot_context["col_pred_with"] = col_pred_with_robot
    
    # Add fields for sparse sphere functionality
    robot_context["mpc_config_paths"] = mpc_config_paths
    robot_context["robot_config_paths"] = robot_config_paths
    robot_context["robot_sphere_counts"] = robot_sphere_counts_split

def create_arm_targets(my_world):
    """
    Create target cubes for the left and right arms
    """
    # Make targets for both arms
    left_target = my_world.scene.add_visual_cube(
        name="left_target",
        position=np.array([0.2, 0.3, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05
    )
    
    right_target = my_world.scene.add_visual_cube(
        name="right_target",
        position=np.array([0.8, 0.3, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([0, 1.0, 0]),
        size=0.05
    )
    
    return [left_target, right_target]

def main():
    """
    Main function for G1 dual-arm decentralized control
    """
    # CuRobo setup
    setup_logger("warn")
    tensor_args = TensorDeviceType()
    
    if SIMULATING:
        # Initialize UsdHelper
        usd_help = UsdHelper()
        stage = my_world.stage
        usd_help.load_stage(stage)
        
        # Set /World as Xform prim and make it the default prim
        stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
        
        # Create /curobo as Xform prim
        _curobo_xform = stage.DefinePrim("/curobo", "Xform")

    # Configure left and right arms
    left_arm_cfg = configure_g1_left_arm()
    right_arm_cfg = configure_g1_right_arm()
    
    # MPC config paths
    mpc_config_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_g1_dual_arm.yml"
    
    # Robot configurations
    robot_cfgs = [left_arm_cfg["robot_cfg"], right_arm_cfg["robot_cfg"]]
    mpc_cfgs = [load_yaml(mpc_config_path)] * 2  # Same MPC config for both arms
    
    # Setup for running both arms
    n_robots = 2  # Left and right arms
    
    # Define base poses for both arms (centered at torso_link)
    X_robots = [
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),  # Left arm base pose
        np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])   # Right arm base pose
    ]
    
    # Define target poses in robot frame
    X_targets_R = [
        [0.3, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0],  # Left arm target
        [0.3, -0.3, 0.5, 1.0, 0.0, 0.0, 0.0]   # Right arm target
    ]
    
    # Collision prediction configuration
    col_pred_with = [[1], [0]]  # Left arm predicts right arm, right arm predicts left arm
    plot_costs = [False, False]
    target_colors = [npColors.red, npColors.green]
    
    # Initialize runtime topics
    init_runtime_topics(n_envs=1, robots_per_env=n_robots)
    runtime_topics = get_topics()
    env_topics = runtime_topics.get_default_env() if runtime_topics is not None else []
    
    # Initialize robot joint indices lists
    robot_idx_lists = [None for _ in range(n_robots)]
    
    # Calculate sphere counts for robots
    robot_sphere_counts_split = [calculate_robot_sphere_count(robot_cfg) for robot_cfg in robot_cfgs]
    robot_sphere_counts = [split[0] + split[1] for split in robot_sphere_counts_split]
    robot_sphere_counts_no_extra = [split[0] for split in robot_sphere_counts_split]
    

    
    # Create ArmMpc instances for each arm
    robot_names = ["g1_left_arm", "g1_right_arm"]
    robot_config_paths = ["left_arm_cfg.yml", "right_arm_cfg.yml"]  # Just for reference
    
    robots = []
    for i in range(n_robots):
        robots.append(ArmMpc(
            robot_cfgs[i], 
            my_world,
            usd_help,
            env_id=0,
            robot_id=i,
            p_R=X_robots[i][:3],
            q_R=X_robots[i][3:], 
            p_T_R=np.array(X_targets_R[i][:3]),
            q_T_R=np.array(X_targets_R[i][3:]), 
            target_color=target_colors[i],
            plot_costs=plot_costs[i],
            override_particle_file=mpc_config_path,
            n_coll_spheres=robot_sphere_counts[i],
            n_coll_spheres_valid=robot_sphere_counts_no_extra[i],
            use_col_pred=OBS_PREDICTION and len(col_pred_with[i]) > 0,
        ))
    
    if SIMULATING:
        # Reset default prim to /World
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
        wait_for_playing(my_world, simulation_app, autoplay=True)
    
    # Publish contexts for communication between robots
    for i in range(n_robots):
        if is_dyn_obs_cost_in_cfg(mpc_cfgs[i]):
            n_obstacle_spheres = sum(robot_sphere_counts[j] for j in col_pred_with[i])
            publish_robot_context(i, env_topics[i], X_robots[i].tolist(), n_obstacle_spheres, 
                                robot_sphere_counts[i], mpc_cfgs[i], col_pred_with[i], 
                                [mpc_config_path] * n_robots, robot_config_paths, robot_sphere_counts_split)
    
    # Initialize robot solvers
    for i, robot in enumerate(robots):
        # Get robot joint indices
        _robot_idx_list = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot_idx_lists[i] = _robot_idx_list
        
        if SIMULATING:
            robot.init_joints_in_sim(_robot_idx_list)
        
        # Initialize MPC solver
        robots[i].init_solver(MPC_DT, DEBUG)
        if SIMULATING:
            robots[i].robot._articulation_view.initialize()
    
    # Start robot control threads
    t_idx = 0
    stop_event = Event()
    t_lock = Lock()
    plans_lock = Lock()
    physx_lock = Lock()
    plans = [None for _ in range(len(robots))]
    
    # Define objects to ignore during collision detection
    cu_world_never_add = [
        "/curobo", 
        *[robot.target_prim_path for robot in robots],
        *[robot.prim_path for robot in robots], 
        "/World/defaultGroundPlane"
    ]
    cu_world_never_update = []
    
    # Create and start control threads for each robot
    robot_threads = [
        Thread(
            target=ctrl_loop_robot,
            args=(idx, stop_event, lambda: t_idx, t_lock, physx_lock, plans_lock, 
                 robots, col_pred_with, plans, tensor_args, 
                 cu_world_never_add, cu_world_never_update, usd_help if SIMULATING else None),
            daemon=True
        ) 
        for idx in range(len(robots))
    ]
    
    for th in robot_threads:
        th.start()
    
    # Simulation loop
    step_batch_start_time = time.time()
    step_batch_size = 100
    
    if SIMULATING:
        while simulation_app.is_running():
            # Step physics with lock to prevent conflicts with robot threads
            with physx_lock:
                step_start_time = time.time()
                my_world.step(render=True)
                step_end_time = time.time()
                print(f"main-world-step-time, {step_end_time - step_start_time}")
                
            # Update time step counter with lock
            with t_lock:
                t_idx += 1
            
            # Print statistics periodically
            if t_idx % step_batch_size == 0:
                step_batch_time = time.time() - step_batch_start_time
                print(f"ts: {t_idx}")
                print("num of actions planned by each robot:")
                print([robots[i].n_actions_planned for i in range(len(robots))])
                print(f"overall avg step time: {(step_batch_time/step_batch_size)*1000:.1f} ms")
                step_batch_start_time = time.time()
                
        simulation_app.close()
    else:
        # Real world loop (not implemented)
        while True:
            with t_lock:
                t_idx += 1
            time.sleep(REAL_TIME_EXPECTED_CTRL_DT)
    
    # Clean up thread pool
    stop_event.set()
    for th in robot_threads:
        th.join()

if __name__ == "__main__":
    if DEBUG_GPU_MEM:
        torch.cuda.memory._record_memory_history()
    main() 
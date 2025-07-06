"""
async version of projects_root/examples/mpc_moving_obstacles_mpc_mpc.py
"""

# Force non-interactive matplotlib backend to avoid GUI operations from worker threads
import os
from typing import Dict
os.environ.setdefault("MPLBACKEND", "Agg") # ?

import sys
# TODO: dynamic imports in container- temp solution instead of 'omni_python - pip install .' (in rl_for_curobo dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SIMULATING = True # if False, then we are running the robot in real time (i.e. the robot will move as fast as the real time allows)
REAL_TIME_EXPECTED_CTRL_DT = 0.03 #1 / (The expected control frequency in Hz). Set that to the avg time measurded between two consecutive calls to my_world.step() in real time. To print that time, use: print(f"Time between two consecutive calls to my_world.step() in real time, run with --print_ctrl_rate "True")
ENABLE_GPU_DYNAMICS = False # # GPU DYNAMICS - OPTIONAL (originally was disabled)# GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU. However, this will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. If enabling GPU dynamics slows down the simulation, it may be that your GPU is not able to handle the additional load. You can enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.# See: https://docs-prod.omniverse.nvidia.com/isaacsim/latest/reference_material/speedup_cheat_sheet.html?utm_source=chatgpt.com # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
OBS_PREDICTION = True # If True, this would be what the original MPC cost function could handle. False means that the cost will consider obstacles as moving and look into the future, while True means that the cost will consider obstacles as static and not look into the future.
DEBUG = True # Currenly, the main feature of True is to run withoug cuda graphs. When its true, we can set breakpoints inside cuda graph code (like in cost computation in "ArmBase" for example)  
VISUALIZE_PLANS_AS_DOTS = True # If True, then the predicted paths of the dynamic obstacles will be rendered in the simulation.
VISUALIZE_MPC_ROLLOUTS = True # If True, then the MPC rollouts will be rendered in the simulation.
VISUALIZE_ROBOT_COL_SPHERES = False # If True, then the robot collision spheres will be rendered in the simulation.
HIGHLIGHT_OBS = False # mark the predicted (or not predicted) dynamic obstacles in the simulation
HIGHLIGHT_OBS_H = 30
DEBUG_GPU_MEM = False # If True, then the GPU memory usage will be printed on every call to my_world.step()
RENDER_DT = 0.03 # original 1/60. All details were moved to notes/all_dts_in_one_place_explained.txt
PHYSICS_STEP_DT = 0.03 # original 1/60. All details were moved to notes/all_dts_in_one_place_explained.txt
MPC_DT = 0.03 # independent of the other dt's, but if you want the mpc to simulate the real step change, set it to be as RENDER_DT and PHYSICS_STEP_DT.
HEADLESS_ISAAC = False 

collision_obstacles_cfg_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/collision_obstacles.yml"
robots_cfgs_dir = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/multi_arm_decentralized/robot/franka"
mpc_cfg_overide_files_dir = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/multi_arm_decentralized/mpc"
robots_collision_spheres_configs_parent_dir = "curobo/src/curobo/content/configs/robot"

################### Imports and initiation ########################
if True: # imports and initiation (put it in an if statement to collapse it)
    import argparse
    parser = argparse.ArgumentParser()
    
    if SIMULATING:
        parser.add_argument( 
            "--show_bnd_spheres",
            action="store_true",
            help="Render bounding sphere approximations of each obstacle in real-time.",
            default=False,
        )
        args = parser.parse_args()
        # CRITICAL: Isaac Sim must be imported FIRST before any other modules
        try:
            import isaacsim
        except ImportError:
            pass
            
        from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics,make_world
        from projects_root.utils.usd_utils import load_usd_to_stage

        # Init Isaac Sim app
        simulation_app = init_app() # SimulationApp
        # Omniverse and IsaacSim modules
        from omni.isaac.core import World
        
        
        # Load USD to stage
        stage = load_usd_to_stage("usd_collection/envs/cv_new.usd") # pxr.Usd.Stage
        
        # Init Isaac Sim world
        my_world:World = make_world(ground_plane=True, set_default_prim=True, to_Xform=True)
        
        # Enable GPU dynamics if needed
        if ENABLE_GPU_DYNAMICS:
            activate_gpu_dynamics(my_world)
        # Set simulation dt
        my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT)
        
        from projects_root.utils.helper import add_extensions # available only after app initiation
        add_extensions(simulation_app, None if not HEADLESS_ISAAC else 'true') # in all of the examples of curobo it happens somwhere around here, before the simulation begins. I am not sure why, but I kept it as that. 
        
        

    else:
        # TODO: figure out what we do when SIMULATING=False (should not be too hard)
        my_world = None
        usd_help = None
        stage = None
        tensor_args = None
        simulation_app = None
    
    # Third party modules (moved after Isaac Sim initialization)
    import time
    from threading import Thread, Event, Lock
    from typing import List, Optional, Tuple, Any
    import torch
    import os
    import numpy as np
    
    # Our modules
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import init_runtime_topics, get_topics
    from projects_root.utils.transforms import transform_poses_batched
    from projects_root.autonomous_arm import ArmMpc
    from projects_root.utils.draw import draw_points
    from projects_root.utils.colors import npColors
    # CuRobo modules
    from curobo.geom.types import Sphere, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.util.logger import setup_logger
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import  load_yaml
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
    from projects_root.autonomous_arm import AutonomousArm
    from projects_root.utils.world_model_wrapper import WorldModelWrapper
    from projects_root.utils.handy_utils import get_rollouts_in_world_frame
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
    from projects_root.utils.usd_pose_helper import get_stage_poses, list_relevant_prims

    # Prevent cuda out of memory errors. Backward competebility with curobo source code...
    a = torch.zeros(4, device="cuda:0")

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
        n_spheres=n_spheres,
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











def calculate_robot_sphere_count(robot_cfg):
    """
    Calculate the number of collision spheres for a robot from its configuration.
    
    Args:
        robot_cfg: Robot configuration dictionary
        
    Returns:
        int: Total number of collision spheres (base + extra)
    """
    # Get collision spheres configuration
    collision_spheres = robot_cfg["kinematics"]["collision_spheres"]
    
    # Handle two cases:
    # 1. collision_spheres is a string path to external file (e.g., Franka)
    # 2. collision_spheres is inline dictionary (e.g., UR5e)
    if isinstance(collision_spheres, str):
        # External file case  
        collision_spheres_cfg = load_yaml(os.path.join(robots_collision_spheres_configs_parent_dir, collision_spheres))
        collision_spheres_dict = collision_spheres_cfg["collision_spheres"]
    else:
        # Inline dictionary case
        collision_spheres_dict = collision_spheres
    
    # Count spheres by counting entries in each link's sphere list
    sphere_count = 0
    for link_name, spheres in collision_spheres_dict.items():
        if isinstance(spheres, list):
            sphere_count += len(spheres)
    
    # Add extra collision spheres
    extra_spheres = robot_cfg["kinematics"].get("extra_collision_spheres", {})
    extra_sphere_count = 0
    for obj_name, count in extra_spheres.items():
        extra_sphere_count += count
        
    return sphere_count, extra_sphere_count


def parse_meta_configs(meta_config_paths: List[str]) -> Tuple[List[str], List[str]]:
    """
    Parse meta-configuration files to extract robot and MPC config paths.
    
    Args:
        meta_config_paths: List of paths to meta-config YAML files
        
    Returns:
        Tuple of (robot_config_paths, mpc_config_paths)
    """
    robot_config_paths = []
    mpc_config_paths = []
    
    for meta_path in meta_config_paths:
        meta_config = load_yaml(meta_path)
        robot_config_paths.append(meta_config["robot"])
        mpc_config_paths.append(meta_config["mpc"])
        
    return robot_config_paths, mpc_config_paths

def init_ccheck_wcfg_in_real() -> WorldConfig:    
    """
    Make the initial collision check world configuration.
    This is the world configuration that will be used for collision checking.
    Note: must be collision check WorldConfig! not a regular WorldConfig.
    Also obstacles needs to be expressed in robot frame!
    """
    return WorldConfig() # NOTE: this is just a placeholder for now. See TODO
    # TODO:
    # Get obstacles from real world
    # Convert to *collision check* world WorldConfig! (not a regular WorldConfig)
    # Obstacles need to be expressed in robot frame!
    # Return the *collision check* WorldConfig
    # See init_ccheck_wcfg_in_sim() for an example of how to do this in simulation

def init_ccheck_wcfg_in_sim(usd_help:UsdHelper, robot_prim_path:str, target_prim_path:str, ignore_substrings:List[str])->WorldConfig:    
    """
    Make the initial collision check world configuration.
    This is the world configuration that will be used for collision checking.
    Note: must be collision check WorldConfig! not a regular WorldConfig.
    Also obstacles needs to be expressed in robot frame!
    """
    # Get obstacles from simulation and convert to WorldConfig (not yet collision check world WorldConfig!)
    cu_world_R = usd_help.get_obstacles_from_stage( 
        only_paths=['/World'], # look for obs only under the world prim path
        reference_prim_path=robot_prim_path, # obstacles are expressed in robot frame! (not world frame)
        ignore_substring=ignore_substrings
    )

    # Convert raw WorldConfig to collision check world WorldConfig! (Must!)
    cu_world_R = cu_world_R.get_collision_check_world()
    
    return cu_world_R



def define_run_setup(n_robots:int):
    """
    returns:
        X_np: list of robot poses in world frame
        col_pred_with: list of lists of robot indices that each robot will use for dynamic obs prediction. In entry i, list of indices of robots that the ith robot will use for dynamic obs prediction
        X_targets_R: list of robot target (initial) poses in robot frame
        plot_costs: list of booleans that indicate whether to plot the cost function for each robot
        target_colors: list of colors for each robot target (ordered by RGBY)
    """

    # robot targets, expressed in robot frames
    X_targets = [0, 0, 0.5, 0, 1, 0, 0] # position and orientation of all targets in world frame (x,y,z,qw, qx,qy,qz)
    
    match(n_robots):
        case 1:
            X_robot = [[-0.5,0,0,1,0,0,0]] # position and orientation of the robot in world frame (x,y,z,qw, qx,qy,qz)
            col_pred_with = [[]]
            plot_costs = [False]
            target_colors = [npColors.red]  
        case 2: # 2 robots in a line
            X_robot = [[-0.5, 0, 0, 1, 0, 0, 0], [0.5, 0, 0, 1, 0, 0, 0]] 
            col_pred_with = [[1], [0]]
            plot_costs = [False, False]
            target_colors = [npColors.red, npColors.green ]
        case 3: # 3 robots in a triangle 
            X_robot = [[0.7071,-0.5 ,0, 1, 0, 0, 0], [-0.7071, -0.5, 0, 1, 0, 0, 0], [0, 0.5, 0, 1, 0, 0, 0]]
            col_pred_with = [[1,2], [0,2], [0,1]]
            plot_costs = [False, False, False]
            target_colors = [npColors.red, npColors.green, npColors.blue]
        case 4: # 4 robots in a square
            X_robot = [[-0.5, -0.5, 0, 1, 0, 0, 0], [-0.5, 0.5, 0, 1, 0, 0, 0], [0.5, 0.5, 0, 1, 0, 0, 0], [0.5, -0.5, 0, 1, 0, 0, 0]]
            col_pred_with = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
            plot_costs = [False, False, False, False]
            target_colors = [npColors.red, npColors.green, npColors.blue, npColors.yellow]

    # technical steps:
    # 1. convert to numpy arrays
    X_robot_np = [np.array(Xi, dtype=np.float32) for Xi in X_robot]
    
    # 2. express targets in robot frames
    X_targets_R = [list(np.array(X_targets[:3]) - X_robot_np[i][:3]) + list(X_targets[3:]) for i in range(n_robots)] 
    return X_robot_np, col_pred_with, X_targets_R, plot_costs, target_colors

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
               cu_world_never_add:List[str],
               cu_world_never_update:List[str],
               usd_help:Optional[UsdHelper]=None
               ):
    """
    Background loop that runs one MPC cycle whenever t_idx increments.
    Args:
        robot_idx: Index of the robot to control
        stop_event: Event to signal the robot thread to stop
        get_t_idx: Function to get the current time step index
        t_lock: Lock to protect access to the time step index
        physx_lock: Lock to protect access to the physical simulation state
        plans_lock: Lock to protect access to the plans
        robots: List of ArmMpc instances
        col_pred_with: List of lists of robot indices that each robot will use for dynamic obs prediction
        plans: List of plans for each robot
        tensor_args: TensorDeviceType object
        custom_substrings: List of strings to ignore in the collision check world configuration (apart from the basic default ones (see init_ccheck_wcfg_in_sim))
    """
    
    
    last_step = -1
    r: ArmMpc = robots[robot_idx]
    
    # Initialize collision check world configuration
    if SIMULATING:
        assert usd_help is not None # Type assertion for linter
        r.reset_wmw(init_ccheck_wcfg_in_sim(usd_help, r.prim_path, r.target_prim_path, cu_world_never_add)) 
    else:
        r.reset_wmw(init_ccheck_wcfg_in_real()) 

    # Control loop
    while not stop_event.is_set():
        # wait for new time step
        with t_lock:
            cur_step = get_t_idx()
        if cur_step == last_step:
            time.sleep(1e-7)
            continue
        last_step = cur_step

        with physx_lock:
            # publish 
            if OBS_PREDICTION:
                r.publish(plans, plans_lock)
        
            # sense
            if SIMULATING:
                update_obs_callback = (r.update_obs_in_sim, {'usd_help':usd_help, 'ignore_list':cu_world_never_add + cu_world_never_update})
                update_target_callback = (r.update_target_in_sim, {})
                update_joint_state_callback = (r.update_joint_state_in_sim, {})
            else:
                update_obs_callback = (r.update_obs_in_real, {})
                update_target_callback = (lambda x: print(f"TODO: Implement update_target_callback in real world"), {})
                update_joint_state_callback = (lambda x: print(f"TODO: Implement update_joint_state_callback in real world"), {})
            
            if r.use_col_pred:
                r.sense(update_obs_callback, update_target_callback, update_joint_state_callback, plans, col_pred_with[robot_idx], cur_step, tensor_args, robot_idx,plans_lock)
            else:
                r.sense(update_obs_callback, update_target_callback, update_joint_state_callback)
        
        # plan (improved curobo mpc planning, (torch-heavy, no PhysX))
        action = r.plan(max_attempts=2)
        # command* 
        r.command(action, num_times=1, physx_lock=physx_lock)

def is_dyn_obs_cost_in_cfg(mpc_cfg:dict) -> bool:
    """
    Check if dynamic obstacle cost is enabled in the MPC config.
    """
    return "cost" in mpc_cfg and "custom" in mpc_cfg["cost"] and "arm_base" in mpc_cfg["cost"]["custom"] and "dynamic_obs_cost" in mpc_cfg["cost"]["custom"]["arm_base"]
  
def publish_robot_context(robot_idx:int, robot_context:dict,robot_pose:list, n_obstacle_spheres:int, robot_sphere_count:int, mpc_cfg:dict, col_pred_with_robot:List[int], mpc_config_paths:List[str], robot_config_paths:List[str], robot_sphere_counts_split:List[Tuple[int, int]]):
    """
    Publish robot context ("topics") to the environment topics.
    TODO: Re-design this whole approach, give better names to the variables ()
    """

    # Populate robot context directly in env_topics[i]
    robot_context["env_id"] = 0
    robot_context["robot_id"] = robot_idx
    robot_context["robot_pose"] = robot_pose
    robot_context["n_obstacle_spheres"] = n_obstacle_spheres
    
    robot_context["n_own_spheres"] = robot_sphere_count
    robot_context["horizon"] = mpc_cfg["model"]["horizon"]
    robot_context["n_rollouts"] = mpc_cfg["mppi"]["num_particles"]
    robot_context["col_pred_with"] = col_pred_with_robot
    
    # Add new fields for sparse sphere functionality
    robot_context["mpc_config_paths"] = mpc_config_paths
    robot_context["robot_config_paths"] = robot_config_paths
    robot_context["robot_sphere_counts"] = robot_sphere_counts_split  # [(base, extra), ...]


def main(meta_config_paths: List[str]):
    """
    Main function for multi-robot MPC simulation with heterogeneous robot models.
    
    Args:
        meta_config_paths: List of paths to meta-configuration files, each specifying  robot and MPC
            config paths for one robot
        
    """
    # ------------
    # Curobo setup 
    # ------------
    setup_logger("warn") # curobo logger in warn mode
    tensor_args = TensorDeviceType() # 
    if SIMULATING:   
        # Initialize UsdHelper (helper for USD stage operations by curobo)
        usd_help = UsdHelper()  
        stage = my_world.stage  # get the stage from the world
        usd_help.load_stage(stage) # set self.stage to the stage (self=usd_help)
        
        # set /World as Xform prim, and make it the default prim
        stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
        
        # Make also /curobo as Xform prim
        _curobo_xform = stage.DefinePrim("/curobo", "Xform")  # Transform for CuRobo-specific objects


    # ------------------
    # Config files setup
    # ------------------    
    # Parse meta-configurations to get robot and MPC config paths
    robot_config_paths, mpc_config_paths = parse_meta_configs(meta_config_paths)
    robot_cfgs, mpc_cfgs = [], []
    for i, (robot_path, mpc_path) in enumerate(zip(robot_config_paths, mpc_config_paths)):
        robot_cfgs.append(load_yaml(robot_path)["robot_cfg"])
        mpc_cfgs.append(load_yaml(mpc_path))
        print(f"Robot {i}: robot_config='{robot_path}', mpc_config='{mpc_path}'")
        
    # -----------------------------------------
    # Scenario setup (simulation or real world)
    # -----------------------------------------
    
    n_robots = len(meta_config_paths)
    
    # basic setup of the scenario (robot poses, target poses, target colors, etc...)
    X_robots, col_pred_with, X_targets_R, plot_costs, target_colors = define_run_setup(n_robots)
    
    # runtime topics 
    # (for communication between robots, storing buffers for shared data like robots plans etc...)
    # TODO: Change to ros topics
    init_runtime_topics(n_envs=1, robots_per_env=n_robots) 
    runtime_topics = get_topics()
    env_topics:List[dict] = runtime_topics.get_default_env() if runtime_topics is not None else []
    
    # robot idx lists (joint indices, will be initialized later)
    robot_idx_lists:List[Optional[List]] = [None for _ in range(n_robots)] 

    # Calculate sphere counts for all robots BEFORE creating instances
    robot_sphere_counts_split:List[Tuple[int, int]] = [calculate_robot_sphere_count(robot_cfg) for robot_cfg in robot_cfgs] # split[0] = 'valid' (base, no extra), split[1] = extra
    robot_sphere_counts:List[int] = [split[0] + split[1] for split in robot_sphere_counts_split] # total (base + extra) sphere count (base + extra)
    robot_sphere_counts_no_extra:List[int] = [split[0] for split in robot_sphere_counts_split] # valid (base only) sphere count (base only)

    
    # ArmMpc instances for each robot (supports heterogeneous robot models)
    robots:List[ArmMpc] = []
    for i in range(n_robots):
        robots.append(ArmMpc(
            robot_cfgs[i], 
            my_world, # TODO: figure out what we do when SIMULATING=False (should not be too hard)
            usd_help, # TODO: figure out what we do when SIMULATING=False (should not be too hard)
            env_id=0,
            robot_id=i,
            p_R=X_robots[i][:3],
            q_R=X_robots[i][3:], 
            p_T_R=np.array(X_targets_R[i][:3]),
            q_T_R=np.array(X_targets_R[i][3:]), 
            target_color=target_colors[i],
            plot_costs=plot_costs[i],
            override_particle_file=mpc_config_paths[i],  # Use individual MPC config per robot
            n_coll_spheres=robot_sphere_counts[i],  # Total spheres (base + extra)
            n_coll_spheres_valid=robot_sphere_counts_no_extra[i],  # Valid spheres (base only)
            use_col_pred=OBS_PREDICTION and len(col_pred_with[i]) > 0  # Enable collision prediction
            )
        )
    if SIMULATING:
        # reset default prim to /World
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World")) # TODO: Try removing this and check if it breaks anything (if nothing breaks, remove it)    
        wait_for_playing(my_world, simulation_app, autoplay=True) 
    
    # -----------------------------------------------------------------------------------------
    # Publish "contexts" 
    # (each robot context serves robot and other robots solver initialization)
    # -----------------------------------------------------------------------------------------
    for i in range(n_robots):
        if is_dyn_obs_cost_in_cfg(mpc_cfgs[i]):
            n_obstacle_spheres:int = sum(robot_sphere_counts[j] for j in col_pred_with[i])
            # robot is publishing its context to the global env_topics (so itself and other robots can access it from different modules, instead of passing it as an argument) 
            # TODO: This is not the best approach so it'd be better to use ros topics or pass it as an argument when can, but it is the only one that works for now   
            publish_robot_context(i, env_topics[i], X_robots[i].tolist(), n_obstacle_spheres, robot_sphere_counts[i], mpc_cfgs[i], col_pred_with[i], mpc_config_paths, robot_config_paths, robot_sphere_counts_split)
            
    # ----------------------------
    # Initialize solvers
    # ----------------------------
    for i, robot in enumerate(robots):
        # Set robot in initial joint configuration (in curobo they call it  the "retract" config)
        _robot_idx_list = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot_idx_lists[i] = _robot_idx_list
        assert _robot_idx_list is not None # Type assertion for linter    
        robot.init_joints(_robot_idx_list)
        
        # Init robot mpc solver
        robots[i].init_solver(MPC_DT, DEBUG)
        robots[i].robot._articulation_view.initialize() # TODO: This is a technical required step in isaac 4.5 but check if actually needed https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207

    # ----------------------------
    # Start robot threads
    # ----------------------------
    t_idx = 0  # global simulation step index (# = world/sim clock ticks minus 1)
    stop_event = Event()      # Event to signal robot threads to stop
    t_lock = Lock()           # Protects access to shared t_idx
    plans_lock = Lock()       # Protects access to shared plans list
    physx_lock = Lock()       # Protects access to shared physx state (robot state, etc...)
    plans: List[Optional[Any]] = [None for _ in range(len(robots))] # TODO: This is a hack to pass plans to the robot threads, but it is not the best approach so it'd be better to use ros topics or pass it as an argument when can, but it is the only one that works for now
    
    cu_world_never_add = ["/curobo", *[robot.target_prim_path for robot in robots],*[robot.prim_path for robot in robots], "/World/ConveyorTrack", '/World/conveyor_cube'] # never treat these names as obstacles (add them to world model)
    cu_world_never_update = ["/World/defaultGroundPlane", "/World/cv_approx"] # add here any substring of an onstacle you assume remains static throughout the entire simulation!
    
    robot_threads = [Thread(target=ctrl_loop_robot,args=(idx, stop_event, lambda: t_idx, t_lock, physx_lock, plans_lock, robots, col_pred_with, plans, tensor_args, cu_world_never_add, cu_world_never_update, usd_help), daemon=True) for idx in range(len(robots))] # TODO: This is a hack to pass plans to the robot threads, but it is not the best approach so it'd be better to use ros topics or pass it as an argument when can, but it is the only one that works for now
    for th in robot_threads:
        th.start()
    
    # point_visualzer_inputs = [] # empty list for draw_points() inputs
    step_batch_start_time = time.time()
    step_batch_size = 100
    if SIMULATING:
        assert simulation_app is not None # Type assertion for linter (it is not None when SIMULATING=True)
        while simulation_app.is_running():
            with physx_lock:
                my_world.step(render=True)           
            with t_lock:
                t_idx += 1
            if t_idx % step_batch_size == 0: # step batch size = 100
                step_batch_time = time.time() - step_batch_start_time
                print(f"ts: {t_idx}")
                print("num of actions planned by each robot:")
                print([robots[i].n_actions_planned for i in range(len(robots))])
                print(f"overall avg step time: {(step_batch_time/step_batch_size)*1000:.1f} ms")
                step_batch_start_time = time.time()
        simulation_app.close() 
        
    else: # Real world (no simulation)
        while True:
            with t_lock:
                t_idx += 1
            time.sleep(REAL_TIME_EXPECTED_CTRL_DT) # TODO: This is a hack to make the robot threads run at the same speed as the simulation, but it is not the best approach so it'd be better to use ros topics or pass it as an argument when can, but it is the only one that works for now
            
    # Clean up thread pool
    stop_event.set()
    for th in robot_threads:
        th.join()

def resolve_meta_config_path(robot_model:str) -> str:
    """
    Resolves the meta-configuration paths to the absolute paths.
    """
    root_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/meta_cfgs"
    return os.path.join(root_path, f"{robot_model}.yml")

if __name__ == "__main__":
    
    if DEBUG_GPU_MEM:
        torch.cuda.memory._record_memory_history() # https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
    input_args = ['ur5e', 'franka', 'franka']
    main([resolve_meta_config_path(robot_model) for robot_model in input_args])
    
    
     
        


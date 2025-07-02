"""
async version of projects_root/examples/mpc_moving_obstacles_mpc_mpc.py
"""

# Force non-interactive matplotlib backend to avoid GUI operations from worker threads
import os
os.environ.setdefault("MPLBACKEND", "Agg")

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
    # CRITICAL: Isaac Sim must be imported FIRST before any other modules
    try:
        import isaacsim
    except ImportError:
        pass
    
    from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics
    simulation_app = init_app() # must happen before importing other isaac sim modules, or any other module which imports isaac sim modules.
    
    # Third party modules (moved after Isaac Sim initialization)
    import time
    from threading import Thread, Event, Lock
    from typing import List, Optional, Tuple, Any
    import torch
    import os
    import numpy as np
    from projects_root.utils.helper import add_extensions # available only after app initiation
    add_extensions(simulation_app, None if not HEADLESS_ISAAC else 'true') # in all of the examples of curobo it happens somwhere around here, before the simulation begins. I am not sure why, but I kept it as that. 
    # Omniverse and IsaacSim modules
    from omni.isaac.core import World 
    from omni.isaac.core.utils.types import ArticulationAction
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
    from curobo.util.logger import setup_curobo_logger
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import  load_yaml
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
    from projects_root.autonomous_arm import AutonomousArm
    # Prevent cuda out of memory errors. Backward competebility with curobo source code...
    a = torch.zeros(4, device="cuda:0")


# GPU memory debugging function removed for production
    

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
               ):
    """Background loop that runs one MPC cycle whenever t_idx increments."""
    last_step = -1
    r: ArmMpc = robots[robot_idx]
    
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
            if r.use_col_pred:
                r.sense(plans, col_pred_with[robot_idx], cur_step, tensor_args, robot_idx,plans_lock)
            else:
                r.sense()
        # plan (improved curobo mpc planning, (torch-heavy, no PhysX))
        action = r.plan(max_attempts=2)
        # command* 
        r.command(action, num_times=1, physx_lock=physx_lock)
    
def main(meta_config_paths: List[str]):
    """
    Main function for multi-robot MPC simulation with heterogeneous robot models.
    
    Args:
        meta_config_paths: List of paths to meta-configuration files, each specifying  robot and MPC
            config paths for one robot
        
    """


    n_robots = len(meta_config_paths)
    print(f"Starting multi-robot simulation with {n_robots} robots")

    # Parse meta-configurations to get robot and MPC config paths
    robot_config_paths, mpc_config_paths = parse_meta_configs(meta_config_paths)
    
    # Print robot configuration summary
    for i, (robot_path, mpc_path) in enumerate(zip(robot_config_paths, mpc_config_paths)):
        print(f"Robot {i}: robot_config='{robot_path}', mpc_config='{mpc_path}'")
    
    # isaac sim and open usd setup
    usd_help = UsdHelper()  # Helper for USD stage operations
    my_world = World(stage_units_in_meters=1.0) 
    my_world.scene.add_default_ground_plane()
    my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT) 
    if ENABLE_GPU_DYNAMICS:
        activate_gpu_dynamics(my_world)
    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")  # Root transform for all objects
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")  # Transform for CuRobo-specific objects
    
    # curobo setup (logger, tensor device, etc...)
    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType()  # Device configuration for tensor operations
    
    # basic setup of the scenario (robot poses, target poses, target colors, etc...)
    X_robots, col_pred_with, X_targets_R, plot_costs, target_colors = define_run_setup(n_robots)
    
    # runtime topics (for communication between robots, storing buffers for shared data like robots plans etc...)
    init_runtime_topics(n_envs=1, robots_per_env=n_robots) 
    runtime_topics = get_topics()
    env_topics = runtime_topics.get_default_env() if runtime_topics is not None else []
    
    
    # curobo joints
    # robots_cu_js: List[Optional[JointState]] = [None for _ in range(n_robots)]# for visualization of robot spheres
    # robot idx lists (joint indices, will be initialized later)
    robot_idx_lists:List[Optional[List]] = [None for _ in range(n_robots)] 
    # collision caches (curobo collision checker for each robot)
    robots_collision_caches = [{"obb": 5, "mesh": 5} for _ in range(n_robots)]
    # collision world model (curobo world model for each robot. Empty for now, will be initialized later)
    robot_world_models = [WorldConfig() for _ in range(n_robots)]
    # curobo robot configs for each robot
    robot_cfgs = [load_yaml(robot_path)["robot_cfg"] for robot_path in robot_config_paths]
    # collision checker (curobo collision checker for each robot, empty for now, will be initialized later)
    ccheckers = []
    # Calculate sphere counts for all robots BEFORE creating instances
    robot_sphere_counts_split = [calculate_robot_sphere_count(robot_cfg) for robot_cfg in robot_cfgs]
    robot_sphere_counts = [split[0] + split[1] for split in robot_sphere_counts_split]
    robot_sphere_counts_valid = [split[0] for split in robot_sphere_counts_split]

    
    # ArmMpc instances for each robot (supports heterogeneous robot models)
    robots:List[ArmMpc] = []
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
            override_particle_file=mpc_config_paths[i],  # Use individual MPC config per robot
            n_coll_spheres=robot_sphere_counts[i],  # Total spheres (base + extra)
            n_coll_spheres_valid=robot_sphere_counts_valid[i],  # Valid spheres (base only)
            use_col_pred=OBS_PREDICTION and len(col_pred_with[i]) > 0  # Enable collision prediction
            )
        )
    
    # ENVIRONMENT OBSTACLES - INITIALIZATION
    # col_ob_cfg = load_yaml(collision_obstacles_cfg_path)
    # env_obstacles = [] # list of obstacles in the world
    # for obstacle in col_ob_cfg:
    #     obstacle = Obstacle(my_world, **obstacle)
    #     for i in range(len(robot_world_models)):
    #         world_model_idx = obstacle.add_to_world_model(robot_world_models[i], X_robots[i])#  usd_helper=usd_help) # inplace modification of the world model with the obstacle
    #         print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")
    #     env_obstacles.append(obstacle) # add the obstacle to the list of obstacles
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    
    # wait for play button in simulator to be pushed
    wait_for_playing(my_world, simulation_app,autoplay=True) 
    
    
    
    # FIRST: Pre-populate robot context with sphere counts BEFORE init_solver()
    # robot_sphere_counts already calculated above when creating robots
    
    
    # Populate robot context BEFORE solver initialization
    for i in range(n_robots):
        # Get MPC config values for this specific robot
        mpc_config = load_yaml(mpc_config_paths[i])
        
        # Check if this robot has DynamicObsCost enabled
        has_dynamic_obs_cost = (
            "cost" in mpc_config and 
            "custom" in mpc_config["cost"] and 
            "arm_base" in mpc_config["cost"]["custom"] and 
            "dynamic_obs_cost" in mpc_config["cost"]["custom"]["arm_base"]
        )
        
        if has_dynamic_obs_cost:
            n_obstacle_spheres = sum(robot_sphere_counts[j] for j in col_pred_with[i])
            
            # Populate robot context directly in env_topics[i]
            env_topics[i]["env_id"] = 0
            env_topics[i]["robot_id"] = i
            env_topics[i]["robot_pose"] = X_robots[i].tolist()
            env_topics[i]["n_obstacle_spheres"] = n_obstacle_spheres
            env_topics[i]["n_own_spheres"] = robot_sphere_counts[i]
            env_topics[i]["horizon"] = mpc_config["model"]["horizon"]
            env_topics[i]["n_rollouts"] = mpc_config["mppi"]["num_particles"]
            env_topics[i]["col_pred_with"] = col_pred_with[i]
            
            # Add new fields for sparse sphere functionality
            env_topics[i]["mpc_config_paths"] = mpc_config_paths
            env_topics[i]["robot_config_paths"] = robot_config_paths
            env_topics[i]["robot_sphere_counts"] = robot_sphere_counts_split  # [(base, extra), ...]


    for i, robot in enumerate(robots):
        # Set robots in initial joint configuration (in curobo they call it  the "retract" config)
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        if robot_idx_lists[i] is None:
            raise RuntimeError(f"Failed to get DOF indices for robot {i}")
        # Type assertion for linter
        idx_list = robot_idx_lists[i]
        assert idx_list is not None
        robot.init_joints(idx_list)
        # Init robot mpc solver
        robots[i].init_solver(robot_world_models[i],robots_collision_caches[i], MPC_DT, DEBUG)
        # Some technical required step in isaac 4.5 https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207
        robots[i].robot._articulation_view.initialize() 
        # # Get initialized collision checker of robot
        # checker = robots[i].get_cchecker() # available only after init_solver
        # ccheckers.append(checker)
        # for j in range(len(env_obstacles)):
        #     env_obstacles[j].register_ccheckers([checker])
    
    # for i in range(len(env_obstacles)):
    #     env_obstacles[i].register_ccheckers(ccheckers)

     
    # -------------- Threaded robot loops --------------
    stop_event = Event()
    t_lock = Lock()          # Protects access to shared t_idx
    plans_lock = Lock()      # Protects access to shared plans list
    physx_lock = Lock()
    plans: List[Optional[Any]] = [None for _ in range(len(robots))]

    t_idx = 0  # global simulation step index
    # total_steps_time = 0.0
    
        
    # Spawn one thread per robot
    robot_threads = [Thread(target=ctrl_loop_robot, args=(idx, stop_event, lambda: t_idx, t_lock, physx_lock, plans_lock, robots, col_pred_with, plans, tensor_args), daemon=True) for idx in range(len(robots))]
    for th in robot_threads:
        th.start()
    
    # point_visualzer_inputs = [] # empty list for draw_points() inputs
    step_batch_start_time = time.time()
    step_batch_size = 100
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
    
    
     
        


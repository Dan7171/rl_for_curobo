"""

# Prior reccomended reading:
https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/core_api_overview.html 
https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/manual_standalone_python.html
https://docs.isaacsim.omniverse.nvidia.com/latest/python_scripting/manual_standalone_python.html
https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/index.html
https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#isaac-sim-glossary-world
World:
    - https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#world:~:text=World%23,from%20different%20extensions.
    -https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World
    - https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World:~:text=.cleaq_Render_callbacks()-,World,-%23
    - Rendering:
        - rendering means rendering a frame of the current application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will be refreshed with this dt as well if running non-headless. Defaults to None.
        - see: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World
    - physics_dt and rendering_dt:
        - World is initialized with set_defaults (bool, optional, defaults to True): defaults settings are applied: 
        [physics_dt = 1.0/ 60.0: dt between physics steps.
        stage units in meters = 0.01 (i.e in cms),
        rendering_dt = 1.0 / 60.0,#  dt between rendering steps. Note: rendering means rendering a frame of the current application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will be refreshed with this dt as well if running non-headless. Defaults to None.
        gravity = -9.81 m / s ccd_enabled, stabilization_enabled, gpu dynamics turned off, broadcast type is MBP, solver type is TGS].
    - world.current_time_step_index: # current number of physics steps that have elapsed since the simulation was *played*(play button pressed) https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.core/docs/index.html?highlight=current_time_step_index

Scene: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#world:~:text=from%20different%20extensions.-,Scene,thus%20providing%20an%20easy%20way%20to%20set/%20get%20its%20common%20properties.,-Task
Articulation: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#world:~:text=Articulation%23,an%20easy%20way.
Application: (SimulationApp) https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#application
Cheat sheet: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#re
Extensions: Extensions are plug-ins to Omniverse Kit that extend its capabilities. They are offered with complete source code to help developers easily create, add, and modify the tools and workflows they need to be productive. Extensions are the core building blocks of Omniverse Kit based applications. https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#real-time-render-mode:~:text=Extensions%23,for%20more%20details.
Stage: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#real-time-render-mode:~:text=specifies%20material%20parameters.-,Stage,See%20the%20USD%20Glossary%20of%20Terms%20%26%20Concepts%20for%20more%20details.,-Prim
Prim: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#real-time-render-mode:~:text=for%20more%20details.-,Prim,See%20the%20USD%20Glossary%20of%20Terms%20%26%20Concepts%20for%20more%20details.,-Mesh
Mesh: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_glossary.html#real-time-render-mode:~:text=Mesh%23,a%20UseGeomMesh%20class.
Docs API (isaac sim and omniverse) https://docs.isaacsim.omniverse.nvidia.com/latest/reference_python_api.html


Model Predictive Control (MPC) example with moving obstacles in Isaac Sim.

This example demonstrates:
1. Real-time MPC for robot motion planning
2. Dynamic obstacle avoidance with moving obstacles (cube or sphere)
3. Support for both physical and non-physical obstacles
4. Integration with NVIDIA's Isaac Sim robotics simulator

The robot follows a target while avoiding a moving obstacle. The obstacle can be:
- Physical: Follows physics laws and can collide with the robot
- Non-physical: Moves in a predetermined way without physical interactions

Usage:
    omni_python mpc_example_with_moving_obstacle.py [options]
    
Example options:
    --obstacle_linear_velocity -0.1 0.1 0.0  # Move diagonally (default: [-0.1, 0.0, 0.0])
    --obstacle_size 0.15    # Set obstacle size (default: 0.1)
    --obstacle_color 0.0 1.0 0.0  # Green color (default: [1.0, 0.0, 0.0])
    --autoplay False    # Disable autoplay (default: True)
"""

# ############################## Run settings ##############################


SIMULATING = True # if False, then we are running the robot in real time (i.e. the robot will move as fast as the real time allows)
REAL_TIME_EXPECTED_CTRL_DT = 0.03 #1 / (The expected control frequency in Hz). Set that to the avg time measurded between two consecutive calls to my_world.step() in real time. To print that time, use: print(f"Time between two consecutive calls to my_world.step() in real time, run with --print_ctrl_rate "True")
ENABLE_GPU_DYNAMICS = True # # GPU DYNAMICS - OPTIONAL (originally was disabled)# GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU. However, this will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. If enabling GPU dynamics slows down the simulation, it may be that your GPU is not able to handle the additional load. You can enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.# See: https://docs-prod.omniverse.nvidia.com/isaacsim/latest/reference_material/speedup_cheat_sheet.html?utm_source=chatgpt.com # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
OBS_PREDICTION  = True # If True, this would be what the original MPC cost function could handle. False means that the cost will consider obstacles as moving and look into the future, while True means that the cost will consider obstacles as static and not look into the future.
DEBUG = False # Currenly, the main feature of True is to run withoug cuda graphs. When its true, we can set breakpoints inside cuda graph code (like in cost computation in "ArmBase" for example)  
VISUALIZE_PREDICTED_OBS_PATHS = False # If True, then the predicted paths of the dynamic obstacles will be rendered in the simulation.
VISUALIZE_MPC_ROLLOUTS = True # If True, then the MPC rollouts will be rendered in the simulation.
VISUALIZE_ROBOT_COL_SPHERES = False # If True, then the robot collision spheres will be rendered in the simulation.
HIGHLIGHT_OBS = False # mark the predicted (or not predicted) dynamic obstacles in the simulation
HIGHLIGHT_OBS_H = 30
DEBUG_GPU_MEM = False # If True, then the GPU memory usage will be printed on every call to my_world.step()
RENDER_DT = 0.03 # original 1/60. All details were moved to notes/all_dts_in_one_place_explained.txt
PHYSICS_STEP_DT = 0.03 # original 1/60. All details were moved to notes/all_dts_in_one_place_explained.txt
MPC_DT = 0.03 # independent of the other dt's, but if you want the mpc to simulate the real step change, set it to be as RENDER_DT and PHYSICS_STEP_DT.
SUPPORT_ASSETS_OUTSIDE_CONFIG = True # Turn on if you want to "drag and drop" assets to the stage manually. Turn otherwise because it takes longer to load the assets.
ASSET_FIXATION_T = 10 # If using SUPPORT_ASSETS_OUTSIDE_CONFIG, After this time step, no updates to collision models will be made, even if they are changed in the stage. This is to prevent the collision model from being updated too frequently after some point. Set to -1 to disable.
################### Imports and initiation ########################
if True: # imports and initiation (put it in an if statement to collapse it)
    # arg parsing:
    import argparse
    parser = argparse.ArgumentParser(
        description="CuRobo MPC example with moving obstacle in Isaac Sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    # Default behavior (red cuboid moving at -0.1 m/s in x direction, physics enabled)
    omni_python mpc_example_with_moving_obstacle.py

    # Sphere obstacle moving diagonally with autoplay disabled
    omni_python mpc_example_with_moving_obstacle.py  --obstacle_linear_velocity -0.1 0.1 0.0 --obstacle_size 0.15 --autoplay False

    # Blue cuboid starting at specific position with physics enabled
    omni_python mpc_example_with_moving_obstacle.py  --obstacle_initial_pos 1.0 0.5 0.3 --obstacle_color 0.0 0.0 1.0 --obstacle_mass 1.0

    # Green sphere moving in y direction with custom size and physics disabled
    omni_python mpc_example_with_moving_obstacle.py  --obstacle_linear_velocity 0.0 0.1 0.0 --obstacle_size 0.2 --obstacle_color 0.0 1.0 0.0 

    # Red cuboid with physics disabled and autoplay disabled
    omni_python mpc_example_with_moving_obstacle.py  --autoplay False
    """
    )
    parser.add_argument(
        "--headless_mode",
        type=str,
        default=None,
        help="Run in headless mode. Options: [native, websocket]. Note: webrtc might not work.",
    )
    parser.add_argument(
        "--autoplay",
        help="Start simulation automatically without requiring manual play button press",
        default="True",
        type=str,
        choices=["True", "False"],
    )
    parser.add_argument(
        "--obstacle_mass",
        type=float,
        default=1.0,
        help="Mass of the obstacle in kilograms",
    )
    parser.add_argument(
        "--gravity_enabled",
        help="Enable gravity for the obstacle  ",
        default="False",
        type=str,
        choices=["True", "False"],
    )
    parser.add_argument(
        "--print_ctrl_rate",
        default="False",
        type=str,
        choices=["True", "False"],
        help="When True, prints the control rate",
    )
    args = parser.parse_args()
    args.autoplay = args.autoplay.lower() == "true"
    args.print_ctrl_rate = args.print_ctrl_rate.lower() == "true"

    # third party modules
    import time
    import signal
    from typing import List, Optional
    import torch
    import numpy as np
    from dataclasses import dataclass
    import copy
    # Isaac Sim app initiation and isaac sim modules
    from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics
    simulation_app = init_app(
        {
        "headless": args.headless_mode is not None, 
        "width": "1920",
        "height": "1080"
        }
    ) # must happen before importing other isaac sim modules, or any other module which imports isaac sim modules.
    from omni.isaac.core import World 
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf
    # from omni.usd import get_context
    # Our modules
    from projects_root.utils.helper import add_extensions
    from projects_root.autonomous_franka import FrankaMpc
    from projects_root.utils.draw import draw_points
    # CuRobo modules
    from curobo.geom.types import Sphere, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.util.logger import setup_curobo_logger
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import  load_yaml
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
    from projects_root.autonomous_franka import AutonomousFranka
    from projects_root.utils.curobo_world_models import update_world_model
    a = torch.zeros(4, device="cuda:0") # prevent cuda out of memory errors (took from curobo examples)

######################### HELPER ##########################
@dataclass
class TargetColors:
    red: np.ndarray = np.array([0.5,0,0])
    green: np.ndarray = np.array([0,0.5,0])
    blue: np.ndarray = np.array([0,0,0.5])
    yellow: np.ndarray = np.array([0.5,0.5,0])
    purple: np.ndarray = np.array([0.5,0,0.5])
    orange: np.ndarray = np.array([0.5,0.3,0])
    pink: np.ndarray = np.array([0.5,0.3,0.5])
  
def print_rate_decorator(func, print_ctrl_rate, rate_name, return_stats=False):
    def wrapper(*args, **kwargs):
        duration, rate = None, None
        if print_ctrl_rate:
            start = time.time()
        result = func(*args, **kwargs)
        if print_ctrl_rate:
            end = time.time()
            duration = end - start
            rate = 1.0 / duration
            print(f"{rate_name} duration: {duration:.3f} seconds, {rate_name} frequency: {rate:.3f} Hz") 
        if return_stats:
            return result, (duration, rate)
        else:
            return result
    return wrapper

def print_ctrl_rate_info(t_idx,real_robot_cfm_start_time,real_robot_cfm_start_t_idx,expected_ctrl_freq_at_mpc,MPC_DT):
    """Prints information about the control loop frequncy (desired vs measured) and warns if it's too different.
    Args:
        t_idx (_type_): _description_   
        real_robot_cfm_start_time (_type_): _description_
        real_robot_cfm_start_t_idx (_type_): _description_
        expected_ctrl_freq_at_mpc (_type_): _description_
        step_dt_traj_mpc (_type_): _description_
    """
    if SIMULATING: 
        cfm_total_steps = t_idx # number of control steps we actually executed.
        cfm_total_time = t_idx * RENDER_DT # NOTE: Unless I have a bug, this should be the formula for the total time simulation think passed.

    else:
        cfm_total_steps = t_idx - real_robot_cfm_start_t_idx # offset by the number of steps since the control frequency measurement has started.
        cfm_total_time = time.time() - real_robot_cfm_start_time # offset by the time since the control frequency measurement has started.
        
    cfm_avg_control_freq = cfm_total_steps / cfm_total_time # Average  measured Control Frequency. num of completed actions / total time of actions Hz
    cfm_avg_step_dt = 1 / cfm_avg_control_freq # Average measured control step duration in seconds
    ctrl_freq_ratio = expected_ctrl_freq_at_mpc / cfm_avg_control_freq # What the mpc thinks the control frequency should be / what is actually measured.
    print(f"expected_ctrl_freq_hz: {expected_ctrl_freq_at_mpc:.5f}")    
    print(f"cfm_avg_control_freq: {cfm_avg_control_freq:.5f}")    
    print(f"cfm_avg_step_dt: {cfm_avg_step_dt:.5f}")    
    if ctrl_freq_ratio > 1.05 or ctrl_freq_ratio < 0.95:
        print(f"WARNING! Control frequency ratio is {ctrl_freq_ratio:.5f}. \
            But MPC is 'thinks' that the frequency of sending commands to the robot is {expected_ctrl_freq_at_mpc:.5f} Hz, {cfm_avg_control_freq:.5f} Hz was assigned.\n\
                You probably need to change mpc_config.step_dt(MPC_DT) from {MPC_DT} to {cfm_avg_step_dt})")

def get_sphere_list_from_sphere_tensor(p_spheres:torch.Tensor, rad_spheres:torch.Tensor, sphere_names:list, tensor_args:TensorDeviceType) -> list[Sphere]:
    """
    Returns a list of Sphere objects from a tensor of sphere positions and radii.
    """
    spheres = []
    for i in range(p_spheres.shape[0]):
        p_sphere = p_spheres[i]
        r_sphere = rad_spheres[i].item()
        X_sphere = p_sphere.tolist() + [1,0,0,0]# Pose(tensor_args.to_device(p_sphere), tensor_args.to_device(torch.tensor([1,0,0,0])))
        name_sphere = sphere_names[i]
        spheres.append(Sphere(name=name_sphere, pose=X_sphere, radius=r_sphere))
    return spheres

def get_full_path_to_asset(asset_subpath):
    return get_assets_root_path() + '/Isaac/' + asset_subpath

def load_asset_to_prim_path(asset_subpath, prim_path='', is_fullpath=False):
    """
    Loads an asset to a prim path.
    Source: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.replicator.isaac/docs/index.html?highlight=add_reference_to_stage

    asset_subpath: sub-path to the asset to load. Must end with .usd or .usda. Normally starts with /Isaac/...
    To browse, go to: asset browser in simulator and add /Issac/% your subpath% where %your subpath% is the path to the asset you want to load.
    Note: to get the asset exact asset_subpath, In the simulator, open: Isaac Assets -> brows to the asset (usd file) -> right click -> copy url path and paste it here (the subpath is the part after the last /Isaac/).
    Normally the assets are coming from web, but this tutorial can help you use local assets: https://docs.omniverse.nvidia.com/launcher/latest/it-managed-launcher/content_install.html.

    For example: http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Props/Mugs/SM_Mug_A2.usd -> asset_subpath should be: Props/Mugs/SM_Mug_A2.usd
        
    prim_path: path to the prim to load the asset to. If not provided, the asset will be loaded to the prim path /World/%asset_subpath%    
    is_fullpath: if True, the asset_subpath is a full path to the asset. If False, the asset_subpath is a subpath to the assets folder in the simulator.
    This is useful if you want to load an asset that is not in the {get_assets_root_path() + '/Isaac/'} folder (which is the root folder for Isaac Sim assets (see asset browser in simulator) but custom assets in your project from a local path.



    Examples:
    load_asset_to_prim_path("Props/Mugs/SM_Mug_A2.usd") will load the asset to the prim path /World/Promps_Mugs_SM_Mug_A2
    load_asset_to_prim_path("Props/Mugs/SM_Mug_A2.usd", "/World/Mug") will load the asset to the prim path /World/Mug
    load_asset_to_prim_path("/home/me/some_folder/SM_Mug_A2.usd", "/World/Mug", is_fullpath=True) will load the asset to the prim path /World/Mug
    load_asset_to_prim_path("http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.0/Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd", "/World/KLT_Bin", is_fullpath=True) will load the asset to the prim path /World/KLT_Bin
    """

    # validate asset 
    if not prim_path:
        # prim_name = asset_subpath.split('/')[-1].split('.')[0]
        asset_subpath_as_prim_name = asset_subpath.replace('/', '_').split('.')[0]
        prim_path = f'/World/{asset_subpath_as_prim_name}'
    else:
        prim_path = prim_path
    
    # define full path to asset
    if not is_fullpath:
        asset_fullpath = get_full_path_to_asset(asset_subpath)
    else:
        asset_fullpath = asset_subpath 
    
    # validate asset path
    assert asset_fullpath.endswith('.usd') or asset_fullpath.endswith('.usda'), "Asset path must end with .usd or .usda"
    
    # load asset to prim path (adds the asset to the stage)
    add_reference_to_stage(asset_fullpath, prim_path)
    return prim_path 

def write_stage_to_usd_file(stage,file_path):
    stage.Export(file_path) # export the stage to a temporary USD file
    
def handle_sigint_gpu_mem_debug(signum, frame):
    print("Caught SIGINT (Ctrl+C), first dump snapshot...")
    torch.cuda.memory._dump_snapshot()
    print("Snapshot dumped to dump_snapshot.pickle, you can upload it to the server: https://docs.pytorch.org/memory_viz")
    print("Now raising KeyboardInterrupt to let the original KeyboardInterrupt handler (of nvidia) to close the app")
    raise KeyboardInterrupt # to let the original KeyboardInterrupt handler (of nvidia) to close the app
    

def main():
    """
    Main simulation loop that demonstrates Model Predictive Control (MPC) with moving obstacles.
    
    The simulation:
    1. Sets up the Isaac Sim environment with a robot and moving obstacle
    2. Initializes the MPC solver for real-time motion planning
    3. Runs a continuous loop that:
       - Updates obstacle position (physical or non-physical)
       - Plans robot motion to follow target while avoiding obstacles
       - Executes planned motion
       - Visualizes planned trajectories
       
    The robot follows a target cube while avoiding a moving obstacle. The obstacle can be:
    - Physical: Uses Isaac Sim's physics engine for realistic collisions
    - Non-physical: Moves in a predetermined way without physical interactions
    
    References:
        Isaac Sim Core API: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.core.api/docs/index.html#python-api
    """
    ###########################################################
    ################  SIMULATION INITIALIZATION ###############
    ###########################################################

    
    usd_help = UsdHelper()  # Helper for USD stage operations
    my_world = World(stage_units_in_meters=1.0) 
    my_world.scene.add_default_ground_plane()
    my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT) 
    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")  # Root transform for all objects
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")  # Transform for CuRobo-specific objects
    setup_curobo_logger("error") # "warn" 
    

    
    # world_model = get_world_model_from_current_stage(stage)
    tensor_args = TensorDeviceType()  # Device configuration for tensor operations
    if ENABLE_GPU_DYNAMICS:
        activate_gpu_dynamics(my_world)
    
    # Adding two frankas to the scene
    # # Inspired by curobo/examples/isaac_sim/batch_motion_gen_reacher.py but this time at the same world (the batch motion gen reacher example is for multiple worlds)
    
    X_Robots = [
        np.array([0,0,0,1,0,0,0], dtype=np.float32),
        np.array([1.2,0,0,1,0,0,0], dtype=np.float32)
        ] # (x,y,z,qw, qx,qy,qz) expressed in world frame
    n_robots = len(X_Robots)
    robots_cu_js: List[Optional[JointState]] =[None for _ in range(n_robots)]# for visualization of robot spheres
    robots_collision_caches = [{"obb": 100, "mesh": 100} for _ in range(n_robots)]
    robot_cfgs = [load_yaml(f"projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/franka{i}.yml")["robot_cfg"] for i in range(1,n_robots+1)]
    robot_idx_lists:List[Optional[List]] = [None for _ in range(n_robots)] 
    robot_world_models = [WorldConfig() for _ in range(n_robots)]
    X_Targets = [[0.6, 0, 0.2, 0, 1, 0, 0], [1.8, 0, 0.2, 0, 1, 0, 0]]# [[0.6, 0, 0.2, 0, 1, 0, 0] for _ in range(n_robots)]
    target_colors = [TargetColors.green, TargetColors.red]
    if OBS_PREDICTION:
        col_pred_with = [[1], [0]] # at each entry i, list of indices of robots that the ith robot will use for dynamic obs prediction
   

    robots:List[AutonomousFranka] = []
    for i in range(n_robots):
        robots.append(FrankaMpc(
            robot_cfgs[i], 
            my_world, 
            usd_help, 
            p_R=X_Robots[i][:3],
            q_R=X_Robots[i][3:], 
            p_T=X_Targets[i][:3],
            q_T=X_Targets[i][3:], 
            target_color=target_colors[i],
            )
        )
    # ENVIRONMENT OBSTACLES - INITIALIZATION
    collision_obstacles_cfg_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/collision_obstacles.yml"
    col_ob_cfg = load_yaml(collision_obstacles_cfg_path)
    env_obstacles:List[Obstacle] = [] # list of obstacles in the world
    for obstacle in col_ob_cfg:
        obstacle = Obstacle(my_world, **obstacle)
        for i in range(len(robot_world_models)):
            world_model_idx = obstacle.add_to_world_model(robot_world_models[i], X_Robots[i])#  usd_helper=usd_help) # inplace modification of the world model with the obstacle
            print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")
        env_obstacles.append(obstacle) # add the obstacle to the list of obstacles
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    
    # add extensions
    add_extensions(simulation_app, args.headless_mode) # in all of the examples of curobo it happens somwhere around here, before the simulation begins. I am not sure why, but I kept it as that. 
    
    # wait for the play button to be pressed
    wait_for_playing(my_world, simulation_app,args.autoplay) # wait for the play button to be pressed
    print("Play button was pressed in simulation!")
    
    # initialize robots
    for i, robot in enumerate(robots):
        # Set robots in initial joint configuration (in curobo they call it  the "retract" config)
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot.init_joints(robot_idx_lists[i])
        # init dynamic obs coll predictors
        if OBS_PREDICTION and len(col_pred_with[i]):
            obs_groups_nspheres = [robots[obs_robot_idx].get_num_of_sphers() for obs_robot_idx in col_pred_with[i]]
            robot.init_col_predictor(obs_groups_nspheres, cost_weight=100, manually_express_p_own_in_world_frame=True)
        # initialize solver
        robot.init_solver(robot_world_models[i],robots_collision_caches[i], MPC_DT, DEBUG)
        robot.robot._articulation_view.initialize() # new (isac 4.5) https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207

    # register ccheckers of robots with environment obstacles
    ccheckers = [robot.get_cchecker() for robot in robots] # available only after init_solver
    for i in range(len(env_obstacles)):
        env_obstacles[i].register_ccheckers(ccheckers)

    if not OBS_PREDICTION: 
        # Treat spheres of other robots as static obstacles 
        # (update the ccheckers of each robot with all other robots spheres 
        # everytime the spheres of other robots change, but no prediction of path in the rollouts)
        spheres_as_obs_grouped_by_robot = []
        for i in range(len(robots)):
            roboti_spheres_as_obstacles = []
            p_validspheresRjcurr, _, _ = robots[i].get_current_spheres_state()
            for sphere_idx in range(len(p_validspheresRjcurr)):
                sphere_r = p_validspheresRjcurr[sphere_idx][3]
                sphere_as_cub_dims = [2*sphere_r, 2*sphere_r, 2*sphere_r]
                sphere_obs = Obstacle(my_world, f"R{i}S{sphere_idx}",'cuboid', np.array(p_validspheresRjcurr[sphere_idx].tolist() + [1,0,0,0]), np.array(sphere_as_cub_dims), simulation_enabled=True)
                roboti_spheres_as_obstacles.append(sphere_obs)
            spheres_as_obs_grouped_by_robot.append(roboti_spheres_as_obstacles)
        for i in range(len(robots)):
            checkers_to_reg_on_robot_i_spheres = [ccheckers[j] for j in range(len(ccheckers)) if j != i] # checkers of other robots except i
            for sphere_obs in spheres_as_obs_grouped_by_robot[i]: # for each sphere of robot i register the checkers of other robots (so they will treat i's spheres as obstacles)
                sphere_obs.register_ccheckers(checkers_to_reg_on_robot_i_spheres)
    
    if SUPPORT_ASSETS_OUTSIDE_CONFIG:
        robots_prim_paths = [robot.get_prim_path() for robot in robots]
        targets_prim_paths = [robot.get_target_prim_path() for robot in robots]
        obstacles_prim_paths = [obstacle.get_prim_path() for obstacle in env_obstacles]
        ignore_prefix = [        
            *robots_prim_paths,
            *targets_prim_paths, # "/World/target",
            # *obstacles_prim_paths,
            "/World/defaultGroundPlane",
            "/curobo"
        ]
        load_klt = True 
        
        if load_klt:
            gravity_klt = False
            klt_position = Gf.Vec3d(0.6, 0, 0.2)
            klt_rotation_euler = Gf.Vec3f(0, 0, 0)
            sim_collision_klt = True

            # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.core.prims/docs/index.html
            klt_prim_path = load_asset_to_prim_path("Props/KLT_Bin/small_KLT_visual.usd")
            klt_prim = stage.GetPrimAtPath(klt_prim_path)
            xform = UsdGeom.XformCommonAPI(klt_prim)
            # Apply translation and rotation
            xform.SetTranslate(klt_position)
            xform.SetRotate(klt_rotation_euler)  # Rotation in degrees (XYZ order)
            if sim_collision_klt:
                # prim = stage.GetPrimAtPath(klt_prim_path)
                # Apply collision APIs
                UsdPhysics.CollisionAPI.Apply(klt_prim)
                PhysxSchema.PhysxCollisionAPI.Apply(klt_prim)
                rigid_api = UsdPhysics.RigidBodyAPI.Apply(klt_prim)  
                # Set the approximation type (using token type)
                attr = klt_prim.CreateAttribute("physxCollision:approximation", Sdf.ValueTypeNames.Token)
                attr.Set("meshSimplification")
                
                if not gravity_klt:
                    rigid_api.CreateKinematicEnabledAttr().Set(True)                
            # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/physics/physics_static_collision.html
            # else: # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/physics/physics_static_collision.html
            #     load_asset_to_prim_path("Props/KLT_Bin/small_KLT_visual.usd")

    # time step index in real world (not simulation) steps. This is the num of completed control steps (actions) in *played* simulation (after play button is pressed)
    t_idx = 0 
    
    # debugging timers
    ctrl_loop_timer = 0
    world_step_timer = 0
    mpc_solver_timer = 0
    targets_update_timer = 0
    joint_state_timer = 0
    action_timer = 0
    robots_as_obs_timer = 0
    env_obstacles_timer = 0
    visualizations_timer = 0

    # main loop
    while simulation_app.is_running():
        point_visualzer_inputs = [] # here we store inputs for draw_points()
        ctrl_loop_timer_start = time.time()
        
        # WORLD STEP
        world_step_timer_start = time.time()                 
        my_world.step(render=True) # print_rate_decorator(lambda: my_world.step(render=True), args.print_ctrl_rate, "my_world.step")() # UPDATE PHYSICS OF SIMULATION AND IF RENDER IS TRUE ALSO UPDATING UI ELEMENTS, VIEWPORTS AND CAMERAS.(Executes one physics step and one rendering step).Note: rendering means rendering a frame of the current application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will be refreshed as well if running non-headless.) See: https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html, see alse https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World       
        world_step_timer += time.time() - world_step_timer_start


        # ENVIRONMENT OBSTACLES - READ STATES AND UPDATE ROBOTS
        env_obstacles_update_timer_start = time.time()
        if SUPPORT_ASSETS_OUTSIDE_CONFIG and (t_idx < ASSET_FIXATION_T or (ASSET_FIXATION_T == -1)): 
            if t_idx % 10 == 0: # less frequent updates because it takes longer to load the assets)
                for i in range(len(robots)):
                    new_world_model:WorldConfig = usd_help.get_obstacles_from_stage(
                        only_paths=["/World"], # only what is under the world prim
                        ignore_substring=ignore_prefix, # expcept these prims (targets, robots, obstacles)
                        reference_prim_path=robots[i].prim_path, # To express the objects in robot's frame (set false to express in world frame)
                    )
                    robots[i].reset_world_model(new_world_model) # replace the current world model with the new one
                    print(f'robot {i} new cchecker: num of obstacles: {len(robots[i].get_world_model().objects)}')

        # obstacles from config
        # update obstacles poses in registed ccheckers (for environment (shared) obstacles) 
        else: # because without this condition, there is a bug
            for i in range(len(env_obstacles)): 
                env_obstacles[i].update_registered_ccheckers()
        env_obstacles_timer += time.time() - env_obstacles_update_timer_start

        # ROBOTS AS OBSTACLES - READ STATES/PLANS
        # get other robots states (no prediction) or plans (with prediction) for collision checking
        robots_as_obs_timer_start = time.time()
        if OBS_PREDICTION:         
            plans = [robots[i].get_plan() for i in range(len(robots))]
        else:
            sphere_states_all_robots:list[torch.Tensor] = [robots[i].get_current_spheres_state()[0] for i in range(len(robots))]
        robots_as_obs_timer += time.time() - robots_as_obs_timer_start

        # ROBOTS AS OBSTACLES - UPDATE STATES/PLANS
        # update robots with other robots as obstacles (robot spheres as obstacles)
        for i in range(len(robots)):
            # ROBOTS AS OBSTACLES - UPDATE STATES/PLANS
            if OBS_PREDICTION and len(col_pred_with[i]): # using prediction of other robots plans
                robots_as_obs_timer_start = time.time()
                p_spheresOthersH = None
                for j in range(len(robots)):
                    if j != i: 
                        planSpheres_robotj = plans[j]['task_space']['spheres'] # robots[j].get_plan(n_steps=robots[i].H)['task_space']['spheres']
                        p_spheresRobotjH = planSpheres_robotj['p'][:robots[i].H].to(tensor_args.device) # get plan (sphere positions) of robot j, up to the horizon length of robot i
                        rad_spheresRobotjH = planSpheres_robotj['r'][0].to(tensor_args.device)
                        if p_spheresOthersH is None:
                            p_spheresOthersH = p_spheresRobotjH
                            rad_spheresOthersH = rad_spheresRobotjH
                        else:
                            p_spheresOthersH = torch.cat((p_spheresOthersH, p_spheresRobotjH), dim=1) # stack the plans horizontally
                            rad_spheresOthersH = torch.cat((rad_spheresOthersH, rad_spheresRobotjH))
                col_pred:DynamicObsCollPredictor = robots[i].dynamic_obs_col_pred
                if t_idx == 0:
                    col_pred.set_obs_rads(rad_spheresOthersH)
                    col_pred.set_own_rads(plans[i]['task_space']['spheres']['r'][0].to(tensor_args.device))
                else:
                    col_pred.update(p_spheresOthersH)
                robots_as_obs_timer += time.time() - robots_as_obs_timer_start

                # if HIGHLIGHT_OBS and t_idx % HIGHLIGHT_OBS_H == 0: # 
                #     if not obs_viz_init:
                #         material = None # glass_material
                #         for h in range(HIGHLIGHT_OBS_H):
                #             for j in range(p_spheresOthersH.shape[1]):
                #                 robots[i].add_obs_viz(p_spheresOthersH[h][j].cpu(),rad_spheresOthersH[j].cpu(),f"o{j}t{h}",h=h,h_max=robots[i].H,material=material)
                #         if i == len(robots) - 1:
                #             obs_viz_init = True
                #     else:
                #         robots[i].update_obs_viz(p_spheresOthersH[:HIGHLIGHT_OBS_H].reshape(-1, 3).cpu()) # collapse first two dimensions
                
                if VISUALIZE_PREDICTED_OBS_PATHS:
                    visualizations_timer_start = time.time()
                    point_visualzer_inputs.append({'points': p_spheresRobotjH, 'color': 'green'})
                    visualizations_timer += time.time() - visualizations_timer_start
            
            elif not OBS_PREDICTION: # using current state of other robots (no prediction)
                robots_as_obs_timer_start = time.time()
                for j in range(len(robots)):    
                    spheres_as_obs = spheres_as_obs_grouped_by_robot[j] 
                    for sphere_idx in range(len(spheres_as_obs)):
                        updated_sphere_pose = np.array(sphere_states_all_robots[j][sphere_idx].tolist() + [1,0,0,0])
                        obs:Obstacle = spheres_as_obs[sphere_idx]
                        obs.update_registered_ccheckers(custom_pose=updated_sphere_pose) 
                robots_as_obs_timer += time.time() - robots_as_obs_timer_start    
            
            # UPDATE STATE IN SOLVER
            joint_state_timer_start = time.time()
            robots_cu_js[i] = robots[i].get_curobo_joint_state(robots[i].get_sim_joint_state())
            robots[i].update_current_state(robots_cu_js[i])    
            joint_state_timer += time.time() - joint_state_timer_start
            
            # UPDATE TARGET IN SOLVER
            targets_update_timer_start = time.time()
            p_T, q_T = robots[i].target.get_world_pose() # print_rate_decorator(lambda: , args.print_ctrl_rate, "target.get_world_pose")() # goal pose
            if robots[i].set_new_target_for_solver(p_T, q_T):
                print(f"robot {i} target changed!")
                robots[i].update_solver_target()
            targets_update_timer += time.time() - targets_update_timer_start

            # MPC STEP
            mpc_solver_timer_start = time.time()
            mpc_result = robots[i].solver.step(robots[i].current_state, max_attempts=2) # print_rate_decorator(lambda: robot1.solver.step(robot1.current_state, max_attempts=2), args.print_ctrl_rate, "mpc.step")()
            mpc_solver_timer += time.time() - mpc_solver_timer_start
            
            # APPLY ACTION
            action_timer_start = time.time()
            art_action = robots[i].get_next_articulation_action(mpc_result.js_action) # get articulated action from joint state action
            robots[i].apply_articulation_action(art_action,num_times=1) # Note: I chhanged it to 1 instead of 3
            action_timer += time.time() - action_timer_start
            
            # VISUALIZATION
            if VISUALIZE_MPC_ROLLOUTS:
                visualizations_timer_start = time.time()
                visual_rollouts = robots[i].solver.get_visual_rollouts()
                visual_rollouts += torch.tensor(robots[i].p_R,device=robots[i].tensor_args.device)
                rollouts_for_visualization = {'points':  visual_rollouts, 'color': 'green'}
                point_visualzer_inputs.append(rollouts_for_visualization)
                visualizations_timer += time.time() - visualizations_timer_start
            
            if VISUALIZE_ROBOT_COL_SPHERES and t_idx % 2 == 0:
                visualizations_timer_start = time.time()
                robots[i].visualize_robot_as_spheres(robots_cu_js[i])
                visualizations_timer += time.time() - visualizations_timer_start

        # VISUALIZATION
        if len(point_visualzer_inputs):
            visualizations_timer_start = time.time()
            draw_points(point_visualzer_inputs) # print_rate_decorator(lambda: draw_points(point_visualzer_inputs), args.print_ctrl_rate, "draw_points")()
            visualizations_timer += time.time() - visualizations_timer_start
        t_idx += 1 # num of completed control steps (actions) in *played* simulation (aft
        ctrl_loop_timer += time.time() - ctrl_loop_timer_start
        
        # PRINT TIME STATISTICS
        k_print = 100
        if t_idx % k_print == 0:    
            print(f"t = {t_idx}")
            print(f"ctrl freq in last {k_print} steps:  {k_print / ctrl_loop_timer}")
            print(f"robots as obs ops freq in last {k_print} steps: {k_print / robots_as_obs_timer}")
            print(f"env obs ops freq in last {k_print} steps: {k_print / env_obstacles_timer}")
            print(f"mpc solver freq in last {k_print} steps: {k_print / mpc_solver_timer}")
            print(f"world step freq in last {k_print} steps: {k_print / world_step_timer}")
            print(f"targets update freq in last {k_print} steps: {k_print / targets_update_timer}")
            print(f"joint states updates freq in last {k_print} steps: {k_print / joint_state_timer}")
            print(f"actions freq in last {k_print} steps: {k_print / action_timer}")
            print(f"visualization ops freq in last {k_print} steps: {k_print / visualizations_timer}")
            
            total_time_measured = mpc_solver_timer + world_step_timer + targets_update_timer + \
            joint_state_timer + action_timer + visualizations_timer + robots_as_obs_timer + env_obstacles_timer
            total_time_actual = ctrl_loop_timer
            delta = total_time_actual - total_time_measured
            print(f"total time actual: {total_time_actual}")
            print(f"total time measured: {total_time_measured}")
            print(f"delta: {delta}")
            print("In percentage %:")
            print(f"mpc solver: {100 * mpc_solver_timer / total_time_actual}")
            print(f"world step: {100 * world_step_timer / total_time_actual}")
            print(f"robots as obs: {100 * robots_as_obs_timer / total_time_actual}")
            print(f"env obs: {100 * env_obstacles_timer / total_time_actual}")
            print(f"targets update: {100 * targets_update_timer / total_time_actual}")
            print(f"joint state: {100 * joint_state_timer / total_time_actual}")
            print(f"action: {100 * action_timer / total_time_actual}")
            print(f"visualizations: {100 * visualizations_timer / total_time_actual}")
            # reset timers
            ctrl_loop_timer = 0
            mpc_solver_timer = 0
            world_step_timer = 0
            targets_update_timer = 0
            joint_state_timer = 0
            action_timer = 0
            visualizations_timer = 0
            robots_as_obs_timer = 0
            env_obstacles_timer = 0
            # print("t = ", t_idx)
            # ctrl_loop_freq = t_idx / (time.time() - ctrl_loop_start_time) 
            # print(f"Control loop frequency [HZ] = {ctrl_loop_freq}")
 
       
if __name__ == "__main__":
    if DEBUG_GPU_MEM:
        signal.signal(signal.SIGINT, handle_sigint_gpu_mem_debug) # register the signal handler for SIGINT (Ctrl+C) 
        torch.cuda.memory._record_memory_history() # https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
    main()
    simulation_app.close()
    
     
        

        
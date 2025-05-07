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
MODIFY_MPC_COST_FN_FOR_DYN_OBS  = True # If True, this would be what the original MPC cost function could handle. False means that the cost will consider obstacles as moving and look into the future, while True means that the cost will consider obstacles as static and not look into the future.
DEBUG = False # If True, then the cost function will be printed on every call to my_world.step()
VISUALIZE_PREDICTED_OBS_PATHS = False # If True, then the predicted paths of the dynamic obstacles will be rendered in the simulation.
VISUALIZE_MPC_ROLLOUTS = True # If True, then the MPC rollouts will be rendered in the simulation.
VISUALIZE_ROBOT_COL_SPHERES = False # If True, then the robot collision spheres will be rendered in the simulation.
HIGHLIGHT_OBS = False # mark the predicted (or not predicted) dynamic obstacles in the simulation
HIGHLIGHT_OBS_H = 30
RENDER_DT = 0.03 # original 1/60
PHYSICS_STEP_DT = 0.03 # original 1/60
MPC_DT = 0.03 # independent of the other dt's, but if you want the mpc to simulate the real step change, set it to be as RENDER_DT and PHYSICS_STEP_DT.
DEBUG_GPU_MEM = True # If True, then the GPU memory usage will be printed on every call to my_world.step()
# NOTE: RENDER_DT and PHYSICS_DT guide from emperical experiments!:
# On every call call to my_world.step():
#  * RULE 1: RENDER_DT controls the average pose change of an object.* 
# If an object has a (constant) linear velocity of V[m/s], then if before the my_world.step() call the object was at position P, then after the my_world.step() call the object will be at position P+V*RENDER_DT on average*. Example: if RENDER_DT = 1/30, then if object is at pose xyz =(1[m],0[m],2[m]) and has a constant linear velocity of (0.15,0,0) [m/s], then after the my_world.step() call the object will be at pose (1+0.15*1/30,0,2)*[m]= (1.005[m],0[m],2[m]) on average(see exact definition below).
# * RULE 2: RENDER_DT/PHYSICS_DT controls the time step index of the simulation: "my_world.current_time_step_index" *
# RENDER_DT/PHYSICS_DT is the number of time steps *on average* that are added to the time step counter of the simulation (my_world.current_time_step_index) on every call to my_world.step(). Example: if before the my_world.step() call the time step counter was 10, and RENDER_DT/PHYSICS_DT = 2, then after the my_world.step() call the time step counter will be 10+2=12. In different words, the simulator takes REDNER_DT/PHYSICS_DT physics steps for every 1 rendering step on every call to my_world.step().
#  * RULE 3: Internal simulation time is determined by my_world.current_time_step_index*PHYSICS_DT.
# The furmula is my_world.current_time = my_world.current_time_step_index*PHYSICS_DT
#  * RULE 4: *
# my_world.current_time_step_index and my_world.current_time not necesarilly updated on every call to my_world.step(), but if they do, they are updated together (meaning that they are synchronized).
# * RULE 5: *
# the call to my_world.step() can perform 0, 1 or more than one physics steps.
# Additional notes:
# - "on average" means that the updated depends on the ratio: PHYSICS_DT/RENDER_DT. For example if the ratio = 4, then the update will be applied only every 4th call to my_world.step(). However if ths ratio is <=1, then the update will be applied every call to my_world.step().
# - For exact APIs see https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=set_simulation_dt and https://docs.omniverse.nvidia.com/isaacsim/latest/simulation_fundamentals.html
# - all info above referse to calls to my_world.step(render=True) (i.e. calls to my_world.step() with rendering=True)

if True: # imports and initiation (put it in if to collapse it)
    try:
        # Third Party
        import isaacsim
    except ImportError:
        pass

    # Third Party
    import time
    import random
    from typing import List, Optional
    from curobo.geom.sdf.world_mesh import WorldMeshCollision
    import torch
    import argparse
    import os
    from typing import Callable, Dict, Union
    import carb
    import numpy as np
    import copy
    from abc import abstractmethod
    import os
    
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # prevent cuda out of memory errors
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()  # Also helps clean up inter-process memory

    # Initialize the simulation app first (must be before "from omni.isaac.core")
    
    from omni.isaac.kit import SimulationApp  

    simulation_app = SimulationApp({"headless": False})

    # Now import other Isaac Sim modules
    # https://medium.com/@kabilankb2003/isaac-sim-core-api-for-robot-control-a-hands-on-guide-f9b27f5729ab
    from omni.isaac.core import World # https://forums.developer.nvidia.com/t/cannot-import-omni-isaac-core/242977/3
    from omni.isaac.core.objects import cuboid, sphere
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.objects import DynamicCuboid
    # from omni.isaac.debug_draw import _debug_draw
    from isaacsim.util.debug_draw import _debug_draw # isaac 4.5
    # from omni.isaac.core.materials import OmniGlass # isaac 4
    from omni.isaac.core.objects import VisualSphere
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.core.utils.types import JointsState as isaac_JointsState
    
    # from omni.isaac.core.prims import XFormPrim
    # from omni.isaac.core.prims import SingleXFormPrim
    import omni.kit.commands as cmd
    from pxr import Gf

    from projects_root.utils.helper import add_extensions, add_robot_to_scene
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.frame_utils import FrameUtils
    from projects_root.autonomous_franka import FrankaMpc

    # CuRobo
    from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
    from curobo.geom.types import Sphere, WorldConfig, Cuboid, Mesh
    from curobo.rollout.rollout_base import Goal
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    # from curobo.types.robot import JointState
    from curobo.types.state import JointState
    from curobo.util.logger import setup_curobo_logger, log_error
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
    from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.types.tensor import T_DOF
    from curobo.types.state import FilterCoeff
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
    from projects_root.utils.decorators import static_vars
    from curobo.wrap.reacher.motion_gen import (MotionGen,MotionGenConfig,MotionGenPlanConfig,PoseCostMetric,)
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig, PoseCostMetric
    # Initialize CUDA device
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" # prevent cuda out of memory errors
    a = torch.zeros(4, device="cuda:0") 

################### Read arguments ########################
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

###########################################################
######################### HELPER ##########################
###########################################################


def draw_points(points_dicts: List[dict], color='green'):
    """
    Visualize points in the simulation.
    _debug_draw docs: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.debug_draw/docs/index.html?highlight=draw_point
    color docs:
        1.  https://docs.omniverse.nvidia.com/kit/docs/kit-manual/106.3.0/carb/carb.ColorRgba.html
        2. rgba (rgb + alpha (transparency)) https://www.w3schools.com/css/css_colors_rgb.asp
    Args:
        points_dicts: List of dictionaries with keys 'points' and 'color'
            points: Tensor of point sequences of shape [num of point-sequences (batch size), num of points per sequence, 3] # 3 is for x,y,z
            color: Color of the points in tensor
    """
    unified_points = []
    unified_colors = []
    unified_sizes = []

    for points_dict in points_dicts:
        rollouts = points_dict['points']
        color = points_dict['color']
        if rollouts is None:
            return
        
        draw = _debug_draw.acquire_debug_draw_interface()
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
            if type(color) == str:
                if color == 'green':
                    colors += [(1.0 - (i + 1.0 / b), 0.3 * (i + 1.0 / b), 0.0, 0.1) for _ in range(h)]
                elif color == 'black':
                    colors += [(0.0, (1.0 - (i + 1.0 / b)), 0.3 * (i + 1.0 / b), 0.5) for _ in range(h)]
            
            elif type(color) == np.ndarray:
                color = list(color) # rgb
                for step_idx in range(h):
                    color_copy = color.copy()
                    color_copy.append(1 - (0.5 * step_idx/h)) # decay alpha (decay transparency)
                    colors.append(color_copy)

        sizes = [10.0 for _ in range(b * h)]

        for p in point_list:
            unified_points.append(p)
        for c in colors:
            unified_colors.append(c)
        for s in sizes:
            unified_sizes.append(s)
    
    draw.draw_points(unified_points, unified_colors, unified_sizes)
        
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

def get_predicted_dynamic_obss_poses_for_visualization(dynamic_obstacles,dynamic_obs_coll_predictor,horizon=30):
    """
    Get the predicted poses of the dynamic obstacles for visualization purposes.
    """
    # predicted_poses = torch.zeros((len(dynamic_obstacles),30, 3))
    obs_for_visualization = [] 
    for obs_index in range(len(dynamic_obstacles)):
        obs_predicted_poses = torch.zeros((1 ,horizon, 3))
        obs_name = dynamic_obstacles[obs_index].name
        predicted_path = dynamic_obs_coll_predictor.get_predicted_path(obs_name)[dynamic_obs_coll_predictor.cur_checker_idx:dynamic_obs_coll_predictor.cur_checker_idx+dynamic_obs_coll_predictor.H]
        predicted_path = predicted_path[:,:3]
        obs_predicted_poses[0] = torch.tensor(predicted_path)
        obs_color = dynamic_obstacles[obs_index].simulation_representation.get_applied_visual_material().get_color()
        obs_for_visualization.append({"points": obs_predicted_poses, "color": obs_color})
    return obs_for_visualization

def print_ctrl_rate_info(t_idx,real_robot_cfm_start_time,real_robot_cfm_start_t_idx,expected_ctrl_freq_at_mpc,step_dt_traj_mpc):
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
                You probably need to change mpc_config.step_dt(step_dt_traj_mpc) from {step_dt_traj_mpc} to {cfm_avg_step_dt})")

def activate_gpu_dynamics(my_world):
    """
    Activates GPU dynamics for the given world.
    """
    my_world_physics_context = my_world.get_physics_context()
    if not my_world_physics_context.is_gpu_dynamics_enabled():
        print("GPU dynamics is disabled. Initializing GPU dynamics...")
        my_world_physics_context.enable_gpu_dynamics(True)
        assert my_world_physics_context.is_gpu_dynamics_enabled()
        print("GPU dynamics is enabled")

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

def wait_for_playing(my_world):
    playing = False
    while simulation_app.is_running() and not playing:
        my_world.step(render=True)
        if my_world.is_playing():
            playing = True
        else:
            if args.autoplay: # if autoplay is enabled, play the simulation immediately
                my_world.play()
                while not my_world.is_playing():
                    print("blocking until playing is confirmed...")
                    time.sleep(0.1)
                playing = True
            else:
                print("Waiting for play button to be pressed...")
                time.sleep(0.1)
    
    my_world.step(render=True)
    my_world.reset()

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


#############################################
# MAIN SIMULATION LOOP
#############################################



def write_stage_to_usd_file(stage,file_path):
    stage.Export(file_path) # export the stage to a temporary USD file
    


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
    setup_curobo_logger("warn")


    
    # world_model = get_world_model_from_current_stage(stage)
    tensor_args = TensorDeviceType()  # Device configuration for tensor operations
    if ENABLE_GPU_DYNAMICS:
        activate_gpu_dynamics(my_world)
    
    # Adding two frankas to the scene
    # # Inspired by curobo/examples/isaac_sim/batch_motion_gen_reacher.py but this time at the same world (the batc)
    
    robots: List[FrankaMpc] = [None, None]
    robots_cu_js: List[Optional[JointState]] =[None, None] # for visualization of robot spheres
    robots_collision_caches = [{"obb": 100, "mesh": 100}, {"obb": 100, "mesh": 100}]
    robot_cfgs = [load_yaml(f"projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/franka{i}.yml")["robot_cfg"] for i in range(1,3)]
    robot_idx_lists:List[Optional[List]] = [None, None]
    X_Robots = [np.array([0,0,0,1,0,0,0], dtype=np.float32), np.array([1.2,0,0,1,0,0,0], dtype=np.float32)] # X_RobotOrigin (x,y,z,qw, qx,qy,qz) (expressed in world frame)
    robot_world_models = [WorldConfig() for _ in range(len(robots))]
    X_binCenter = np.array([0.6, 0, 0.2, 1, 0, 0, 0], dtype=np.float32)
    X_Targets = [[0.6, 0, 0.2, 0, 1, 0, 0], [0.6, 0, 0.2, 0, 1, 0, 0]]


    # X_target = X_binCenter.copy()
    # X_target[3:5] = [0,1] # upside down
    # X_Targets = [X_target.copy(), X_target.copy()] 
    # bin_dim = 0.4 # depends on the cfg file
    # p_infront, p_behind, p_on_left, p_on_right = X_binCenter[:3] + np.array([0,0.75 * bin_dim,bin_dim]),  X_binCenter[:3] + np.array([0,-0.75 * bin_dim,bin_dim]), X_binCenter[:3] + np.array([0.75 * bin_dim,0,bin_dim]), X_binCenter[:3] + np.array([- 0.75 * bin_dim,0,bin_dim])
    # valid_neihborhood = [[p_infront, p_behind, X_binCenter[:3]], [p_infront, p_behind, X_binCenter[:3]]]

    
    collision_obstacles_cfg_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/collision_obstacles.yml"
    col_ob_cfg = load_yaml(collision_obstacles_cfg_path)
    env_obstacles = [] # list of obstacles in the world
    for obstacle in col_ob_cfg:
        obstacle = Obstacle(my_world, **obstacle)
        for i in range(len(robot_world_models)):
            world_model_idx = obstacle.add_to_world_model(robot_world_models[i], X_Robots[i])#  usd_helper=usd_help) # inplace modification of the world model with the obstacle
            print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")
        env_obstacles.append(obstacle) # add the obstacle to the list of obstacles

    # load_asset_to_prim_path('/home/dan/Desktop/small_KLT.usd', '/World/curobo_world_cfg_obs_visual_twins/small_KLT',True)
    # tmp_world_model = read_world_model_from_usd('/home/dan/Desktop/small_KLT.usd',usd_helper=usd_help)
    
    # stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    
    
    robots = [ 
        FrankaMpc(robot_cfgs[0], my_world, usd_help, p_R=X_Robots[0][:3],q_R=X_Robots[0][3:], p_T=X_Targets[0][:3], R_T=X_Targets[0][3:], target_color=np.array([0,0.5,0])) ,
        FrankaMpc(robot_cfgs[1], my_world, usd_help, p_R=X_Robots[1][:3],q_R=X_Robots[1][3:], p_T=X_Targets[1][:3],R_T=X_Targets[1][3:], target_color=np.array([0.5,0,0]))     
    ]    
    
    add_extensions(simulation_app, args.headless_mode)
    
    ################ PRE PLAYING SIM ###################
    if args.print_ctrl_rate and SIMULATING:
        real_robot_cfm_is_initialized, real_robot_cfm_start_t_idx, real_robot_cfm_start_time = None, None, None
    
    if not SIMULATING:
        real_robot_cfm_start_time:float = np.nan # system time when control frequency measurement has started (not yet started if np.nan)
        real_robot_cfm_start_t_idx:int = -1 # actual step index when control frequency measurement has started (not yet started if -1)
        real_robot_cfm_min_start_t_idx:int = 10 # minimal step index allowed to start measuring control frequency. The reason for this is that the first steps are usually not representative of the control frequency (due to the overhead at the times of the first steps which include initialization of the simulation, etc.).

    init_world = False # ugly, do we need this?
    if not init_world:
        for _ in range(10):
            my_world.step(render=True) 
        init_world = True
    wait_for_playing(my_world) # wait for the play button to be pressed
    
    ################# SIM IS PLAYING ###################    
    # Set robots in initial joint configuration (in curobo they call it  the "retract" config)
    # step_dt_traj_mpc = RENDER_DT if SIMULATING else REAL_TIME_EXPECTED_CTRL_DT  
    step_dt_traj_mpc = MPC_DT
    dynamic_obs_coll_predictors:List[DynamicObsCollPredictor] = []
    ccheckers = []
    # expected_ctrl_freq_at_mpc = 1 / step_dt_traj_mpc # This is what the mpc "thinks" the control frequency should be. It uses that to generate the rollouts.                
    obs_viz_init = False
    total_obs_all_robots = sum([robots[i].n_coll_spheres_valid for i in range(len(robots))])

    for i, robot in enumerate(robots):
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot.init_joints(robot_idx_lists[i])
         
        if MODIFY_MPC_COST_FN_FOR_DYN_OBS:
            n_obs_roboti = total_obs_all_robots - robots[i].n_coll_spheres_valid
            col_pred_roboti = DynamicObsCollPredictor(tensor_args, step_dt_traj_mpc,robot.H,robot.num_particles, robot.n_coll_spheres_valid, n_obs_roboti)
            dynamic_obs_coll_predictors.append(col_pred_roboti) # Now if we are modifying the MPC cost function to predict poses of moving obstacles, we need to initialize the mechanism which does it. That's the  DynamicObsCollPredictor() class.
            # p_obs = torch.zeros((robots[i].H, n_obs_roboti, 3), device=robots[i].tensor_args.device)
            # rad_obs = torch.zeros(n_obs_roboti, device=robots[i].tensor_args.device)
            # col_pred_roboti.reset_obs(p_obs, rad_obs)
 
        else:
            dynamic_obs_coll_predictors.append(None)
        
        robots[i].init_solver(robot_world_models[i],robots_collision_caches[i], step_dt_traj_mpc, dynamic_obs_coll_predictors[i], DEBUG)
        checker = robots[i].get_cchecker() # available only after init_solver
        ccheckers.append(checker)
        robots[i].robot._articulation_view.initialize() # new (isac 4.5) https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207

        
    for i in range(len(env_obstacles)):
        env_obstacles[i].register_ccheckers(ccheckers)

    if not MODIFY_MPC_COST_FN_FOR_DYN_OBS: 
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
            
    t_idx = 0 # time step index in real world (not simulation) steps. This is the num of completed control steps (actions) in *played* simulation (after play button is pressed)
    if HIGHLIGHT_OBS:
        glass_material = None # OmniGlass("/World/looks/glass_obsviz", color=np.array([1, 1, 1]),
                             #       ior=1.25, depth=0.001, thin_walled=True,
                                #  ) - ISAAC 4
    ctrl_loop_start_time = time.time()
    while simulation_app.is_running():                 
        
        # Update simulation
        # print("(debug) t_idx= ", t_idx)
        my_world.step(render=True) # print_rate_decorator(lambda: my_world.step(render=True), args.print_ctrl_rate, "my_world.step")() # UPDATE PHYSICS OF SIMULATION AND IF RENDER IS TRUE ALSO UPDATING UI ELEMENTS, VIEWPORTS AND CAMERAS.(Executes one physics step and one rendering step).Note: rendering means rendering a frame of the current application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will be refreshed as well if running non-headless.) See: https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html, see alse https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World       
        
        # Measure control frequency
        if args.print_ctrl_rate and not SIMULATING:
            real_robot_cfm_is_initialized = not np.isnan(real_robot_cfm_start_time) # is the control frequency measurement already initialized?
            real_robot_cfm_can_be_initialized = t_idx > real_robot_cfm_min_start_t_idx # is it valid to start measuring control frequency now?
            if not real_robot_cfm_is_initialized and real_robot_cfm_can_be_initialized:
                real_robot_cfm_start_time = time.time()
                real_robot_cfm_start_t_idx = t_idx # my_world.current_time_step_index is "t", current time step. Num of *completed* control steps (actions) in *played* simulation (after play button is pressed)
        
        # update obstacles poses in registed ccheckers (for environment (shared) obstacles)
        for i in range(len(env_obstacles)): 
            env_obstacles[i].update_registered_ccheckers()
               
        if MODIFY_MPC_COST_FN_FOR_DYN_OBS:         
            plans = [robots[i].get_plan() for i in range(len(robots))]
        else:
            sphere_states_all_robots:list[torch.Tensor] = [robots[i].get_current_spheres_state()[0] for i in range(len(robots))]

        point_visualzer_inputs = []

        for i in range(len(robots)):
            # update obstacles (robot spheres as obstacles)
            if MODIFY_MPC_COST_FN_FOR_DYN_OBS:
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
                
                if t_idx == 0:
                    dynamic_obs_coll_predictors[i].activate(p_spheresOthersH, rad_spheresOthersH)
                else:
                    dynamic_obs_coll_predictors[i].update(p_spheresOthersH)
                
                if HIGHLIGHT_OBS and t_idx % HIGHLIGHT_OBS_H == 0: # 
                    if not obs_viz_init:
                        material = glass_material
                        for h in range(HIGHLIGHT_OBS_H):
                            for j in range(p_spheresOthersH.shape[1]):
                                robots[i].add_obs_viz(p_spheresOthersH[h][j].cpu(),rad_spheresOthersH[j].cpu(),f"o{j}t{h}",h=h,h_max=robots[i].H,material=material)
                        if i == len(robots) - 1:
                            obs_viz_init = True
                    else:
                        robots[i].update_obs_viz(p_spheresOthersH[:HIGHLIGHT_OBS_H].reshape(-1, 3).cpu()) # collapse first two dimensions
                
                if VISUALIZE_PREDICTED_OBS_PATHS:
                    point_visualzer_inputs.append({'points': p_spheresRobotjH, 'color': 'green'})
            else:
                for i in range(len(robots)):    
                    spheres_as_obs_i = spheres_as_obs_grouped_by_robot[i] 
                    for sphere_idx in range(len(spheres_as_obs_i)):
                        updated_sphere_pose = np.array(sphere_states_all_robots[i][sphere_idx].tolist() + [1,0,0,0])
                        obs:Obstacle = spheres_as_obs_i[sphere_idx]
                        obs.update_registered_ccheckers(custom_pose=updated_sphere_pose) 
                    
            # update robot state and target
            robots_cu_js[i] = robots[i].get_curobo_joint_state(robots[i].get_sim_joint_state())
            robots[i].update_current_state(robots_cu_js[i])    
            p_T, q_T = robots[i].target.get_world_pose() # print_rate_decorator(lambda: , args.print_ctrl_rate, "target.get_world_pose")() # goal pose
            if robots[i].set_new_target_for_solver(p_T, q_T):
                print(f"robot {i} target changed!")
                robots[i].update_solver_target()
            
            # mpc step
            mpc_result = robots[i].solver.step(robots[i].current_state, max_attempts=2) # print_rate_decorator(lambda: robot1.solver.step(robot1.current_state, max_attempts=2), args.print_ctrl_rate, "mpc.step")()
            art_action = robots[i].get_next_articulation_action(mpc_result.js_action) # get articulated action from joint state action
            robots[i].apply_articulation_action(art_action,num_times=1) # Note: I chhanged it to 1 instead of 3
            
            # visualization
            if VISUALIZE_MPC_ROLLOUTS:
                rollouts_for_visualization = {'points': robots[i].solver.get_visual_rollouts(), 'color': 'green'}
                point_visualzer_inputs.append(rollouts_for_visualization)
            
            if VISUALIZE_ROBOT_COL_SPHERES and t_idx % 2 == 0:
                robots[i].visualize_robot_as_spheres(robots_cu_js[i])

            
        if len(point_visualzer_inputs):
            draw_points(point_visualzer_inputs) # print_rate_decorator(l



        t_idx += 1 # num of completed control steps (actions) in *played* simulation (aft
        
        if t_idx % 100 == 0:
            print("t = ", t_idx)
            ctrl_loop_freq = t_idx / (time.time() - ctrl_loop_start_time) 
            print(f"Control loop frequency [HZ] = {ctrl_loop_freq}")
 
       
        
    

import signal
import sys

def handle_sigint_gpu_mem_debug(signum, frame):
    print("Caught SIGINT (Ctrl+C), first dump snapshot")
    torch.cuda.memory._dump_snapshot()
    # now upload the snapshot dump_snapshot.pickle to the server: https://docs.pytorch.org/memory_viz
    raise KeyboardInterrupt # to let the original KeyboardInterrupt handler (of nvidia) to close the app
    


        
        
if __name__ == "__main__":
    if DEBUG_GPU_MEM:
        signal.signal(signal.SIGINT, handle_sigint_gpu_mem_debug)
        torch.cuda.memory._record_memory_history() # https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
    main()
    simulation_app.close()
    
     
        

        
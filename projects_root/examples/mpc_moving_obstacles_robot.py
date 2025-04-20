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
    - https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World:~:text=.clear_render_callbacks()-,World,-%23
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
DEBUG_COST_FUNCTION = True # If True, then the cost function will be printed on every call to my_world.step()
VISUALIZE_PREDICTED_OBS_PATHS = True # If True, then the predicted paths of the dynamic obstacles will be rendered in the simulation.
VISUALIZE_MPC_ROLLOUTS = True # If True, then the MPC rollouts will be rendered in the simulation.
VISUALIZE_ROBOT_COL_SPHERES = False # If True, then the robot collision spheres will be rendered in the simulation.
HIGHLIGHT_OBS = False # mark the predicted (or not predicted) dynamic obstacles in the simulation
RENDER_DT = 0.03 # original 1/60
PHYSICS_STEP_DT = 0.03 # original 1/60
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
    from typing import List, Optional
    from curobo.geom.sdf.world_mesh import WorldMeshCollision
    import torch
    import argparse
    import os
    import carb
    import numpy as np
    import copy
    from abc import abstractmethod

    # Initialize the simulation app first (must be before "from omni.isaac.core")

    from omni.isaac.kit import SimulationApp  
    simulation_app = SimulationApp({"headless": False})

    # Now import other Isaac Sim modules
    # https://medium.com/@kabilankb2003/isaac-sim-core-api-for-robot-control-a-hands-on-guide-f9b27f5729ab
    from omni.isaac.core import World # https://forums.developer.nvidia.com/t/cannot-import-omni-isaac-core/242977/3
    from omni.isaac.core.objects import cuboid, sphere
    from omni.isaac.core.utils.types import ArticulationAction
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.debug_draw import _debug_draw
    from omni.isaac.core.materials import OmniGlass
    from omni.isaac.core.objects import VisualSphere

    # Import helper from curobo examples

    from projects_root.utils.helper import add_extensions, add_robot_to_scene

    # CuRobo
    from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
    from curobo.geom.types import Sphere, WorldConfig, Cuboid
    from curobo.rollout.rollout_base import Goal
    from curobo.types.base import TensorDeviceType
    from curobo.types.math import Pose
    from curobo.types.robot import JointState
    from curobo.types.state import JointState
    from curobo.util.logger import setup_curobo_logger, log_error
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
    from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    from curobo.types.robot import RobotConfig
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
    from projects_root.utils.decorators import static_vars
    from curobo.wrap.reacher.motion_gen import (MotionGen,MotionGenConfig,MotionGenPlanConfig,PoseCostMetric,)

    # Initialize CUDA device
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
    default="True",
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
def spawn_target(path="/World/target", position=np.array([0.5, 0.0, 0.5]), orientation=np.array([0, 1, 0, 0]), color=np.array([0, 1, 0]), size=0.05):
    """ 
    Create a target pose "hologram" in the simulation. By "hologram", 
    we mean a visual representation of the target pose that is not used for collision detection or physics calculations.
    In isaac-sim they call holograms viual objects (like visual coboid and visual spheres...)
    """
    target = cuboid.VisualCuboid(
        path,
        position=position,
        orientation=orientation,
        color=color,
        size=size,
    )
    
    return target

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

class AutonomousFranka:
    
    instance_counter = 0
    def __init__(self,robot_cfg, world, usd_help, p_R=np.array([0.0,0.0,0.0]),R_R=np.array([1,0,0,0]), p_T=np.array([0.5,0.0,0.5]), R_T=np.array([0.0,1.0,0.0,0.0]), target_color=np.array([0.0,1.0,0.0]), target_size=0.05):
        """
        Spawns a franka robot in the scene andd setting the target for the robot to follow.
        All notations will follow Drake. See: https://drake.mit.edu/doxygen_cxx/group__multibody__quantities.html#:~:text=Typeset-,Monogram,-Meaning%E1%B5%83
        https://drake.mit.edu/doxygen_cxx/group__multibody__notation__basics.html https://drake.mit.edu/doxygen_cxx/group__multibody__frames__and__bodies.html
        Args:
            world (_type_): simulator world instance.
            p_R: Position vector from Wo (W (World frame) origin (o)) to Ro (R's origin (robot's base frame)), expressed in the world frame W (implied).
            R_R: Frame R's (second R, representing robot's base frame) orientation (first R, representing rotation) in the world frame W (implied, could be represented as R_WR). Quaternion (w,x,y,z)
            p_T: Position vector from Wo (W (World frame) origin (o)) to To (target's origin), expressed in the world frame W (implied).
            R_T: Frame T's (representing target's base frame) orientation (R, representing rotation) in the world frame W (implied, could be represented as R_WT). Quaternion (w,x,y,z)

        """
        # simulator paths etc.
        self.instance_id = AutonomousFranka.instance_counter
        self.world = world
        self.world_root ='/World'
        self.robot_name = f'robot_{self.instance_id}'
        self.subroot_path = f'{self.world_root}/world_{self.instance_id}' # f'{self.world_root}/world_{self.robot_name}'
        
        # robot base frame settings (static, since its an arm and not a mobile robot. Won't change)
        self.p_R = p_R  
        self.R_R = R_R 
        
        # target settings
        self.initial_p_T = p_T # initial target frame position (expressed in the world frame W)
        self.initial_R_T = R_T # initial target frame rotation (expressed in the world frame W)
        self.initial_target_color = target_color
        self.initial_target_size = target_size
        self.target_path = self.subroot_path + '/target' # f'/World/targets/{self.robot_name}'
        self.target_last_synced_position = None # current target position of robot as set to the planner
        self.target_last_synced_orientation = None # current target orientation of robot as set to the planner

        self.robot_cfg = robot_cfg # the section under the key 'robot_cfg' in the robot config file (yml). https://curobo.org/tutorials/1_robot_configuration.html#tut-robot-configuration
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"] # joint names for the robot
        self.initial_joint_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"] # initial ("/retract") joint configuration for the robot
        
        self.cu_stat_obs_world_model = self._init_curobo_stat_obs_world_model(usd_help) # will be initialized in the _init_curobo_stat_obs_world_model method. Static obstacles world configuration for curobo collision checking.
        self.solver = None # will be initialized in the init_solver method.
        self.tensor_args = TensorDeviceType()
        self._vis_spheres = None # for visualization of robot spheres
        self.crm = CudaRobotModel(CudaRobotModelConfig.from_data_dict(self.robot_cfg)) # https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_model.html#curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig
        self.obs_viz = [] # for visualization of robot spheres
        self.obs_viz_obs_names = []
        self.obs_viz_prim_path = f'/obstacles/{self.robot_name}'
        AutonomousFranka.instance_counter += 1

    def _init_curobo_stat_obs_world_model(self, usd_help:UsdHelper):
        """Initiating curobo world configuration for static obstacles.
        # NOTE: In other files its initialized a bit differently every time. So thats not the only way to do it.
        
        Args:
            usd_help (UsdHelper): _description_
        """

        world_cfg_table = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "collision_table.yml")))
        world_cfg_table.cuboid[0].pose[2] -= 0.04  # Adjust table height
        world_cfg1 = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))).get_mesh_world()
        world_cfg1.mesh[0].name += "_mesh"
        world_cfg1.mesh[0].pose[2] = -10.5  # Place mesh below ground
        
        world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh) # representation of the world for use in curobo
        usd_help.add_world_to_stage(world_cfg, base_frame=self.subroot_path) 
        return world_cfg
        # self.cu_stat_obs_world_model = world_cfg

        
    def _spawn_robot_and_target(self, usd_help:UsdHelper):
        X_R = Pose.from_list(list(self.p_R) + list(self.R_R)) # initial pose (X) of robot's base frame (R) (expressed implicitly in the world frame (W))
        
        # spawn the robot in the scene in the initial pose X_R
        usd_help.add_subroot(self.world_root, self.subroot_path, X_R)
        self.robot, self.robot_prim_path = add_robot_to_scene(self.robot_cfg, self.world, self.subroot_path+'/', robot_name=self.robot_name, position=self.p_R)

        # spawn the target in the scene in the initial pose (X_T = p_T, R_T)
        self.target = spawn_target(self.target_path, self.initial_p_T, self.initial_R_T, self.initial_target_color, self.initial_target_size)
        
        # Load world configuration for collision checking
        

    @abstractmethod
    def _check_prerequisites_for_syncing_target_pose(self, real_target_position:np.ndarray, real_target_orientation:np.ndarray,sim_js:None) -> bool:
        pass

    def update_target_if_needed(self, real_target_position:np.ndarray, real_target_orientation:np.ndarray,sim_js=None):
        """ Syncs (sets) the target pose of the robot with the target pose of the target.
        real_target_position: target position in world frame
        real_target_orientation: target orientation in world frame
        real_target_position_prev_ts: target position in previous time step
        real_target_orientation_prev_ts: target orientation in previous time step
        sim_js: joint state of the robot (in simulation representation, not curobo representation)

        Returns:
            _type_: _description_
        """
        
        if self.target_last_synced_position is None:
            self.target_last_synced_position = real_target_position + 1.0 # to force the first sync
            self.target_last_synced_orientation = real_target_orientation
            
        sync_target = self._check_prerequisites_for_syncing_target_pose(real_target_position, real_target_orientation, sim_js)

        if sync_target:
            self.target_last_synced_position = real_target_position
            self.target_last_synced_orientation = real_target_orientation
            return True
        
        else:
            return False
    
    def get_last_synced_target_pose(self):
        return Pose(position=self.tensor_args.to_device(self.target_last_synced_position),quaternion=self.tensor_args.to_device(self.target_last_synced_orientation),)
            
    def _post_init_solver(self):
        return None

    @abstractmethod
    def init_solver(self, collision_cache, step_dt_traj_mpc, dynamic_obs_coll_predictor):
        pass

    def _check_target_pose_changed(self, real_target_position, real_target_orientation) -> bool:
        return np.linalg.norm(real_target_position - self.target_last_synced_position) > 1e-3 or np.linalg.norm(real_target_orientation - self.target_last_synced_orientation) > 1e-3

    def _check_target_pose_static(self, real_target_position, real_target_orientation) -> bool:
        if not hasattr(self, '_real_target_pos_prev_t'):
            self._real_target_pos_prev_t = real_target_position
            self._real_target_orient_prev_t = real_target_orientation
        
        is_static = np.linalg.norm(real_target_position - self._real_target_pos_prev_t) == 0.0 and np.linalg.norm(real_target_orientation - self._real_target_orient_prev_t) == 0.0
        self._real_target_pos_prev_t = real_target_position
        self._real_target_orient_prev_t = real_target_orientation
        return is_static
    
    def update_real_target(self, real_target_position, real_target_orientation):
        self.target = spawn_target(self.target_path, real_target_position, real_target_orientation, self.initial_target_color, self.initial_target_size)
        
    def _check_robot_static(self, sim_js) -> bool:
        return np.max(np.abs(sim_js.velocities)) < 0.2
    
    def init_joints(self, idx_list:list):
        """Set the maximum efforts for the robot.
        Args:
          
        """
        # robot.robot._articulation_view.initialize()
        self.robot.set_joint_positions(self.initial_joint_config, idx_list) 
        self.robot._articulation_view.set_max_efforts(values=np.array([5000 for _ in range(len(idx_list))]), joint_indices=idx_list)

    def get_robot_as_spheres(self, cu_js, express_in_world_frame=False) -> list[Sphere]:
        """Get the robot as spheres from the curobot joints state.
        # NOTE: spheres are expressed in the robot base frame and not in the world frame. Shifting to the world frame requires adding the robot base frame position to the sphere position.
        Args:
            cu_js (_type_): curobo joints state
        Returns:
            list[Sphere]: list of spheres
        """
        assert isinstance(self.solver, MpcSolver) or isinstance(self.solver, MotionGen), "Solver not initialized"
        sph_list = self.solver.kinematics.get_robot_as_spheres(cu_js.position)[0] # at this point each sph.position is expressed in the robot base frame, not in the world frame
        if express_in_world_frame:
            for sph in sph_list:
                sph.position = sph.position + self.p_R # express the spheres in the world frame
                sph.pose[:3] = sph.pose[:3] + self.p_R
        return sph_list

    def visualize_robot_as_spheres(self, cu_js):
        if cu_js is None:
            return
        sph_list = self.get_robot_as_spheres(cu_js, express_in_world_frame=True)
        if self._vis_spheres is None: # init visualization spheres
            self._vis_spheres = []
            for si, s in enumerate(sph_list):
                sp = sphere.VisualSphere(
                            prim_path=f"/curobo/robot_{self.instance_id}_sphere_" + str(si),
                            position=np.ravel(s.position),
                            radius=float(s.radius),
                            color=np.array([0, 0.8, 0.2]),
                        )
                self._vis_spheres.append(sp)

        else: # update visualization spheres
            for si, s in enumerate(sph_list):
                if not np.isnan(s.position[0]):
                    self._vis_spheres[si].set_world_pose(position=np.ravel(s.position))
                    self._vis_spheres[si].set_radius(float(s.radius))

    def update_obs_viz(self,p_spheres:torch.Tensor):
        for i in range(len(self.obs_viz)):
            self.obs_viz[i].set_world_pose(position=np.array(p_spheres[i].tolist()), orientation=np.array([1., 0., 0., 0.]))

    def add_obs_viz(self,p_sphere:torch.Tensor,rad_sphere:torch.Tensor, obs_name:str,h=0,h_max=30):
        
        obs_viz = VisualSphere(
            prim_path=f"{self.obs_viz_prim_path}/{obs_name}",
            position=np.ravel(p_sphere),
            radius=float(rad_sphere),
            color=np.array([1-(h/h_max),0,0]))
        self.obs_viz.append(obs_viz)

    def get_dof_names(self):
        return self.robot.dof_names

    def get_sim_joint_state(self):
        return self.robot.get_joints_state()
    
    def get_curobo_joint_state(self, sim_js, zero_vel:bool) -> JointState:
        """Returns the curobo joint configuration (robot joint state represented as a JointState object,
        which is curobo's representation of the robot joint state) from the simulation joint state (the joint state of the robot as returned by the simulation).
        For more details about JointState see https://curobo.org/advanced_examples/4_robot_segmentation.html
        Args:
            sim_js (_type_): the joint state of the robot as returned by the simulation.
            zero_vel (bool): should multiply the velocities by 0.0 to set them to zero (differs for robot types - MPC and Cumotion).

        Returns:
            JointState: the robot’s joint configuration in curobo's representation.
        """
        if sim_js is None:
            sim_js = self.get_sim_joint_state()
        position = self.tensor_args.to_device(sim_js.positions)
        velocity = self.tensor_args.to_device(sim_js.velocities) * 0.0 if zero_vel else self.tensor_args.to_device(sim_js.velocities)
        acceleration = self.tensor_args.to_device(sim_js.velocities) * 0.0
        jerk = self.tensor_args.to_device(sim_js.velocities) * 0.0
        cu_js = JointState(position=position,velocity=velocity,acceleration=acceleration,jerk=jerk,joint_names=self.get_dof_names()) # joint_names=self.robot.dof_names) 
        return cu_js
    

    def get_current_spheres_state(self,express_in_world_frame:bool=True, valid_only=True,zero_vel=False):
        cu_js = self.get_curobo_joint_state(self.get_sim_joint_state(),zero_vel=zero_vel) # zero vel doesent matter since we are getting sphere poses and radii
        link_spheres_R = self.crm.compute_kinematics_from_joint_state(cu_js).get_link_spheres()
        p_link_spheres_R = link_spheres_R[:,:,:3].cpu() # position of spheres expressedin robot base frame
        if express_in_world_frame:
            p_link_spheres_W = p_link_spheres_R + self.p_R
            p_link_spheres_F = p_link_spheres_W
        else:
            p_link_spheres_F = p_link_spheres_R
        p_link_spheres_F = p_link_spheres_F.squeeze(0)
        rad_link_spheres = link_spheres_R[:,:,3].cpu().squeeze(0)
        
        if valid_only:
            sphere_indices = torch.nonzero(rad_link_spheres > 0, as_tuple=True)[0] # valid is positive radius
            p_link_spheres_F = p_link_spheres_F[sphere_indices]
            rad_link_spheres = rad_link_spheres[sphere_indices]
        else:
            sphere_indices = torch.arange(p_link_spheres_F.shape[0]) # all sphere indices

        return p_link_spheres_F, rad_link_spheres, sphere_indices
        
    @abstractmethod
    def apply_articulation_action(self, art_action:ArticulationAction):
        pass
    
class FrankaMpc(AutonomousFranka):
    def __init__(self, robot_cfg, world,usd_help:UsdHelper, p_R=np.array([0.0,0.0,0.0]), R_R=np.array([1,0,0,0]), p_T=np.array([0.5, 0.0, 0.5]), R_T=np.array([0, 1, 0, 0]), target_color=np.array([0, 0.5, 0]), target_size=0.05):
        """
        Spawns a franka robot in the scene andd setting the target for the robot to follow.

        Args:
            world (_type_): _description_
            robot_name (_type_): _description_
            p_R (_type_): _description_
        """
        super().__init__(robot_cfg, world, usd_help, p_R, R_R, p_T, R_T, target_color, target_size)
        self.robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02  # Add safety margin
        self._spawn_robot_and_target(usd_help)
        self.articulation_controller = self.robot.get_articulation_controller()
        self._cmd_state_full = None


    def init_solver(self, collision_cache, step_dt_traj_mpc, dynamic_obs_coll_predictor):
        """Initialize the MPC solver.

        Args:
            world_cfg (_type_): _description_
            collision_cache (_type_): _description_
            step_dt_traj_mpc (_type_): _description_
            dynamic_obs_coll_predictor (_type_): _description_
        """
        dynamic_obs_checker = dynamic_obs_coll_predictor # New
        override_particle_file = 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc.yml' # New. settings in the file will overide the default settings in the default particle_mpc.yml file. For example, num of optimization steps per time step.
        
        mpc_config = MpcSolverConfig.load_from_robot_config(
            self.robot_cfg, #  Robot configuration. Can be a path to a YAML file or a dictionary or an instance of RobotConfig https://curobo.org/_api/curobo.types.robot.html#curobo.types.robot.RobotConfig
            self.cu_stat_obs_world_model, #  World configuration. Can be a path to a YAML file or a dictionary or an instance of WorldConfig. https://curobo.org/_api/curobo.geom.types.html#curobo.geom.types.WorldConfig
            use_cuda_graph= not DEBUG_COST_FUNCTION, # Use CUDA graph for the optimization step. If you want to set breakpoints in the cost function, set this to False.
            use_cuda_graph_metrics=True, # Use CUDA graph for computing metrics.
            use_cuda_graph_full_step=False, #  Capture full step in MPC as a single CUDA graph. This is experimental and might not work reliably.
            self_collision_check=True, # Enable self-collision check during MPC optimization.
            collision_checker_type=CollisionCheckerType.MESH, # type of collision checker to use. See https://curobo.org/get_started/2c_world_collision.html#world-collision 
            collision_cache=collision_cache,
            use_mppi=True,  # Use Model Predictive Path Integral for optimization
            use_lbfgs=False, # Use L-BFGS solver for MPC. Highly experimental.
            use_es=False, # Use Evolution Strategies (ES) solver for MPC. Highly experimental.
            store_rollouts=True,  # Store trajectories for visualization
            step_dt=step_dt_traj_mpc,  # NOTE: Important! step_dt is the time step to use between each step in the trajectory. If None, the default time step from the configuration~(particle_mpc.yml or gradient_mpc.yml) is used. This dt should match the control frequency at which you are sending commands to the robot. This dt should also be greater than the compute time for a single step. For more info see https://curobo.org/_api/curobo.wrap.reacher.solver.html
            dynamic_obs_checker=dynamic_obs_checker, # New
            override_particle_file=override_particle_file # New
        )
        
        self.solver = MpcSolver(mpc_config)
        self.H = self.solver.rollout_fn.horizon
        self._post_init_solver()
        retract_cfg = self.solver.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        joint_names = self.solver.rollout_fn.joint_names
        state = self.solver.rollout_fn.compute_kinematics(JointState.from_position(retract_cfg, joint_names=joint_names))
        self.current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
        retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        
        # Set up goal pose (target position and orientation)
        goal_mpc = Goal(current_state=self.current_state, goal_state=JointState.from_position(retract_cfg, joint_names=joint_names), goal_pose=retract_pose,)

        # Initialize MPC solver with goal
        goal_buffer = self.solver.setup_solve_single(goal_mpc, 1)
        self.goal_buffer = goal_buffer
        self.solver.update_goal(self.goal_buffer)
        mpc_result = self.solver.step(self.current_state, max_attempts=2)
    
    def _check_prerequisites_for_syncing_target_pose(self, real_target_position:np.ndarray, real_target_orientation:np.ndarray,sim_js:None) -> bool:
        has_target_pose_changed = self._check_target_pose_changed(real_target_position, real_target_orientation)
        return has_target_pose_changed

    def get_curobo_joint_state(self, sim_js=None,zero_vel=True) -> JointState:
        """See super().get_curobo_joint_state() for more details.
        """
        cu_js =  super().get_curobo_joint_state (sim_js, zero_vel)        
        cu_js = cu_js.get_ordered_joint_state(self.solver.rollout_fn.joint_names)
        return cu_js
    

    def update_current_state(self, cu_js):
        if self._cmd_state_full is None:
            self.current_state.copy_(cu_js)
        else:
            current_state_partial = self._cmd_state_full.get_ordered_joint_state(
                self.solver.rollout_fn.joint_names
            )
            self.current_state.copy_(current_state_partial)
            self.current_state.joint_names = current_state_partial.joint_names
        self.current_state.copy_(cu_js)


    def get_next_articulation_action(self, js_action):
        """Get articulated action from joint state action (supplied by MPC solver).
        Args:
            js_action (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._cmd_state_full = js_action
        idx_list = []
        common_js_names = []
        for x in self.get_dof_names():
            if x in self._cmd_state_full.joint_names:
                idx_list.append(self.robot.get_dof_index(x))
                common_js_names.append(x)
        self._cmd_state_full = self._cmd_state_full.get_ordered_joint_state(common_js_names)
        art_action = ArticulationAction(self._cmd_state_full.position.cpu().numpy(),joint_indices=idx_list,)
        return art_action
    

    def apply_articulation_action(self, art_action: ArticulationAction,num_times:int=3):
        for _ in range(num_times):
            ans = self.articulation_controller.apply_action(art_action)
        return ans
class FrankaCumotion(AutonomousFranka):
    def __init__(self, robot_cfg, world,usd_help:UsdHelper, p_R=np.array([0.0,0.0,0.0]), R_R=np.array([1,0,0,0]), p_T=np.array([0.5, 0.0, 0.5]), R_T=np.array([0, 1, 0, 0]), target_color=np.array([0, 0.5, 0]), target_size=0.05, reactive=False ):
        """
        Spawns a franka robot in the scene andd setting the target for the robot to follow.

        Args:
            world (_type_): _description_
            robot_name (_type_): _description_
            p_R (_type_): _description_
            reactive (bool, optional): _description_. Defaults to False.
        """
        super().__init__(robot_cfg, world, usd_help, p_R, R_R, p_T, R_T, target_color, target_size)

        self.solver = None
        self.past_cmd:JointState = None
        self.reactive = reactive
        self.num_targets = 0 # the number of the targets which are defined by curobo (after being static and ready to plan to) and have a successfull a plan for.
        self.max_attempts = 4 if not self.reactive else 1
        self.enable_finetune_trajopt = True if not self.reactive else False
        self.pose_metric = None
        self.constrain_grasp_approach = False
        self.reach_partial_pose = None
        self.hold_partial_pose = None
        self.cmd_plan = None
        self.cmd_idx = 0
        self.n_coll_spheres = 65
        self._spawn_robot_and_target(usd_help)
        self.articulation_controller = self.robot.get_articulation_controller()
        
    def apply_articulation_action(self, art_action: ArticulationAction):
        self.cmd_idx += 1
        if self.cmd_idx >= len(self.cmd_plan.position): # NOTE: all cmd_plans (global plans) are at the same length from my observations (currently 61), no matter how many time steps (step_indexes) take to complete the plan.
            self.cmd_idx = 0
            self.cmd_plan = None
            self.past_cmd = None
        return self.articulation_controller.apply_action(art_action)

    
    def init_solver(self, collision_cache,tensor_args):
        """Initialize the motion generator (cumotion global planner).

        Args:
            world_cfg (_type_): _description_
            collision_cache (_type_): _description_
            tensor_args (_type_): _description_
        """
        
        trajopt_dt = None
        optimize_dt = True
        trajopt_tsteps = 32
        trim_steps = None
        
        interpolation_dt = 0.05
        if self.reactive:
            trajopt_tsteps = 40
            trajopt_dt = 0.04
            optimize_dt = False
            trim_steps = [1, None]
            interpolation_dt = trajopt_dt
        # See very good explainations for all the paramerts here: https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenConfig
        motion_gen_config = MotionGenConfig.load_from_robot_config( # solver config
            self.robot_cfg, # robot_cfg – Robot configuration to use for motion generation. This can be a path to a yaml file, a dictionary, or an instance of RobotConfig. See Supported Robots for a list of available robots. You can also create a a configuration file for your robot using Configuring a New Robot.
            self.cu_stat_obs_world_model, # world_model – World configuration to use for motion generation. This can be a path to a yaml file, a dictionary, or an instance of WorldConfig. See Collision World Representation for more details.
            tensor_args, # tensor_args - Numerical precision and compute device to use for motion generation
            collision_checker_type=CollisionCheckerType.MESH, # collision_checker_type – Type of collision checker to use for motion generation. Default of CollisionCheckerType.MESH supports world represented by Cuboids and Meshes. See Collision World Representation for more details.
            num_trajopt_seeds=12, # num_trajopt_seeds – Number of seeds to use for trajectory optimization per problem query. Default of 4 is found to be a good number for most cases. Increasing this will increase memory usage.
            num_graph_seeds=12, # num_graph_seeds – Number of seeds to use for graph planner per problem query. When graph planning is used to generate seeds for trajectory optimization, graph planner will attempt to find collision-free paths from the start state to the many inverse kinematics solutions.
            interpolation_dt=interpolation_dt, # interpolation_dt – Time step in seconds to use for generating interpolated trajectory from optimized trajectory. Change this if you want to generate a trajectory with a fixed timestep between waypoints.
            collision_cache=collision_cache, # collision_cache – Cache of obstacles to create to load obstacles between planning calls. An example: {"obb": 10, "mesh": 10}, to create a cache of 10 cuboids and 10 meshes.
            optimize_dt=optimize_dt, # optimize_dt – Optimize dt during trajectory optimization. Default of True is recommended to find time-optimal trajectories. Setting this to False will use the provided trajopt_dt for trajectory optimization. Setting to False is required when optimizing from a non-static start state.
            trajopt_dt=trajopt_dt, # trajopt_dt – Time step in seconds to use for trajectory optimization. A good value to start with is 0.15 seconds. This value is used to compute velocity, acceleration, and jerk values for waypoints through finite difference.
            trajopt_tsteps=trajopt_tsteps, # trajopt_tsteps – Number of waypoints to use for trajectory optimization. Default of 32 is found to be a good number for most cases.
            trim_steps=trim_steps, # trim_steps – Trim waypoints from optimized trajectory. The optimized trajectory will contain the start state at index 0 and have the last two waypoints be the same as T-2 as trajectory optimization implicitly optimizes for zero acceleration and velocity at the last waypoint. An example: [1,-2] will trim the first waypoint and last 3 waypoints from the optimized trajectory.
        )
        self.solver = MotionGen(motion_gen_config)
        if not self.reactive:
            print("warming up...")
            self.solver.warmup(enable_graph=True, warmup_js_trajopt=False)
        
        print("Curobo is Ready")

    def init_plan_config(self):
        """Initialize the plan config for the motion generator.
        See all the documentation here: https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenPlanConfig
        """
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False, # Use graph planner to generate collision-free seed for trajectory optimization.
            enable_graph_attempt=2, # Number of failed attempts at which to fallback to a graph planner for obtaining trajectory seeds.
            max_attempts=self.max_attempts, # Maximum number of attempts allowed to solve the motion generation problem.
            enable_finetune_trajopt=self.enable_finetune_trajopt, # Run finetuning trajectory optimization after running 100 iterations of trajectory optimization. This will provide shorter and smoother trajectories. When MotionGenConfig.optimize_dt is True, this flag will also scale the trajectory optimization by a new dt. Leave this to True for most cases. If you are not interested in finding time-optimal solutions and only want to use motion generation as a feasibility check, set this to False. Note that when set to False, the resulting trajectory is only guaranteed to be collision-free and within joint limits. When False, it’s not guaranteed to be smooth and might not execute on a real robot.
            time_dilation_factor=0.5 if not self.reactive else 1.0, # Slow down optimized trajectory by re-timing with a dilation factor. This is useful to execute trajectories at a slower speed for debugging. Use this to generate slower trajectories instead of reducing MotionGenConfig.velocity_scale or MotionGenConfig.acceleration_scale as those parameters will require re-tuning of the cost terms while MotionGenPlanConfig.time_dilation_factor will only post-process the trajectory.
        )
    
    def _check_prerequisites_for_syncing_target_pose(self, real_target_position, real_target_orientation,sim_js) -> bool:
        robot_prerequisites = self._check_robot_static(sim_js) or self.reactive # robot is allowed to reset_command_plan global plan if stopped (in the non-reactive mode) or anytime in the reactive mode
        target_prerequisites = self._check_target_pose_changed(real_target_position, real_target_orientation) and self._check_target_pose_static(real_target_position, real_target_orientation)
        return robot_prerequisites and target_prerequisites
        
    def reset_command_plan(self, cu_js):
        print("reset_command_planning a new global plan - goal pose has changed!")
            
        # Set EE teleop goals, use cube for simple non-vr init:
        # ee_translation_goal = self.target_last_synced_position # cube position is the updated target pose (which has moved) 
        # ee_orientation_teleop_goal = self.target_last_synced_orientation # cube orientation is the updated target orientation (which has moved)

        # compute curobo solution:
        p_RT, R_RT  = self.target.get_local_pose() # NOTE: position (p) and rotation (R) of frame T (target frame) expressed in the robot's base frame R. 
        ik_goal = Pose(position=self.tensor_args.to_device(p_RT), quaternion=self.tensor_args.to_device(R_RT))
        self.plan_config.pose_cost_metric = self.pose_metric
        start_state = cu_js.unsqueeze(0) # cu_js is the current joint state of the robot
        goal_pose = ik_goal # ik_goal is the updated target pose (which has moved)
        
        result: MotionGenResult = self.solver.plan_single(start_state, goal_pose, self.plan_config) # https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGen.plan_single:~:text=GraphResult-,plan_single,-( , https://curobo.org/_api/curobo.wrap.reacher.motion_gen.html#curobo.wrap.reacher.motion_gen.MotionGenResult:~:text=class-,MotionGenResult,-(
        succ = result.success.item()  # an attribute of this returned object that signifies whether a trajectory was successfully generated. success tensor with index referring to the batch index.
        
        if self.num_targets == 1: # it's 1 only immediately after the first time it found a successfull plan for the FIRST time (first target).
            if self.constrain_grasp_approach:
                # cuRobo also can enable constrained motions for part of a trajectory.
                # This is useful in pick and place tasks where traditionally the robot goes to an offset pose (pre-grasp pose) and then moves 
                # to the grasp pose in a linear motion along 1 axis (e.g., z axis) while also constraining it’s orientation. We can formulate this two step process as a single trajectory optimization problem, with orientation and linear motion costs activated for the second portion of the timesteps. 
                # https://curobo.org/advanced_examples/3_constrained_planning.html#:~:text=Grasp%20Approach%20Vector,behavior%20as%20below.
                # Enables moving to a pregrasp and then locked orientation movement to final grasp.
                # Since this is added as a cost, the trajectory will not reach the exact offset, instead it will try to take a blended path to the final grasp without stopping at the offset.
                # https://curobo.org/_api/curobo.rollout.cost.pose_cost.html#curobo.rollout.cost.pose_cost.PoseCostMetric.create_grasp_approach_metric
                self.pose_metric = PoseCostMetric.create_grasp_approach_metric() # 
            if self.reach_partial_pose is not None:
                # This is probably a way to update the cost metric for reaching a partial pose reaching (not sure how, no documentation).
                reach_vec = self.solver.tensor_args.to_device(args.reach_partial_pose)
                self.pose_metric = PoseCostMetric(
                    reach_partial_pose=True, reach_vec_weight=reach_vec
                )
            if self.hold_partial_pose is not None:
                # This is probably a way to update the cost metric for reaching a partial pose reaching (not sure how, no documentation).
                hold_vec = self.solver.tensor_args.to_device(args.hold_partial_pose)
                self.pose_metric = PoseCostMetric(hold_partial_pose=True, hold_vec_weight=hold_vec)
        
        if succ: 
            print(f"target counter - targets with a reachible plan = {self.num_targets}") 
            self.num_targets += 1
            cmd_plan = result.get_interpolated_plan() # TODO: To clarify myself what get_interpolated_plan() is doing to the initial "result"  does. Also see https://curobo.org/get_started/2a_python_examples.html#:~:text=result%20%3D%20motion_gen.plan_single(start_state%2C%20goal_pose%2C%20MotionGenPlanConfig(max_attempts%3D1))%0Atraj%20%3D%20result.get_interpolated_plan()%20%20%23%20result.interpolation_dt%20has%20the%20dt%20between%20timesteps%0Aprint(%22Trajectory%20Generated%3A%20%22%2C%20result.success)
            cmd_plan = self.solver.get_full_js(cmd_plan) # get the full joint state from the interpolated plan
            # get only joint names that are in both:
            self.idx_list = []
            self.common_js_names = []
            for x in self.get_dof_names():
                if x in cmd_plan.joint_names:
                    self.idx_list.append(self.robot.get_dof_index(x))
                    self.common_js_names.append(x)

            cmd_plan = cmd_plan.get_ordered_joint_state(self.common_js_names)
            
            self.cmd_plan = cmd_plan # global plan
            self.cmd_idx = 0 # commands executed from the global plan (counter)
        else:
            carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            cmd_plan = None

    def get_curobo_joint_state(self, sim_js=None,zero_vel=False) -> JointState:
        """
        See super().get_curobo_joint_state() for more details.

        Args:
            sim_js (_type_): _description_

        Returns:
            JointState: _description_
        """
        cu_js = super().get_curobo_joint_state(sim_js, zero_vel)

        if not self.reactive: # In reactive mode, we will not wait for a complete stopping of the robot before navigating to a new goal pose (if goal pose has changed). In the default mode on the other hand, we will wait for the robot to stop.
                cu_js.velocity *= 0.0
                cu_js.acceleration *= 0.0

        if self.reactive and self.past_cmd is not None:
            cu_js.position[:] = self.past_cmd.position
            cu_js.velocity[:] = self.past_cmd.velocity
            cu_js.acceleration[:] = self.past_cmd.acceleration

        cu_js = cu_js.get_ordered_joint_state(self.solver.kinematics.joint_names) 


        return cu_js    
    

    def get_current_plan_as_tensor(self, to_go_only=True) -> torch.Tensor:
        """
        Returns the joint states at at the start of each command in the current plan and the joint velocity commands in the current plan.

        Args:
            to_go_only (bool, optional):If true, only the commands left to execute in the current plan are returned (else, all commands, including the ones already executed, are returned). Defaults to True.

        Returns:
            _type_: _description_
        """
        if self.cmd_plan is None:
            return None # robot is not following any plan at the moment
        n_total = len(self.cmd_plan) # total num of commands (actions) to apply to controller in current command (global) plan
        n_applied = self.cmd_idx # number of applied actions from total command plan
        n_to_go = n_total - n_applied # num of commands left to execute 
        start_idx = 0 if not to_go_only else n_applied
        start_q = self.cmd_plan.position[start_idx:] # at index i: joint positions just before applying the ith command
        vel_cmd = self.cmd_plan.velocity[start_idx:] # at index i: joint velocities to apply to the ith command
        return torch.stack([start_q, vel_cmd]) # shape: (2, len(plan), dof_num)
    
    def get_next_articulation_action(self,idx_list):
        next_cmd = self.cmd_plan[self.cmd_idx] # get the next joint command from the plan
        self.past_cmd = next_cmd.clone() # save the past command for future use
        next_cmd_joint_pos = next_cmd.position.cpu().numpy() # Joint configuration of the next command.
        next_cmd_joint_vel = next_cmd.velocity.cpu().numpy() # Joint velocities of the next command.
        art_action = ArticulationAction(next_cmd_joint_pos, next_cmd_joint_vel,joint_indices=idx_list,) # controller command
        return art_action
    
#############################################
# MAIN SIMULATION LOOP
#############################################
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
    
    tensor_args = TensorDeviceType()  # Device configuration for tensor operations
    if ENABLE_GPU_DYNAMICS:
        activate_gpu_dynamics(my_world)
    
    # Adding two frankas to the scene
    # # Inspired by curobo/examples/isaac_sim/batch_motion_gen_reacher.py but this time at the same world (the batc)

    robots: List[Optional[AutonomousFranka]] = [None, None]
    robots_cu_js: List[Optional[JointState]] =[None, None] # for visualization of robot spheres
    robots_collision_caches = [{"obb": 100, "mesh": 100}, {"obb": 30, "mesh": 10}]
    robot_cfgs = [load_yaml(f"projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/franka{i}.yml")["robot_cfg"] for i in range(1,3)]
    robot_idx_lists:List[Optional[List]] = [None, None]
    
    # First set robot2 (cumotion robot) so we can use it to initialize the collision predictor of robot1.
    p_Trobot2 =np.array([0.3,0,0.5])
    robot2 = FrankaCumotion(robot_cfgs[1], my_world, usd_help, p_R=np.array([1,0.0,0.0]), p_T=p_Trobot2) # cumotion robot - interferer
    robots[1] = robot2 
    # init cumotion solver and plan config
    robot2.init_solver(robots_collision_caches[1],tensor_args)
    robot2.init_plan_config() # TODO: Can probably be move to constructor.
    robot1 = FrankaMpc(robot_cfgs[0], my_world,usd_help) # MPC robot - avoider
    robots[0] = robot1
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
    
    # reset world
    my_world.step(render=True)
    my_world.reset()
    
    # Set robots in initial joint configuration (in curobo they call it  the "retract" config)
    for i, robot in enumerate(robots):
        assert robot is not None
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot.init_joints(robot_idx_lists[i])

    step_dt_traj_mpc = RENDER_DT if SIMULATING else REAL_TIME_EXPECTED_CTRL_DT  
    expected_ctrl_freq_at_mpc = 1 / step_dt_traj_mpc # This is what the mpc "thinks" the control frequency should be. It uses that to generate the rollouts.                
    dynamic_obs_coll_predictor = DynamicObsCollPredictor(tensor_args, step_dt_traj_mpc) if MODIFY_MPC_COST_FN_FOR_DYN_OBS else None # Now if we are modifying the MPC cost function to predict poses of moving obstacles, we need to initialize the mechanism which does it. That's the  DynamicObsCollPredictor() class.
    robot1.init_solver(robots_collision_caches[0], step_dt_traj_mpc, dynamic_obs_coll_predictor)
  
    
    ctrl_loop_start_time = time.time()
    robot1_init_obs = False
    t_idx = 0 # time step index in real world (not simulation) steps. This is the num of completed control steps (actions) in *played* simulation (after play button is pressed)
    while simulation_app.is_running(): # not necessarily playing, just running                
        
        my_world.step(render=True) # print_rate_decorator(lambda: my_world.step(render=True), args.print_ctrl_rate, "my_world.step")() # UPDATE PHYSICS OF SIMULATION AND IF RENDER IS TRUE ALSO UPDATING UI ELEMENTS, VIEWPORTS AND CAMERAS.(Executes one physics step and one rendering step).Note: rendering means rendering a frame of the current application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will be refreshed as well if running non-headless.) See: https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html, see alse https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World       
        ########## ROBOT 1 STEP ##########
        #### START MEASURING CTRL FREQ IF NEEDED AND CAN #######    
        if args.print_ctrl_rate and not SIMULATING:
            real_robot_cfm_is_initialized = not np.isnan(real_robot_cfm_start_time) # is the control frequency measurement already initialized?
            real_robot_cfm_can_be_initialized = t_idx > real_robot_cfm_min_start_t_idx # is it valid to start measuring control frequency now?
            if not real_robot_cfm_is_initialized and real_robot_cfm_can_be_initialized:
                real_robot_cfm_start_time = time.time()
                real_robot_cfm_start_t_idx = t_idx # my_world.current_time_step_index is "t", current time step. Num of *completed* control steps (actions) in *played* simulation (after play button is pressed)
        
        ######### UPDATE GOAL POSE AT MPC IF GOAL MOVED #########
        # Get target position and orientation
        real_world_pos_target1, real_world_orient_target1 = robot1.target.get_world_pose() # print_rate_decorator(lambda: , args.print_ctrl_rate, "target.get_world_pose")() # goal pose
        # Update goals if targets has moved
        robot1_target_changed = robot1.update_target_if_needed(real_world_pos_target1, real_world_orient_target1)
        if robot1_target_changed:
            print("robot1 target changed!")
            robot1.goal_buffer.goal_pose.copy_(robot1.get_last_synced_target_pose())
            robot1.solver.update_goal(robot1.goal_buffer)
 
        
        ############################################################
        ########## ROBOT 2 STEP ##########
        ############################################################
        print("t_idx = ", t_idx)
        
        robots_cu_js[0] = robot1.get_curobo_joint_state(robot1.get_sim_joint_state())
        robot1.update_current_state(robots_cu_js[0])
        sim_js_robot2 = robot2.get_sim_joint_state() # robot2.robot.get_joints_state() # reading current joint state from robot
        robots_cu_js[1] = robot2.get_curobo_joint_state(sim_js_robot2) 
        real_world_pos_target2, real_world_orient_target2 = robot2.target.get_world_pose() # print_rate_decorator(lambda: , args.print_ctrl_rate, "target.get_world_pose")() # goal pose
        robot2_target_changed = robot2.update_target_if_needed(real_world_pos_target2, real_world_orient_target2,sim_js_robot2)
        
        if robot2_target_changed:
            print("robot2 target changed, updating plan...")
            robot2.reset_command_plan(robots_cu_js[1]) # replanning a new global plan and setting robot2.cmd_plan to point the new plan.
            robot2_plan = robot2.get_current_plan_as_tensor()
            if robot2_plan is not None:    
                pos_jsR2fullplan, vel_jsR2fullplan = robot2_plan[0], robot2_plan[1] # from current time step t to t+H-1 inclusive
                # Compute FK on robot2 plan: all poses and orientations are expressed in robot2 frame (R2). Get poses of robot2's end-effector and links in robot2 frame (R2) and spheres (obstacles) in robot2 frame (R2).
                p_eeR2fullplan_R2, q_eeR2fullplan_R2, _, _, p_linksR2fullplan_R2, q_linksR2fullplan_R2, p_rad_spheresR2fullplan_R2 = robot2.crm.forward(pos_jsR2fullplan, vel_jsR2fullplan) # https://curobo.org/_api/curobo.cuda_robot_model.cuda_robot_model.html#curobo.cuda_robot_model.cuda_robot_model.CudaRobotModelConfig
                # convert to world frame (W):
                p_rad_spheresR2fullplan = p_rad_spheresR2fullplan_R2[:,:,:].cpu() # copy of the spheres in robot2 frame (R2)
                p_rad_spheresR2fullplan[:,:,:3] = p_rad_spheresR2fullplan[:,:,:3] + robot2.p_R # # offset of robot2 origin in world frame (only position, radius is not affected)
                p_spheresR2fullplan = p_rad_spheresR2fullplan[:,:,:3]
                rad_spheresR2 = p_rad_spheresR2fullplan[0,:,3] # 65x4 sphere centers (x,y,z) and radii (4th column)
        
                if MODIFY_MPC_COST_FN_FOR_DYN_OBS: # init the new robot2 plan as obstacles for robot1    
                    assert dynamic_obs_coll_predictor is not None # just to ignore warnings
                    p_spheresR2H = p_spheresR2fullplan[:robot1.H].to(tensor_args.device) # horizon length
                    
                    if not robot1_init_obs: # after planning the first global plan by R2 (but before executing it)
                        print("Obstacles initiation: Adding dynamic obstacle to collision checker")
                        dynamic_obs_coll_predictor.add_obs(p_spheresR2H, rad_spheresR2.to(tensor_args.device))
                        # robot2_as_obs_obnames = []
                        if HIGHLIGHT_OBS:
                            for h in range(p_spheresR2H.shape[0]):
                                for i in range(p_spheresR2H.shape[1]):
                                    obs_nameih = f'{robot2.robot_name}_obs{i}_h{h}'
                                    robot1.obs_viz_obs_names.append(obs_nameih)
                                    robot1.add_obs_viz(p_spheresR2H[h,i].cpu(),rad_spheresR2[i].cpu(),obs_nameih,h=h,h_max=robot1.H)
                        print("Added dynamic obstacles tocollision checker")
                        robot1_init_obs = True
             
                        
                else:
                    if not robot1_init_obs: # after planning the first global plan by robot 2 (but before executing it)
                        print("Obstacles initiation: Adding static obstacles to original curobo collision checker")
                        p_validspheresR2curr, rad_validspheresR2, valid_sphere_indices_R2 = robot2.get_current_spheres_state()
                        robot2_as_obs_obnames = [f'{robot2.robot_name}_obs_{i}' for i in valid_sphere_indices_R2]
                        for i in range(len(robot2_as_obs_obnames)):
                            robot1.obs_viz_obs_names.append(robot2_as_obs_obnames[i])
                        robot2_sphere_list = get_sphere_list_from_sphere_tensor(p_validspheresR2curr, rad_validspheresR2, robot2_as_obs_obnames, robot2.tensor_args)
                        robot2_cube_list = [sphere.get_cuboid() for sphere in robot2_sphere_list]
                        r1_mesh_cchecker = WorldMeshCollision(WorldCollisionConfig(tensor_args, world_model=WorldConfig.create_collision_support_world(robot1.cu_stat_obs_world_model)))
                        for cube in robot2_cube_list:
                            robot1.cu_stat_obs_world_model.add_obstacle(cube)
                        print("Obstacles initiation: Added static obstacles to original curobo collision checker")
                        if HIGHLIGHT_OBS:
                            for i in range(len(robot2_as_obs_obnames)):
                                robot1.add_obs_viz(p_validspheresR2curr[i],rad_validspheresR2[i],robot2_as_obs_obnames[i],h=0,h_max=1)
                        robot1_init_obs = True        
        
        if robot2.cmd_plan is not None and robot1_init_obs: # if the robot2 has a plan to execute (otherwise it should be static)    
            if MODIFY_MPC_COST_FN_FOR_DYN_OBS: # Update obstacles in robot1's collision checker according to the current plan of robot2
                # move sliding window of predicted dynamic obstacles
                max_idx_window = robot2.cmd_idx + robot1.H - 1
                n_cmds_plan = len(p_spheresR2fullplan)
                max_cmd_idx_plan = n_cmds_plan - 1 
                if max_idx_window <= max_cmd_idx_plan: # if the window is within the plan
                    p_spheresR2H = p_spheresR2fullplan[robot2.cmd_idx: max_idx_window + 1]
                else: # else embed in window the last predicted positions in the plan 
                    p_spheresR2H = torch.cat([p_spheresR2H[1:],p_spheresR2H[-1].unsqueeze(0)])
                dynamic_obs_coll_predictor.update_p_obs(p_spheresR2H.to(tensor_args.device))
                if HIGHLIGHT_OBS and t_idx % robot1.H == 0: # 
                    p_spheresR2H_reshaped_for_viz = p_spheresR2H.reshape(-1, 3) # collapse first two dimensions
                    robot1.update_obs_viz(p_spheresR2H_reshaped_for_viz.cpu())                
            
            else: # Update static obstacles in robot1's collision checker by reading the current state of robot2
                p_validspheresR2curr, _, _ = robot2.get_current_spheres_state()     
                for i in range(len(robot2_as_obs_obnames)):
                    name = robot2_as_obs_obnames[i]
                    X_sphere = Pose.from_list(p_validspheresR2curr[i].tolist() + [1,0,0,0])
                    r1_mesh_cchecker.update_obstacle_pose(name, X_sphere)
                if HIGHLIGHT_OBS:
                    robot1.update_obs_viz(p_validspheresR2curr)
        # mpc planning
        mpc_result = robot1.solver.step(robot1.current_state, max_attempts=2) # print_rate_decorator(lambda: robot1.solver.step(robot1.current_state, max_attempts=2), args.print_ctrl_rate, "mpc.step")()
        
        # apply actions in robots
        robot1_art_action = robot1.get_next_articulation_action(mpc_result.js_action) # get articulated action from joint state action
        robot1.apply_articulation_action(robot1_art_action,num_times=3) # Note: I chhanged it to 1 instead of 3
        if robot2.cmd_plan is not None:
            robot2_art_action = robot2.get_next_articulation_action(idx_list=robot_idx_lists[1])
            robot2.apply_articulation_action(robot2_art_action)
         
        if t_idx % 100 == 0 and robot1_init_obs: # change pose of target in simulator of robot2 every 100 steps
            p_validspheresR1curr, _, valid_sphere_indices_R1 = robot1.get_current_spheres_state()
            new_target_idx = np.random.choice(valid_sphere_indices_R1[20:]) # above the base of the robot
            p_new_target =  p_validspheresR1curr[new_target_idx]
            robot2.target.set_world_pose(position=np.array(p_new_target.tolist()), orientation=np.random.rand(4))

        ############ OPTIONAL VISUALIZATIONS ###########
        # Visualize spheres, rollouts and predicted paths of dynamic obstacles (if needed) ############
        if VISUALIZE_ROBOT_COL_SPHERES and t_idx % 2 == 0:
            for i, robot in enumerate(robots):
                robot.visualize_robot_as_spheres(robots_cu_js[i])

        if VISUALIZE_MPC_ROLLOUTS or (VISUALIZE_PREDICTED_OBS_PATHS and MODIFY_MPC_COST_FN_FOR_DYN_OBS): # rendering using draw_points()
            point_visualzer_inputs = [] # collect the different points sequences for visualization
            # collect the rollouts
            if VISUALIZE_MPC_ROLLOUTS:
                rollouts_for_visualization = {'points': robot1.solver.get_visual_rollouts(), 'color': 'green'}
                point_visualzer_inputs.append(rollouts_for_visualization)
        
            # render the points
            if MODIFY_MPC_COST_FN_FOR_DYN_OBS:
                global_plan_points = {'points': p_spheresR2H, 'color': 'green'}
                point_visualzer_inputs.append(global_plan_points)
            draw_points(point_visualzer_inputs) # print_rate_decorator(lambda: draw_points(point_visualzer_inputs), args.print_ctrl_rate, "draw_points")() 

        ############### UPDATE TIME STEP INDEX  ###############
        t_idx += 1 # num of completed control steps (actions) in *played* simulation (after play button is pressed)
        print(f"New t_idx: (num of control steps done, in the control loop):{t_idx}")    
        # print(f'Control loop elapsed time (time we executed the simulation so far, in real world time, not simulation internal clock): {(time.time() - ctrl_loop_start_time):.5f}')
        # print(f'Sim stats: my_world.current_time_step_index: {my_world.current_time_step_index}')
        # print(f'Sim stats: my_world.current_time: {my_world.current_time:.5f} (physics_dt={PHYSICS_STEP_DT:.5f})')  
        if args.print_ctrl_rate and (SIMULATING or real_robot_cfm_is_initialized):
            print_ctrl_rate_info(t_idx,real_robot_cfm_start_time,real_robot_cfm_start_t_idx,expected_ctrl_freq_at_mpc,step_dt_traj_mpc)



        
        
if __name__ == "__main__":
    main()
    simulation_app.close() 
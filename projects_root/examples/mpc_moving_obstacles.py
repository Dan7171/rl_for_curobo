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
ENABLE_GPU_DYNAMICS = True
MODIFY_MPC_COST_FUNCTION_TO_HANDLE_MOVING_OBSTACLES = False # If True, this would be what the original MPC cost function could handle. False means that the cost will consider obstacles as moving and look into the future, while True means that the cost will consider obstacles as static and not look into the future.
DEBUG_COST_FUNCTION = False # If True, then the cost function will be printed on every call to my_world.step()
FORCE_CONSTANT_VELOCITIES = True # If True, then the velocities of the dynamic obstacles will be forced to be constant. This eliminates the phenomenon that the dynamic obstacle is slowing down over time.
VISUALIZE_PREDICTED_OBS_PATHS = True # If True, then the predicted paths of the dynamic obstacles will be rendered in the simulation.
VISUALIZE_MPC_ROLLOUTS = True # If True, then the MPC rollouts will be rendered in the simulation.
VISUALIZE_ROBOT_COL_SPHERES = False # If True, then the robot collision spheres will be rendered in the simulation.

###################### RENDER_DT and PHYSICS_STEP_DT ########################
RENDER_DT = 0.03 # original 1/60
PHYSICS_STEP_DT = 0.03 # original 1/60
# NOTE: RENDER_DT and PHYSICS_STEP_DT guide from emperical experiments!:

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


try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import time
from typing import List
from curobo.geom.sdf.world_mesh import WorldMeshCollision
import torch
import argparse
import os
import carb
import numpy as np
import copy

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


# Import helper from curobo examples
from projects_root.utils.helper import add_extensions, add_robot_to_scene

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
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle

from curobo.wrap.reacher.motion_gen import ( # For visualizing robot spheres
    MotionGen,
    MotionGenConfig,
)

# Initialize CUDA device
a = torch.zeros(4, device="cuda:0") 

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
################### Read arguments ########################
parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="Run in headless mode. Options: [native, websocket]. Note: webrtc might not work.",
)
parser.add_argument(
    "--robot",
    type=str,
    default="franka.yml",
    help="Robot configuration file to load (e.g., franka.yml)",
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

# Convert string arguments to boolean
args.autoplay = args.autoplay.lower() == "true"
args.print_ctrl_rate = args.print_ctrl_rate.lower() == "true"


###########################################################
######################### HELPER ##########################
###########################################################
def create_target_pose_hologram():
    """ 
    Create a target pose "hologram" in the simulation. By "hologram", 
    we mean a visual representation of the target pose that is not used for collision detection or physics calculations.
    In isaac-sim they call holograms viual objects (like visual coboid and visual spheres...)
    """
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([0, 1, 0]),
        size=0.05,
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
        predicted_path = dynamic_obs_coll_predictor.get_predicted_path(obs_name)
        predicted_path = predicted_path[:,:3]
        obs_predicted_poses[0] = torch.tensor(predicted_path)
        obs_color = dynamic_obstacles[obs_index].simulation_representation.get_applied_visual_material().get_color()
        obs_for_visualization.append({"points": obs_predicted_poses, "color": obs_color})
    return obs_for_visualization

def visualize_spheres(motion_gen, spheres, cu_js):
    """
    Render collision spheres of the robot for visualization purposes only.
    Took from examples like motion_gen_reacher.py.

    Args:
        motion_gen (_type_): _description_
        spheres (_type_): _description_
        cu_js (_type_): _description_
    """
    
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

def init_robot_spheres_visualizer(robot_cfg,world_cfg,tensor_args,collision_cache):
    """
    Initialize the robot spheres visualizer.
    """
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    interpolation_dt = 0.05
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
    tensor_args,
    collision_checker_type=CollisionCheckerType.MESH,
    num_trajopt_seeds=12,
    num_graph_seeds=12,
    interpolation_dt=interpolation_dt,
    collision_cache=collision_cache,
    optimize_dt=optimize_dt,
    trajopt_dt=trajopt_dt,
    trajopt_tsteps=trajopt_tsteps,
    trim_steps=trim_steps,
)
    motion_gen = MotionGen(motion_gen_config)
    print("warming up motion gen...")
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    return motion_gen

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
    # Initialize Isaac Sim world with 1 meter units
    my_world = World(stage_units_in_meters=1.0) 
    
    # GPU DYNAMICS - OPTIONAL (originally was disabled)
    # GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU. However, this will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. If enabling GPU dynamics slows down the simulation, it may be that your GPU is not able to handle the additional load. You can enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.
    # See: https://docs-prod.omniverse.nvidia.com/isaacsim/latest/reference_material/speedup_cheat_sheet.html?utm_source=chatgpt.com
    # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
    
    if ENABLE_GPU_DYNAMICS:
        my_world_physics_context = my_world.get_physics_context()
        if not my_world_physics_context.is_gpu_dynamics_enabled():
            print("GPU dynamics is disabled")
            my_world_physics_context.enable_gpu_dynamics(True)
            assert my_world_physics_context.is_gpu_dynamics_enabled()
            print("debug- experimental: GPU dynamics is enabled")
    
    # Set the simulation dt to the desired values.
    my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT) 
    
    stage = my_world.stage

    # Set up the world hierarchy
    xform = stage.DefinePrim("/World", "Xform")  # Root transform for all objects
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")  # Transform for CuRobo-specific objects
    my_world.scene.add_default_ground_plane()

    # Create a target cube for the robot to follow
    target = create_target_pose_hologram()

    # Configure CuRobo logging and parameters
    setup_curobo_logger("warn")
    
    past_pose = None
    n_obstacle_cuboids = 30  # Number of collision boxes for obstacle approximation https://curobo.org/get_started/2c_world_collision.html
    n_obstacle_mesh = 10     # Number of mesh triangles for obstacle approximation https://curobo.org/get_started/2c_world_collision.html

    # Initialize CuRobo components
    usd_help = UsdHelper()  # Helper for USD stage operations
    target_pose = None
    tensor_args = TensorDeviceType()  # Device configuration for tensor operations

    # Load and configure robot
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02  # Add safety margin

    # Add robot to scene and get controller
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()

    # Load world configuration for collision checking
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04  # Adjust table height
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5  # Place mesh below ground
    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh) # representation of the world for use in curobo
    
    # Create and configure obstacles 
    collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh}
    if SIMULATING:
        step_dt_traj_mpc = RENDER_DT 
    else:
        step_dt_traj_mpc = REAL_TIME_EXPECTED_CTRL_DT    
    expected_ctrl_freq_at_mpc = 1 / step_dt_traj_mpc # This is what the mpc "thinks" the control frequency should be. It uses that to generate the rollouts.            
    
    
    # Set the world config for the dynamic obstacle collision checker.
    # If modifying cost (adding new cost term for dynamic obstacles), we pass a new instance of WorldConfig which will include only the dynamic obstacles, ignoring the static obstacles.
    # Else, we pass the original world config instance which will include both static and dynamic obstacles (as that's how the MPC would address this).
    if MODIFY_MPC_COST_FUNCTION_TO_HANDLE_MOVING_OBSTACLES:
        world_cfg_dynamic_obs = WorldConfig() # curobo collision checker world config for dynamic obstacles. note: this instance should be pased to all obstacles. Will only be used as a template for the dynamic obstacle collision checker.
    else:
        world_cfg_dynamic_obs = world_cfg # Use MPCs original world config instance to include both static and dynamic obstacles.

    # Initialize the dynamic obstacles.
    dynamic_obstacles = [
        Obstacle( 
            name="dynamic_cuboid1", 
            initial_pos=np.array([0.8,0.0,0.5]), 
            dims=0.1, 
            obstacle_type=DynamicCuboid, 
            color=np.array([1,0,0]), # red 
            mass=args.obstacle_mass,
            linear_velocity=[-0.25, 0.0, 0.0],
            angular_velocity=[1,1,1],
            gravity_enabled=args.gravity_enabled.lower() == "true",
            world=my_world,
            world_cfg=world_cfg_dynamic_obs
        )
        ,  
        Obstacle(
            name="dynamic_cuboid2",
            initial_pos=np.array([0.8,0.8,0.3]), 
            dims=0.1, 
            obstacle_type=DynamicCuboid, 
            color=np.array([0,0,1]),# blue 
            mass=args.obstacle_mass,
            linear_velocity=[-0.15, -0.15, 0.05],
            angular_velocity=[0,0,0],
            gravity_enabled=args.gravity_enabled.lower() == "true",
            world=my_world,
            world_cfg=world_cfg_dynamic_obs
        )  
        # NOTE: 1.Add more obstacles here if needed (Call the Obstacle() constructor for each obstacle as in item in the list).
        # NOTE: 2.must initialize the Obstacle() instances before initializing the MpcSolverConfig.
        # NOTE: 3.must initialize the Obstacle() instances before DynamicObsCollisionChecker() initialization.
        ]
    
    # Now if we are modifying the MPC cost function to predict poses of moving obstacles, we need to initialize the mechanism which does it. That's the  DynamicObsCollPredictor() class.
    if MODIFY_MPC_COST_FUNCTION_TO_HANDLE_MOVING_OBSTACLES:    
        dynamic_obs_coll_predictor = DynamicObsCollPredictor(tensor_args, world_cfg_dynamic_obs, collision_cache, step_dt_traj_mpc)
    else:
        dynamic_obs_coll_predictor = None # this will deactivate the prediction of poses of dynamic obstacles over the horizon in MPC cost function.
 
    # Initialize MPC solver
    init_curobo = False
    # Configuration for MPC
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg, #  Robot configuration. Can be a path to a YAML file or a dictionary or an instance of RobotConfig https://curobo.org/_api/curobo.types.robot.html#curobo.types.robot.RobotConfig
        world_cfg, #  World configuration. Can be a path to a YAML file or a dictionary or an instance of WorldConfig. https://curobo.org/_api/curobo.geom.types.html#curobo.geom.types.WorldConfig
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
        step_dt=step_dt_traj_mpc,  # NOTE: Important! step_dt is the time step to use between each step in the trajectory. If None, the default time step from the configuration~(particle_mpc.yml or gradient_mpc.yml) is used. This dt should match the control frequency at which you are sending commands to the robot. This dt should also be greater than the compute time for a single step. For more info see https://curobo.org/_api/curobo.wrap.reacher.mpc.html
        dynamic_obs_checker=dynamic_obs_coll_predictor,  # Add this line
        override_particle_file='projects_root/projects/dynamic_obs/dynamic_obs_predictor/particle_mpc.yml' # settings in the file will overide the default settings in the default particle_mpc.yml file. For example, num of optimization steps per time step.
    )

    mpc = MpcSolver(mpc_config)
  
     
     
    # Set up initial robot state and goal
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

    # Initialize MPC solver with goal
    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)


    # Load stage and initialize simulation
    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None


    # Initialize motion gen
    
    if VISUALIZE_ROBOT_COL_SPHERES:
        motion_gen, spheres = init_robot_spheres_visualizer(robot_cfg,world_cfg,tensor_args,collision_cache), None

    add_extensions(simulation_app, args.headless_mode)
    
    if not SIMULATING:
        real_robot_cfm_start_time:float = np.nan # system time when control frequency measurement has started (not yet started if np.nan)
        real_robot_cfm_start_t_idx:int = -1 # actual step index when control frequency measurement has started (not yet started if -1)
        real_robot_cfm_min_start_t_idx:int = 10 # minimal step index allowed to start measuring control frequency. The reason for this is that the first steps are usually not representative of the control frequency (due to the overhead at the times of the first steps which include initialization of the simulation, etc.).
    
    

    t_idx = 0 # time step index in real world (not simulation) steps. This is the num of completed control steps (actions) in *played* simulation (after play button is pressed)
    
    # Main simulation loop
    while simulation_app.is_running(): # not necessarily playing, just running
        
        point_visualzer_inputs = []

        # Initialize world if needed
        if not init_world:
            for _ in range(10):
                my_world.step(render=True) 
            init_world = True
            if args.autoplay:
                my_world.play()
                
        # Visualize planned trajectories
        if VISUALIZE_MPC_ROLLOUTS:
            rollouts_for_visualization = {'points': mpc.get_visual_rollouts(), 'color': 'green'}
            point_visualzer_inputs.append(rollouts_for_visualization)
        
        
        # Try stepping simulation (steps will be skipped if the simulation is not playing)
        print_rate_decorator(lambda: my_world.step(render=True), args.print_ctrl_rate, "my_world.step")() # UPDATE PHYSICS OF SIMULATION AND IF RENDER IS TRUE ALSO UPDATING UI ELEMENTS, VIEWPORTS AND CAMERAS.(Executes one physics step and one rendering step).Note: rendering means rendering a frame of the current application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will be refreshed as well if running non-headless.) See: https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html, see alse https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World
        
        if not my_world.is_playing(): # if the play button is not pressed yet
            if args.autoplay: # if autoplay is enabled, play the simulation immediately
                my_world.play()
            continue # skip the rest of the loop
        
        while not my_world.is_playing():
            print("Waiting for play button to be pressed...")
            time.sleep(0.1)
        
        
        # NOW PLAYING!

        # Here the control step starts
        # Reset robot to initial configuration

        
        if t_idx == 0: # number of simulation steps since the play button was pressed
            my_world.reset()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            # Set maximum joint efforts
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            ctrl_loop_start_time = time.time()

        if not init_curobo:
            init_curobo = True

        # Start measuring control frequency if not already started
        if SIMULATING:
            real_robot_cfm_is_initialized, real_robot_cfm_start_t_idx, real_robot_cfm_start_time = None, None, None
        else:
            real_robot_cfm_is_initialized = not np.isnan(real_robot_cfm_start_time) # is the control frequency measurement already initialized?
            real_robot_cfm_can_be_initialized = t_idx > real_robot_cfm_min_start_t_idx # is it valid to start measuring control frequency now?
            if not real_robot_cfm_is_initialized and real_robot_cfm_can_be_initialized:
                real_robot_cfm_start_time = time.time()
                real_robot_cfm_start_t_idx = t_idx # my_world.current_time_step_index is "t", current time step. Num of *completed* control steps (actions) in *played* simulation (after play button is pressed)
        
        # Maintain dynamic obstacle velocities to ovecome the phenomenon that the dynamic obstacle is slowing down over time.
        if FORCE_CONSTANT_VELOCITIES:
            for obs_index in range(len(dynamic_obstacles)):
                dynamic_obstacles[obs_index].simulation_representation.set_linear_velocity(dynamic_obstacles[obs_index].linear_velocity)
                dynamic_obstacles[obs_index].simulation_representation.set_angular_velocity(dynamic_obstacles[obs_index].angular_velocity)
        
        # Update curobo collision checkers with the new dynamic obstacles poses from the simulation (if we modify the MPC cost function to predict poses of dynamic obstacles, the checkers are looking into the future. If not, the checkers are looking at the pose of an object in present during rollouts). 
        if MODIFY_MPC_COST_FUNCTION_TO_HANDLE_MOVING_OBSTACLES:
            # Update curobo collision checkers with the new dynamic obstacles poses from the simulation (if we modify the MPC cost function to predict poses of dynamic obstacles, the checkers are looking into the future. If not, the checkers are looking at the pose of an object in present during rollouts). 
            print_rate_decorator(lambda: dynamic_obs_coll_predictor.update_predictive_collision_checkers(dynamic_obstacles), args.print_ctrl_rate, "dynamic_obs_coll_predictor.update_predictive_collision_checkers")()
            
            # Render predicted paths of dynamic obstacles            
            if VISUALIZE_PREDICTED_OBS_PATHS:
                visualization_points_per_obstacle = get_predicted_dynamic_obss_poses_for_visualization(dynamic_obstacles, dynamic_obs_coll_predictor)                
                point_visualzer_inputs.extend(visualization_points_per_obstacle)
                        
        else:
            for obs_index in range(len(dynamic_obstacles)):
                dynamic_obstacles[obs_index].update_world_coll_checker_with_sim_pose(mpc.world_coll_checker)
            

        
        # ######
        # print("debug: real obstacle poses: ", dynamic_obstacles[0].simulation_representation.get_world_pose())
        # ######

        ######## UPDATE TARGET POSE IF NEEDED (IN CASE IT MOVES) ########
        # Get target position and orientation
        cube_position, cube_orientation = print_rate_decorator(lambda: target.get_world_pose(), args.print_ctrl_rate, "get_world_pose of target")() # goal pose

        # Update goal if target has moved
        if past_pose is None:
            past_pose = cube_position + 1.0
        
        if np.linalg.norm(cube_position - past_pose) > 1e-3: # if the target has moved
            # Set new end-effector goal based on target position
            ee_translation_goal = cube_position
            ee_orientation_teleop_goal = cube_orientation
            ik_goal = Pose(
                position=tensor_args.to_device(ee_translation_goal),
                quaternion=tensor_args.to_device(ee_orientation_teleop_goal),
            )
            goal_buffer.goal_pose.copy_(ik_goal)
            mpc.update_goal(goal_buffer)
            past_pose = cube_position
        ###################################################################

        # Get current robot state
        sim_js = robot.get_joints_state() # get the current joint state of the robot
        js_names = robot.dof_names # get the joint names of the robot
        sim_js_names = robot.dof_names # get the joint names of the robot
        
        # Convert to CuRobo joint state format
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions), 
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )

        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        robot_as_spheres = mpc.kinematics.get_robot_as_spheres(cu_js.position)[0]

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

        
        # Run MPC step
        mpc_result = print_rate_decorator(lambda: mpc.step(current_state, max_attempts=2), args.print_ctrl_rate, "mpc.step")()

        # Process MPC result
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state_full = cmd_state

        ####### Apply robot action #######
        # Create and apply robot action
        art_action = ArticulationAction(cmd_state.position.cpu().numpy(),joint_indices=idx_list,)
        # Execute planned motion
        for _ in range(3):
            articulation_controller.apply_action(art_action)
        
        
        # Visualize spheres, rollouts and predicted paths of dynamic obstacles (if needed)
        if t_idx % 2 == 0 and VISUALIZE_ROBOT_COL_SPHERES:
            visualize_spheres(motion_gen, spheres, cu_js)
        if VISUALIZE_ROBOT_COL_SPHERES or VISUALIZE_PREDICTED_OBS_PATHS:
            print_rate_decorator(lambda: draw_points(point_visualzer_inputs), args.print_ctrl_rate, "draw_points")() 

        # CONTROL STEP FINISHED! We can update the time step index.
        t_idx += 1 # num of completed control steps (actions) in *played* simulation (after play button is pressed)
        print(f"New t_idx: (num of control steps done, in the control loop):{t_idx}")    
        print(f'Control loop elapsed time (time we executed the simulation so far, in real world time, not simulation internal clock): {(time.time() - ctrl_loop_start_time):.5f}')
        print(f'Sim stats: my_world.current_time_step_index: {my_world.current_time_step_index}')
        print(f'Sim stats: my_world.current_time: {my_world.current_time:.5f} (physics_dt={PHYSICS_STEP_DT:.5f})')  
        if args.print_ctrl_rate and (SIMULATING or real_robot_cfm_is_initialized):
            print_ctrl_rate_info(t_idx,real_robot_cfm_start_time,real_robot_cfm_start_t_idx,expected_ctrl_freq_at_mpc,step_dt_traj_mpc)
        
        
if __name__ == "__main__":
    main()
    simulation_app.close() 
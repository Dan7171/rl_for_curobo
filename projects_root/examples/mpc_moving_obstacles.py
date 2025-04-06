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
    --obstacle_type sphere    # Use a sphere instead of cube (default: cuboid)
    --obstacle_velocity -0.1 0.1 0.0  # Move diagonally (default: [-0.1, 0.0, 0.0])
    --enable_physics False    # Disable physical collisions (default: True)
    --obstacle_size 0.15    # Set obstacle size (default: 0.1)
    --obstacle_color 0.0 1.0 0.0  # Green color (default: [1.0, 0.0, 0.0])
    --autoplay False    # Disable autoplay (default: True)
"""

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import time
import torch
import argparse
import os
import carb
import numpy as np


# Initialize the simulation app first (must be before "from omni.isaac.core")

from omni.isaac.kit import SimulationApp  
simulation_app = SimulationApp({"headless": False})

# Now import other Isaac Sim modules
# https://medium.com/@kabilankb2003/isaac-sim-core-api-for-robot-control-a-hands-on-guide-f9b27f5729ab
from omni.isaac.core import World # https://forums.developer.nvidia.com/t/cannot-import-omni-isaac-core/242977/3
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction


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
  omni_python mpc_example_with_moving_obstacle.py --obstacle_type sphere --obstacle_velocity -0.1 0.1 0.0 --obstacle_size 0.15 --autoplay False

  # Blue cuboid starting at specific position with physics enabled
  omni_python mpc_example_with_moving_obstacle.py --obstacle_type cuboid --obstacle_initial_pos 1.0 0.5 0.3 --obstacle_color 0.0 0.0 1.0 --obstacle_mass 1.0

  # Green sphere moving in y direction with custom size and physics disabled
  omni_python mpc_example_with_moving_obstacle.py --obstacle_type sphere --obstacle_velocity 0.0 0.1 0.0 --obstacle_size 0.2 --obstacle_color 0.0 1.0 0.0 --enable_physics False

  # Red cuboid with physics disabled and autoplay disabled
  omni_python mpc_example_with_moving_obstacle.py --enable_physics False --autoplay False
"""
)

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
    "--obstacle_velocity",
    type=float,
    nargs=3,
    default=[-0.1, 0.0, 0.0],
    help="Velocity of the obstacle in x, y, z (m/s). Example: --obstacle_velocity -0.1 0.0 0.0",
)
parser.add_argument(
    "--obstacle_type",
    type=str,
    choices=["cuboid", "sphere"],
    default="cuboid",
    help="Type of obstacle to create (cuboid or sphere)",
)
parser.add_argument(
    "--obstacle_size",
    type=float,
    default=0.1,
    help="Size of the obstacle (diameter for sphere, side length for cuboid) in meters",
)
parser.add_argument(
    "--obstacle_initial_pos",
    type=float,
    nargs=3,
    default=[0.8, 0.0, 0.5],
    help="Initial position of the obstacle in x, y, z (meters). Example: --obstacle_initial_pos 0.8 0.0 0.5",
)
parser.add_argument(
    "--obstacle_color",
    type=float,
    nargs=3,
    default=[1.0, 0.0, 0.0],
    help="RGB color of the obstacle (values between 0 and 1). Example: --obstacle_color 1.0 0.0 0.0 for red",
)
parser.add_argument(
    "--autoplay",
    help="Start simulation automatically without requiring manual play button press",
    default="True",
    type=str,
    choices=["True", "False"],
)
parser.add_argument(
    "--enable_physics",
    help="Enable physical collision between obstacle and robot",
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
    help="Enable gravity for the obstacle (only used if enable_physics=True)",
    default="False",
    type=str,
    choices=["True", "False"],
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--print_ctrl_rate",
    default="False",
    type=str,
    choices=["True", "False"],
    help="When True, prints the control rate",
)
# parser.add_argument(
#     "--allow_low_level_debugger",
#     action="store_true",
#     help="When True, removes the cuda graph improvements to allow debugging with a debugger in a lower level (experimental)", 
#     default=True, # TODO find a way to make it false
# )

args = parser.parse_args()

# Convert string arguments to boolean
args.enable_physics = args.enable_physics.lower() == "true"
args.autoplay = args.autoplay.lower() == "true"
args.print_ctrl_rate = args.print_ctrl_rate.lower() == "true"
# use_cuda_graph = not args.allow_low_level_debugger  # Disable CUDA graph
# use_cuda_graph_metrics= not args.allow_low_level_debugger  # Disable metrics CUDA graph
# use_cuda_graph_full_step= True
# if args.allow_low_level_debugger:
#     print("WARNING: DEBUGGER MODE IS SLOWING KERNELS DOWN")

###########################################################

def draw_points(rollouts: torch.Tensor):
    """
    Visualize MPC rollouts in the simulation.
    
    Args:
        rollouts: Tensor of shape [batch_size, horizon, 3] containing trajectory points
    """
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    from omni.isaac.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
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
def disable_gravity(prim_stage_path,stage):
    """
    Disabling gravity for a given object using Python API, Without disabling physics.
    source: https://forums.developer.nvidia.com/t/how-do-i-disable-gravity-for-a-given-object-using-python-api/297501

    Args:
        prim_stage_path (_type_): like "/World/box"
    """
    from pxr import PhysxSchema
    prim = stage.GetPrimAtPath(prim_stage_path)
    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    physx_api.CreateDisableGravityAttr(True)
def init_cube_obstacle(world, position, size, color, enable_physics=True, mass=1.0, gravity_enabled=True,linear_velocity:np.array=np.nan, angular_velocity:np.array=np.nan):
    """
    Initialize a cube obstacle.
    
    Args:
        world: Isaac Sim world instance
        position: Initial position [x, y, z]
        size: Side length of cube
        color: RGB color array
        enable_physics: If True, creates a physical obstacle that can collide and follow physics.
                      If False, creates a visual-only obstacle that moves without physics.
        mass: Mass in kg (only used if enable_physics=True)
        gravity_enabled: If False, disables gravity for the obstacle (only used if enable_physics=True)
    """
    from omni.isaac.core.objects import DynamicCuboid
    obstacle_name = "dynamic_cuboid1"
    prim = DynamicCuboid(prim_path=f"/World/{obstacle_name}",name=obstacle_name, position=position,size=size,color=color,mass=mass,density=0.9)         

    if linear_velocity is not np.nan:
        prim.set_linear_velocity(linear_velocity)
    if angular_velocity is not np.nan:
        prim.set_angular_velocity(angular_velocity)
    if not gravity_enabled:
        disable_gravity(f"/World/{obstacle_name}", world.stage)
    world.scene.add(prim)
    return prim





def print_rate_decorator(func, print_ctrl_rate, rate_name, return_stats=False):
    def wrapper(*args, **kwargs):
        duration, rate = None, None
        if print_ctrl_rate:
            start = time.time()
        result = func(*args, **kwargs)
        if print_ctrl_rate:
            end = time.time()
            duration = end - start
            print(f"{rate_name} Duration: {duration:.3f} seconds") 
            rate = 1.0 / duration
            print(f"{rate_name} Rate: {rate:.3f} Hz")
        if return_stats:
            return result, (duration, rate)
        else:
            return result
    return wrapper

def create_moving_obstacle(world, position, size=0.1, obstacle_type="cuboid", color=None, enable_physics=False, mass=1.0, gravity_enabled=True):
    """
    Create a moving obstacle in the simulation.
    
    Args:
        world: Isaac Sim world instance
        position: Initial position [x, y, z]
        size: Size of obstacle (diameter for sphere, side length for cube)
        obstacle_type: "cuboid" or "sphere"
        color: RGB color array (defaults to blue if None)
        enable_physics: If True, creates a physical obstacle that can collide and follow physics.
                      If False, creates a visual-only obstacle that moves without physics.
        mass: Mass in kg (only used if enable_physics=True)
        gravity_enabled: If False, disables gravity for the obstacle (only used if enable_physics=True)
    """
    if color is None:
        color = np.array([0.0, 0.0, 0.1])  # Default blue color
    
    if obstacle_type == "cuboid":
        obstacle = init_cube_obstacle(world, position, size, color, enable_physics, mass, gravity_enabled)
    elif obstacle_type == "sphere":
        obstacle = init_sphere_obstacle(world, position, size, color, enable_physics, mass, gravity_enabled)
        
    if args.enable_physics:
        # For physical obstacles, use Isaac Sim's physics engine
        obstacle.set_linear_velocity(np.array(args.obstacle_velocity))
    
    # Add obstacle to CuRobo's collision checker
    if args.obstacle_type == "cuboid":
        moving_obstacle = Cuboid(
            name="moving_obstacle",
            pose=[position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0],
            dims=[args.obstacle_size, args.obstacle_size, args.obstacle_size],
        )
    else:  # sphere
        from curobo.geom.types import Sphere
        moving_obstacle = Sphere(
            name="moving_obstacle",
            pose=[position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0],
            radius=args.obstacle_size/2,
        )
    return moving_obstacle
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
    enable_gpu_dynamics = True
    if enable_gpu_dynamics:
        my_world_physics_context = my_world.get_physics_context()
        if not my_world_physics_context.is_gpu_dynamics_enabled():
            print("GPU dynamics is disabled")
            my_world_physics_context.enable_gpu_dynamics(True)
            assert my_world_physics_context.is_gpu_dynamics_enabled()
            print("debug- experimental: GPU dynamics is enabled")

    print(f"my_world time deltas: physics_dt {my_world.get_physics_dt()} rendering_dt {my_world.get_rendering_dt()}") # 0.166666 (1/60)

    # TUNE PHYSICS AND RENDERING DT - OPTIONAL (originally was disabled, default values are 1/60)
    # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
    # See docs of isaac-sim 4.0.0: 
    # https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.core/docs/index.html?highlight=world#module-omni.isaac.core.world
    # It is recommended that the two values be divisible, with the rendering_dt being equal to or greater than the physics_dt

    tune_physics_and_renderring_dt = True
    if tune_physics_and_renderring_dt:

        new_rendering_dt = 1/60 # 1/60 # originally 1/60 
        new_physics_dt = 1/60 # originally 1/60
        
        # is_devisble = new_rendering_dt / new_physics_dt == int(new_rendering_dt / new_physics_dt)
        # is_rend_dt_goe_phys_dt = new_rendering_dt >= new_physics_dt
        # assert (is_devisble and is_rend_dt_goe_phys_dt), "warning: new_rendering_dt and new_physics_dt are not divisible or new_rendering_dt is less than new_physics_dt. Read docs for more inf:https://docs.isaacsim.omniverse.nvidia.com/4.0.0/py/source/extensions/omni.isaac.core/docs/index.html?highlight=world#module-omni.isaac.core.world"            
        my_world.set_simulation_dt(new_physics_dt, new_rendering_dt) 



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
    n_obstacle_cuboids = 30  # Number of collision boxes for obstacle approximation
    n_obstacle_mesh = 10     # Number of mesh triangles for obstacle approximation

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

    # Create and configure moving obstacle
    moving_obstacle = create_moving_obstacle(my_world, np.array(args.obstacle_initial_pos),args.obstacle_size, args.obstacle_type, np.array(args.obstacle_color),args.enable_physics, args.obstacle_mass, args.gravity_enabled.lower() == "true")
    world_cfg.add_obstacle(moving_obstacle)

    # Initialize MPC solver
    init_curobo = False
    
    # Configuration for MPC
    step_dt_traj_mpc = 0.033 # 0.02
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg, #  Robot configuration. Can be a path to a YAML file or a dictionary or an instance of RobotConfig https://curobo.org/_api/curobo.types.robot.html#curobo.types.robot.RobotConfig
        world_cfg, #  World configuration. Can be a path to a YAML file or a dictionary or an instance of WorldConfig. https://curobo.org/_api/curobo.geom.types.html#curobo.geom.types.WorldConfig
        use_cuda_graph=True, # Use CUDA graph for the optimization step.
        use_cuda_graph_metrics=True, # Use CUDA graph for computing metrics.
        use_cuda_graph_full_step=False, #  Capture full step in MPC as a single CUDA graph. This is experimental and might not work reliably.
        self_collision_check=True, # Enable self-collision check during MPC optimization.
        collision_checker_type=CollisionCheckerType.MESH, # type of collision checker to use. See https://curobo.org/get_started/2c_world_collision.html#world-collision 
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,  # Use Model Predictive Path Integral for optimization
        use_lbfgs=False, # Use L-BFGS solver for MPC. Highly experimental.
        use_es=False, # Use Evolution Strategies (ES) solver for MPC. Highly experimental.
        store_rollouts=True,  # Store trajectories for visualization
        step_dt=step_dt_traj_mpc,  # NOTE: Important! step_dt is the time step to use between each step in the trajectory. If None, the default time step from the configuration~(particle_mpc.yml or gradient_mpc.yml) is used. This dt should match the control frequency at which you are sending commands to the robot. This dt should also be greater than the compute time for a single step. For more info see https://curobo.org/_api/curobo.wrap.reacher.mpc.html
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
    step = 0
    add_extensions(simulation_app, args.headless_mode)
    
    # Main simulation loop
    world_step_calls_count = 0
    played_sim_start_time = -1
    while simulation_app.is_running(): # not necessarily playing, just running
        # Initialize world if needed
        if not init_world:
            for _ in range(10):
                my_world.step(render=True) 
            init_world = True
            if args.autoplay:
                my_world.play()
                
        # Visualize planned trajectories
        print_rate_decorator(lambda: draw_points(mpc.get_visual_rollouts()), args.print_ctrl_rate, "draw_points")() 
        
        # Try stepping simulation (steps will be skipped if the simulation is not playing)
        print_rate_decorator(lambda: my_world.step(render=True), args.print_ctrl_rate, "my_world.step")() # UPDATE PHYSICS OF SIMULATION AND IF RENDER IS TRUE ALSO UPDATING UI ELEMENTS, VIEWPORTS AND CAMERAS.(Executes one physics step and one rendering step).Note: rendering means rendering a frame of the current application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will be refreshed as well if running non-headless.) See: https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html, see alse https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World
        world_step_calls_count += 1
        print("my_world.current_time_step_index (simulation physics steps count): ", my_world.current_time_step_index) # stays 0 until the play button is pressed
        print("my_world.current_time (physics steps count * physics_dt = total simulation time from simulator's perspective): ", my_world.current_time) # stays 0 until the play button is pressed
        print("world_step_calls_count: (loop iteration count) ", world_step_calls_count)
        if not my_world.is_playing(): # if the play button is not pressed yet
            if args.autoplay: # if autoplay is enabled, play the simulation immediately
                my_world.play()
            continue # skip the rest of the loop
        
        while not my_world.is_playing():
            print("Waiting for play button to be pressed...")
            time.sleep(0.1)
        # NOW PLAYING
        
        # Here the control step starts
        # Reset robot to initial configuration
        if world_step_calls_count == 1:
            my_world.reset()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            # Set maximum joint efforts
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
            played_sim_start_time = time.time()

        if not init_curobo:
            init_curobo = True
        
        # obstacles = usd_help.get_obstacles_from_stage(
        #         # only_paths=[obstacles_path],
        #         reference_prim_path="/World",
        #         ignore_substring=["/World/target", "/World/defaultGroundPlane"]
        #     ).get_collision_check_world()
        # mpc.update_world(obstacles)
        
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
        succ = True
        cmd_state_full = mpc_result.js_action
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in cmd_state_full.joint_names:
                idx_list.append(robot.get_dof_index(x))
                common_js_names.append(x)

        cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
        cmd_state_full = cmd_state

        # Create and apply robot action
        art_action = ArticulationAction(
            cmd_state.position.cpu().numpy(),
            joint_indices=idx_list,
        )
        
        # # Print metrics periodically
        # if step_index % 1000 == 0:
        #     print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        # Execute planned motion
        if succ:
            for _ in range(3):
                articulation_controller.apply_action(art_action)
        else:
            carb.log_warn("No action is being taken.")

        curr_time = time.time()
        total_played_sim_time = curr_time - played_sim_start_time
        t = my_world.current_time_step_index  # current time step. Num of *completed* control steps (actions) in *played* simulation (after play button is pressed)
        avg_control_freq_hz = t / total_played_sim_time # Control Rate:num of completed actions / total time of actions Hz
        avg_step_dt = total_played_sim_time / t
        expected_ctrl_freq_hz = 1/step_dt_traj_mpc
        ctrl_freq_ratio = expected_ctrl_freq_hz / avg_control_freq_hz
        if ctrl_freq_ratio > 1.05 or ctrl_freq_ratio < 0.95:
            print(f"WARNING! Control frequency ratio is {ctrl_freq_ratio:.2f}. Expected {expected_ctrl_freq_hz:.2f} Hz, but {avg_control_freq_hz:.2f} Hz was assigned.\n\
                    Change mpc_config.step_dt from {step_dt_traj_mpc} to {avg_step_dt})")

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

if __name__ == "__main__":
    main()
    simulation_app.close() 
 
# ############################## Run settings ##############################





SIMULATING = True # if False, then we are running the robot in real time (i.e. the robot will move as fast as the real time allows)
REAL_TIME_EXPECTED_CTRL_DT = 0.03 #1 / (The expected control frequency in Hz). Set that to the avg time measurded between two consecutive calls to my_world.step() in real time. To print that time, use: print(f"Time between two consecutive calls to my_world.step() in real time, run with --print_ctrl_rate "True")
ENABLE_GPU_DYNAMICS = True # # GPU DYNAMICS - OPTIONAL (originally was disabled)# GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU. However, this will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. If enabling GPU dynamics slows down the simulation, it may be that your GPU is not able to handle the additional load. You can enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.# See: https://docs-prod.omniverse.nvidia.com/isaacsim/latest/reference_material/speedup_cheat_sheet.html?utm_source=chatgpt.com # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
OBS_PREDICTION  = True # If True, this would be what the original MPC cost function could handle. False means that the cost will consider obstacles as moving and look into the future, while True means that the cost will consider obstacles as static and not look into the future.
DEBUG = True # Currenly, the main feature of True is to run withoug cuda graphs. When its true, we can set breakpoints inside cuda graph code (like in cost computation in "ArmBase" for example)  
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
CUROBO_WORLD_MODELS_RESET_T = 100 # If the world models are not updated for a long time, they will be restarted. Set to -1 to disable.
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
    from dataclasses import fields
    import copy
    import math
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
    from isaacsim.core.api.materials import OmniPBR, OmniGlass, PreviewSurface
    from isaacsim.core.api.objects import VisualCuboid, DynamicCuboid,FixedCuboid, VisualSphere, DynamicSphere, VisualCapsule, DynamicCapsule, VisualCone, DynamicCone, VisualCylinder, DynamicCylinder
    from omni.isaac.franka.controllers import PickPlaceController
    from omni.isaac.core.tasks import BaseTask
    from omni.isaac.franka import Franka
    # from omni.isaac.core.utils.stage import add_reference_to_stage
    # from omni.isaac.core.utils.nucleus import get_assets_root_path
    from pxr import UsdGeom, Gf, PhysxSchema, UsdPhysics, Sdf, UsdShade
    # from omni.usd import get_context
    from isaacsim.gui.components import TextBlock
    # from omni.ui import window
    import omni.ui as ui
    # Our modules
    from projects_root.utils.helper import add_extensions
    from projects_root.utils.quaternion import isaacsim_euler2quat
    from projects_root.autonomous_franka import FrankaMpc
    from projects_root.utils.draw import draw_points
    # CuRobo modules
    from curobo.geom.types import Sphere, Cuboid,WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.util.logger import setup_curobo_logger
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import  load_yaml

    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
    from projects_root.utils.sim_prims.klt import Klt
    from projects_root.utils.sim_prims.seattle_lab_table import SeattleLabTable
    # from projects_root.utils.sim_prims.packing_table import PackingTable # currently curobo cant read this because meshes are not triangulated!!!!!!
    from projects_root.utils.sim_prims.stand import Stand
    from projects_root.utils.sim_prims.self_made.simple_table import SimpleTable

    from projects_root.autonomous_franka import AutonomousFranka
    from projects_root.utils.curobo_world_models import update_world_model
    a = torch.zeros(4, device="cuda:0") # prevent cuda out of memory errors (took from curobo examples)

######################### HELPER ##########################
@dataclass
class NPColors:
    red: np.ndarray = np.array([0.5,0,0])
    green: np.ndarray = np.array([0,0.5,0])
    blue: np.ndarray = np.array([0,0,0.5])
    yellow: np.ndarray = np.array([0.5,0.5,0])
    purple: np.ndarray = np.array([0.5,0,0.5])
    orange: np.ndarray = np.array([0.5,0.3,0])
    pink: np.ndarray = np.array([0.5,0.3,0.5])
    white: np.ndarray = np.array([1,1,1])
    black: np.ndarray = np.array([0,0,0])
    # data: list[np.ndarray] = [red, green, blue, yellow, purple, orange, pink, white, black]
    # names: list[str] = ["red", "green", "blue", "yellow", "purple", "orange", "pink", "white", "black"]

class Task:
    
    dst_pose:tuple[np.ndarray, np.ndarray] = None
    @staticmethod
    def sample_target():
        next_target_idx = np.random.randint(0, len(fields(NPColors))) # randomly select a target index
        next_target_color = fields(NPColors)[next_target_idx].name
        print(f"start target color: {next_target_color}")
        return next_target_idx, next_target_color
    
    @staticmethod
    def set_dst_target(robot,p_T,q_T,color_changing_meterial):
        robot.target.set_world_pose(p_T, q_T)
        color_changing_meterial.set_color(getattr(NPColors, next_target_color))
    
    
    


    
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
    
def handle_sigint_gpu_mem_debug(signum, frame):
    print("Caught SIGINT (Ctrl+C), first dump snapshot...")
    torch.cuda.memory._dump_snapshot()
    print("Snapshot dumped to dump_snapshot.pickle, you can upload it to the server: https://docs.pytorch.org/memory_viz")
    print("Now raising KeyboardInterrupt to let the original KeyboardInterrupt handler (of nvidia) to close the app")
    raise KeyboardInterrupt # to let the original KeyboardInterrupt handler (of nvidia) to close the app

def get_robots_mid_posotion(X_Robots:List[np.ndarray]) -> np.ndarray:
    """
    Returns the mid point of the two robots.
    """
    p_mid = (X_Robots[0][:3] + X_Robots[1][:3])/2
    return p_mid
def add_task_in_python_standalone(world:World, task:BaseTask):
    world.add_task(task)
    # to call the task setup_scene run world.reset() 
    # (it replaces the call to await self._world.play_async() from the original example, as we are not using the simulation app but in a standalone app.)
    world.reset() 

def make_3d_grid(center_position, num_points_per_axis, spacing) -> list[np.ndarray]:
    """
    Returns a list of positions (np.ndarray) of the form (x,y,z) for a 3D grid of targets.
    - center_position: Base position for grid (3D)
    - num_points_per_axis: List of [num_x, num_y, num_z] points per axis
    - spacing: List of [step_x, step_y, step_z] distances between points
    """
    targets = []
    half_x = (num_points_per_axis[0] - 1) * spacing[0] / 2
    half_y = (num_points_per_axis[1] - 1) * spacing[1] / 2
    half_z = (num_points_per_axis[2] - 1) * spacing[2] / 2
    
    for i in range(num_points_per_axis[0]):
        for j in range(num_points_per_axis[1]):
            for k in range(num_points_per_axis[2]):
                x = center_position[0] + i * spacing[0] - half_x
                y = center_position[1] + j * spacing[1] - half_y
                z = center_position[2] + k * spacing[2] - half_z
                position = np.array([x, y, z], dtype=np.float32)
                # orientation = np.array([1, 0, 0, 0], dtype=np.float32)  # default orientation
                targets.append(position)
    return targets

def difuse_visual_material(material,base_color, t, T=50):
    """
    Changes the color of the material to a sinusoidal pattern.
    - material: The material to change the color of.
    - t: The current time step.
    - T: The period of the sinusoidal pattern.
    """
    intensity = 0.5 + 0.5 * math.sin(2 * math.pi * t / T)
    # base_color = material.get_color()
    color = base_color + np.array([intensity, intensity, intensity]) 
    material.set_color(color)
    
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
    tensor_args = TensorDeviceType()  # Device configuration for tensor operations
    if ENABLE_GPU_DYNAMICS:
        activate_gpu_dynamics(my_world)
    
    # add_task_in_python_standalone(my_world, FrankaPlaying(name="my_first_task"))
      
    
    # Adding two frankas to the scene
    # # Inspired by curobo/examples/isaac_sim/batch_motion_gen_reacher.py but this time at the same world (the batch motion gen reacher example is for multiple worlds)
    
    include_robots = [True, True] # to include/exclude robots in the simulation
    X_Robots = [
        np.array([0,0,0,1,0,0,0], dtype=np.float32),
        np.array([1.2,0,0,1,0,0,0], dtype=np.float32)
        ] # (x,y,z,qw, qx,qy,qz) expressed in world frame
    n_robots = len(include_robots)
    robots_cu_js: List[Optional[JointState]] =[None for _ in range(n_robots)]# for visualization of robot spheres
    robots_collision_caches = [{"obb": 100, "mesh": 100} for _ in range(n_robots)]
    robot_cfgs = [load_yaml(f"projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/franka{i}.yml")["robot_cfg"] for i in range(1,n_robots+1)]
    robot_idx_lists:List[Optional[List]] = [None for _ in range(n_robots)] 
    robot_world_models = [WorldConfig() for _ in range(n_robots)]
    X_Targets = [[0.6, 0, 0.5, 0, 1, 0, 0], [0.6, 0, 0.5, 0, 1, 0, 0]]# [[0.6, 0, 0.2, 0, 1, 0, 0] for _ in range(n_robots)]
    DIM_SIZE_TARGETS = 0.05
    target_colors = [NPColors.green, NPColors.red]
    if OBS_PREDICTION:
        col_pred_with = [[1], [0]] # at each entry i, list of indices of robots that the ith robot will use for dynamic obs prediction
   
    
    robots:List[Optional[AutonomousFranka]] = []
    for i in range(len(include_robots)):
        if include_robots[i]:
            robots.append(FrankaMpc(
                robot_cfgs[i], 
                my_world, 
                usd_help, 
                p_R=X_Robots[i][:3],
                q_R=X_Robots[i][3:], 
                p_T=X_Targets[i][:3],
                q_T=X_Targets[i][3:], 
                target_color=target_colors[i],
                target_size=DIM_SIZE_TARGETS
                )
        )
        else:
            robots.append(None)
    # ENVIRONMENT OBSTACLES - INITIALIZATION
 
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    
    # add extensions
    add_extensions(simulation_app, args.headless_mode) # in all of the examples of curobo it happens somwhere around here, before the simulation begins. I am not sure why, but I kept it as that. 
    
    # wait for the play button to be pressed
    wait_for_playing(my_world, simulation_app,args.autoplay) # wait for the play button to be pressed
    print("Play button was pressed in simulation!")
    
    # initialize robots
    for i, robot in enumerate(robots):
        if not include_robots[i]:
            continue
        # Set robots in initial joint configuration (in curobo they call it  the "retract" config)
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot.init_joints(robot_idx_lists[i])
        # init dynamic obs coll predictors
        if OBS_PREDICTION and len(col_pred_with[i]) and include_robots[i]:
            obs_groups_nspheres = [robots[obs_robot_idx].get_num_of_sphers() for obs_robot_idx in col_pred_with[i] if include_robots[obs_robot_idx]]
            robot.init_col_predictor(obs_groups_nspheres, cost_weight=100, manually_express_p_own_in_world_frame=True)
        # initialize solver
        robot.init_solver(robot_world_models[i],robots_collision_caches[i], MPC_DT, DEBUG)
        robot.robot._articulation_view.initialize() # new (isac 4.5) https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207
 

        

    robots_prim_paths = [robot.get_prim_path() for robot in robots if robot is not None]
    targets_prim_paths = [robot.get_target_prim_path() for robot in robots if robot is not None]
    ignore_prefix = [        
        *robots_prim_paths, # robots
        *targets_prim_paths, # targets
        "/World/defaultGroundPlane",
        "/curobo"
        "/World/Looks", # visual materials
        "/World/target" # targets
    ]
    
    # add assets or any obstacle (not all obstacles curobo can detect successfully)
    mid_point_robots = get_robots_mid_posotion(X_Robots)
    underground_table = FixedCuboid(prim_path="/World/underground_table", position=mid_point_robots,visible=False, scale=[100,100,0.001])
    # klt = Klt(stage, position=[0.6, 0, 0.2], collision=True, gravity=False, scale=[2]*3)
    # seattle_lab_table = SeattleLabTable(stage, position=[0.6, 1, 1], collision=True, gravity=False, scale=[0.1]*3)
    # packing_table = PackingTable(stage, position=[0.6, 0.0, 0.0],rotation=[0,0,90], collision=True, gravity=False, scale=[0.005]*3)
    # stand = Stand(stage, position=[0.6, 0.0, 0.0],collision=False, gravity=False)
    # visual_materials = {}
    # for field in fields(NPColors):
    #     visual_materials[field.name] = PreviewSurface(f"/World/Looks/{field.name}", color=getattr(NPColors, field.name)) # OmniPBR(f"/World/Looks/{field.name}", color=getattr(NPColors, field.name))
    
    # set up the task

    # set up the prims with the changing color material (targets, and central task table)
    color_changing_meterial = PreviewSurface("/World/Looks/targets_color_changing", color=getattr(NPColors, "white"))        
    blinking_material = PreviewSurface("/World/Looks/targets_blinking", color=getattr(NPColors, "white"))
    # blinking_material_shader = UsdShade.Shader(blinking_material_path + "/Shader")
    
    # central_task_table = SimpleTable(stage,  path="/World/simple_table_center", position=mid_point_robots, collision=True, gravity=False, scale=[0.3,0.3,0.3])
    # central_task_table.apply_visual_material(color_changing_meterial, True)
    # prims_to_apply_to = [central_task_table] + [robots[i].target for i in range(len(robots)) if include_robots[i]]
    dst_targets_center = mid_point_robots + np.array([0,0,0.6])
    # target_dst_prim = VisualCuboid(prim_path="/World/destination", position=dst_targets_center, orientation=[0,1,0,0], scale=[0.05]*3, visual_material=color_changing_meterial)
    # prims_to_apply_to = [target_dst_prim] + [robots[i].target for i in range(len(robots)) if include_robots[i]]
    prims_to_apply_to = [robots[i].target for i in range(len(robots)) if include_robots[i]]
    for prim in prims_to_apply_to:
        prim.apply_visual_materials(color_changing_meterial)
    
    # set up the private tables
    private_tables_offset_z = np.array([0, 0, 0.45], dtype=np.float32)
    private_tables_offset_xy = mid_point_robots *(2/3)
    private_tables_positions = [X_Robots[0][:3] + private_tables_offset_z - private_tables_offset_xy, X_Robots[1][:3] + private_tables_offset_z + private_tables_offset_xy]
    private_tables_scale = [0.15, 0.75, 0.02]
    for i, position in enumerate(private_tables_positions):
        if include_robots[i]:   
            private_table = VisualCuboid(prim_path=f"/World/private_table_{i}", position=private_tables_positions[i], scale=private_tables_scale)

    # select first target color
    next_target_idx, next_target_color = Task.sample_target()
    # set up the source targets optional poses
    src_targets:list[list[tuple[np.ndarray, np.ndarray]]] = [[] for _ in range(len(include_robots))]
    
    q_targets_euler = [180,0,0]
    q_targets = isaacsim_euler2quat(*q_targets_euler,degrees=True,order="ZYX")
    # np.array([0,1,0,0],dtype=np.float32)
    
    for robot_idx in range(len(src_targets)):
        if not include_robots[robot_idx]:
            continue
        p_src_targets = make_3d_grid(
            private_tables_positions[robot_idx] + np.array([0,0, (DIM_SIZE_TARGETS + private_tables_scale[2]/2 + 0.125)]),
            [1, len(fields(NPColors)), 1], 
            [0, private_tables_scale[1] / len(fields(NPColors)), 0]
        )
        for i, p_src_target in enumerate(p_src_targets):
            src_targets[robot_idx].append((p_src_target, q_targets))
     

    # set up destination targets optional poses
    dst_targets:list[tuple[np.ndarray, np.ndarray]] = []
    dst_targets_step_z = 0.1
    dst_targets_step_x = 0
    dst_targets_step_y = 0.1
    row_dim = int(np.floor(np.sqrt(len(fields(NPColors)))))
    col_dim = row_dim
    p_dst_targets = make_3d_grid(dst_targets_center, [row_dim, 1, col_dim], [dst_targets_step_x, dst_targets_step_y, dst_targets_step_z])
    for i,p_Tdst in enumerate(p_dst_targets):
        dst_targets.append((p_Tdst, q_targets))
        print(f"destination target {i}: p,q = {list(p_Tdst)}, {list(q_targets)}")
    
    # initiate targets
    for robot_idx in range(len(include_robots)):
        if not include_robots[robot_idx]:
            continue
        robots[robot_idx].set_target_pose(src_targets[robot_idx][next_target_idx][0], src_targets[robot_idx][next_target_idx][1])

    # create ui window
    ui_window = ui.Window("Simulation Stats", width=300, height=150)
    with ui_window.frame:
        with ui.VStack(spacing=2):
            timestep_label = TextBlock("Time Step: ")
            ee_error_label = TextBlock("Pose Error: ")
            control_freq = TextBlock("control freq: ")
            task_score = TextBlock("task scores: ")
            task_status_label = TextBlock("task status: ")
    
    # loop
    
    # time step
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

    # task data
    task_status = [0] * len(include_robots)
    score = [0] * len(include_robots)
    ee_err_pos = [np.inf] * len(include_robots)
    ee_err_rot = [np.inf] * len(include_robots)
    finger_targets = [None] * len(include_robots)

  
    while simulation_app.is_running():
        
        # accumulate inputs for draw_points()
        point_visualzer_inputs = [] 
    
        # step world
        ctrl_loop_timer_start = time.time()       
        world_step_timer_start = time.time()                 
        my_world.step(render=True) # print_rate_decorator(lambda: my_world.step(render=True), args.print_ctrl_rate, "my_world.step")() # UPDATE PHYSICS OF SIMULATION AND IF RENDER IS TRUE ALSO UPDATING UI ELEMENTS, VIEWPORTS AND CAMERAS.(Executes one physics step and one rendering step).Note: rendering means rendering a frame of the current application and not only rendering a frame to the viewports/ cameras. So UI elements of Isaac Sim will be refreshed as well if running non-headless.) See: https://docs.isaacsim.omniverse.nvidia.com/latest/core_api_tutorials/tutorial_core_hello_world.html, see alse https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.world.World       
        world_step_timer += time.time() - world_step_timer_start

        # update obstacles in 
        # curobo world models
        env_obstacles_update_timer_start = time.time()
        if SUPPORT_ASSETS_OUTSIDE_CONFIG and (t_idx < ASSET_FIXATION_T or (ASSET_FIXATION_T == -1)): 
            if t_idx % CUROBO_WORLD_MODELS_RESET_T == 0: # less frequent updates because it takes longer to load the assets)
                for i in range(len(include_robots)):
                    if not include_robots[i]:
                        continue
                    new_world_model:WorldConfig = usd_help.get_obstacles_from_stage(
                        only_paths=["/World"], # only what is under the world prim
                        ignore_substring=ignore_prefix, # expcept these prims (targets, robots, obstacles)
                        reference_prim_path=robots[i].prim_path, # To express the objects in robot's frame (set false to express in world frame)
                    )
                    robots[i].reset_world_model(new_world_model) # replace the current world model with the new one
                    print(f'robot {i} new cchecker: num of obstacles: {len(robots[i].get_world_model().objects)}')
                    for o in robots[i].get_world_model().objects:
                        print(o.name)
        env_obstacles_timer += time.time() - env_obstacles_update_timer_start

        # get plans from all robots (for collision predictors)
        robots_as_obs_timer_start = time.time()
        if OBS_PREDICTION:
            plans = []
            for i in range(len(include_robots)):
                if include_robots[i]:
                    plans.append(robots[i].get_plan()) 
                else:
                    plans.append(None)
        robots_as_obs_timer += time.time() - robots_as_obs_timer_start

        for i in range(len(include_robots)):
            if not include_robots[i]:
                continue
        
            # update collision predictor
            if OBS_PREDICTION and len(col_pred_with[i]): # using prediction of other robots plans
                robots_as_obs_timer_start = time.time()
                p_spheresOthersH = None
                rad_spheresOthersH = None
                for j in range(len(include_robots)):
                    if include_robots[j] and j != i: 
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
                if t_idx == 0 and rad_spheresOthersH is not None:
                    col_pred.set_obs_rads(rad_spheresOthersH)
                    col_pred.set_own_rads(plans[i]['task_space']['spheres']['r'][0].to(tensor_args.device))
                else:
                    if p_spheresOthersH is not None:
                        col_pred.update(p_spheresOthersH)
                robots_as_obs_timer += time.time() - robots_as_obs_timer_start
                if VISUALIZE_PREDICTED_OBS_PATHS:
                    visualizations_timer_start = time.time()
                    point_visualzer_inputs.append({'points': p_spheresRobotjH, 'color': 'green'})
                    visualizations_timer += time.time() - visualizations_timer_start
            
            # update current state in solver
            joint_state_timer_start = time.time()
            robots_cu_js[i] = robots[i].get_curobo_joint_state(robots[i].get_sim_joint_state())
            robots[i].update_current_state(robots_cu_js[i])    
            joint_state_timer += time.time() - joint_state_timer_start
            
            # if target was moved in the world, update the target in the solver
            targets_update_timer_start = time.time()
            p_T, q_T = robots[i].get_target_pose()# robots[i].target.get_world_pose() # print_rate_decorator(lambda: , args.print_ctrl_rate, "target.get_world_pose")() # goal pose
            if robots[i].set_new_target_for_solver(p_T, q_T):
                print(f"robot {i} target changed!")
                robots[i].update_solver_target()
            targets_update_timer += time.time() - targets_update_timer_start

            # make mpc step (update policy and select action)
            mpc_solver_timer_start = time.time()
            mpc_result = robots[i].solver.step(robots[i].current_state, max_attempts=2) # print_rate_decorator(lambda: robot1.solver.step(robot1.current_state, max_attempts=2), args.print_ctrl_rate, "mpc.step")()            
            ee_pose = robots[i].get_fingers_center_pose()# get_ee_pose()
            pos_err = np.linalg.norm(ee_pose[0] - p_T)
            rot_err = np.linalg.norm(ee_pose[1] - q_T)

            ee_err_pos[i] = pos_err 
            ee_err_rot[i] = rot_err # mpc_result.metrics.ee_error.item()
            
            mpc_solver_timer += time.time() - mpc_solver_timer_start
            
            # apply action
            action_timer_start = time.time()
            art_action = robots[i].get_next_articulation_action(mpc_result.js_action) # get articulated action from joint state action
            robots[i].apply_articulation_action(art_action,num_times=1) # Note: I chhanged it to 1 instead of 3
            action_timer += time.time() - action_timer_start
        
        
        # manage task       
        for i in range(len(include_robots)):
            if not include_robots[i]:
                continue
            if ee_err_pos[i] < 0.1 and ee_err_rot[i] < 0.2:
                if task_status[i] == 0: # heading to src   
                    print(f"robot {i} reached source target")
                    if Task.dst_pose is None:
                        Task.dst_pose = dst_targets[next_target_idx]
                    # robots[i].target.set_world_pose(*Task.dst_pose)
                    robots[i].set_target_pose(*Task.dst_pose)
                    print(f"robot {i} reached source target")
                    if all(task_status) == 0:
                        blinking_material.set_color(color_changing_meterial.get_color())

                    task_status[i] = 1 # reached src
                    # robots[i].target.apply_visual_material(blinking_material)
                    robots[i].target.apply_visual_materials(blinking_material)
                elif task_status[i] == 1: # src reached, heading to dst
                    # meaning it now reached dst
                    print(f"robot {i} reached destination target")
                    print(f"resetting targets...")
                    print(f"ee_erros:{ee_err_pos[i]}, {ee_err_rot[i]}")
                    print(f"score:{score}")
                    score[i] +=1
                    # reset
                    robots[i].target.apply_visual_materials(blinking_material)
                    for blink_step in range(50):
                        difuse_visual_material(blinking_material, NPColors.green, blink_step,T=3)
                        my_world.step(render=True)
                        

                    next_target_idx, next_target_color = Task.sample_target()
                    color_changing_meterial.set_color(getattr(NPColors, next_target_color))
                    Task.dst_pose = None
                    for j in range(len(include_robots)):
                        if not include_robots[j]:
                            continue
                        p_target, q_target = src_targets[j][next_target_idx]
                        # robots[j].target.set_world_pose(p_target, q_target)
                        robots[j].set_target_pose(p_target, q_target)
                        task_status[j] = 0 
                        # robots[j].target.apply_visual_material(color_changing_meterial)
                        robots[j].target.apply_visual_materials(color_changing_meterial)
                        
                    # my_world.step(render=True)
                    # pass
        # handle central targets if any robot is heading to the center
        if any(task_status) == 1:
            # make the target blink (visual only)
            difuse_visual_material(blinking_material, color_changing_meterial.get_color(), t_idx)
            
            # add some noise to the target, to prevent infinite "duals" (break ties)
            if t_idx % 100 == 0:
                p_dst_noisy, q_dst_noisy = copy.deepcopy(list(dst_targets[next_target_idx]))
                noise = np.random.uniform(-0.1,0.1,7).astype(np.float32)
                p_dst_noisy += noise[:3]
                q_dst_noisy += noise[3:]
                Task.dst_pose = (p_dst_noisy, q_dst_noisy)
                for robot_idx in range(len(include_robots)):
                    if not include_robots[robot_idx]:
                        continue
                    if task_status[robot_idx] == 1:
                        # robots[robot_idx].target.set_world_pose(*Task.dst_pose)
                        robots[robot_idx].set_target_pose(*Task.dst_pose)
  
        # visualiations
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

        if len(point_visualzer_inputs):
            visualizations_timer_start = time.time()
            draw_points(point_visualzer_inputs) # print_rate_decorator(lambda: draw_points(point_visualzer_inputs), args.print_ctrl_rate, "draw_points")()
            visualizations_timer += time.time() - visualizations_timer_start
        
        # update ctrl loop timer
        ctrl_loop_timer += time.time() - ctrl_loop_timer_start
        
        # print stats
        k_print = 100
        if t_idx % k_print == 0:
            ctrl_freq = k_print / ctrl_loop_timer
            print(f"t = {t_idx}")
            print(f"ctrl freq in last {k_print} steps:  {ctrl_freq:.2f}")
            print(f"robots as obs ops freq in last {k_print} steps: {k_print / robots_as_obs_timer:.2f}")
            print(f"env obs ops freq in last {k_print} steps: {k_print / env_obstacles_timer:.2f}")
            print(f"mpc solver freq in last {k_print} steps: {k_print / mpc_solver_timer:.2f}")
            print(f"world step freq in last {k_print} steps: {k_print / world_step_timer:.2f}")
            print(f"targets update freq in last {k_print} steps: {k_print / targets_update_timer:.2f}")
            print(f"joint states updates freq in last {k_print} steps: {k_print / joint_state_timer:.2f}")
            print(f"actions freq in last {k_print} steps: {k_print / action_timer:.2f}")
            print(f"visualization ops freq in last {k_print} steps: {k_print / visualizations_timer:.2f}")
            
            total_time_measured = mpc_solver_timer + world_step_timer + targets_update_timer + \
            joint_state_timer + action_timer + visualizations_timer + robots_as_obs_timer + env_obstacles_timer
            total_time_actual = ctrl_loop_timer
            delta = total_time_actual - total_time_measured
            print(f"total time actual: {total_time_actual:.2f}")
            print(f"total time measured: {total_time_measured:.2f}")
            print(f"delta: {delta:.2f}")
            print("In percentage %:")
            print(f"    mpc solver: {100 * mpc_solver_timer / total_time_actual:.2f}%")
            print(f"    world step: {100 * world_step_timer / total_time_actual:.2f}%")
            print(f"    robots as obs: {100 * robots_as_obs_timer / total_time_actual:.2f}%")
            print(f"    env obs: {100 * env_obstacles_timer / total_time_actual:.2f}%")
            print(f"    targets update: {100 * targets_update_timer / total_time_actual:.2f}%")
            print(f"    joint state: {100 * joint_state_timer / total_time_actual:.2f}%")
            print(f"    action: {100 * action_timer / total_time_actual:.2f}%")
            print(f"    visualizations: {100 * visualizations_timer / total_time_actual:.2f}%")

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

        # update ui window
        
        # general data
        timestep_label.text_block.text = f"Time Step: {t_idx}"
        control_freq.text_block.text = f"Control Freq: {ctrl_freq:.2f}"
        # task data
        ee_errs_prints = [f"({ee_err_pos[i]:.2f}, {ee_err_rot[i]:.2f})" for i in range(len(ee_err_pos))]
        ee_error_label.text_block.text = f"Pose Errors (pos, rot): {', '.join(ee_errs_prints)}"
        task_score.text_block.text = f"task scores: {score}"
        task_status_label.text_block.text = f"task status: {task_status}"
        # update time step
        t_idx += 1 

if __name__ == "__main__":
    if DEBUG_GPU_MEM:
        signal.signal(signal.SIGINT, handle_sigint_gpu_mem_debug) # register the signal handler for SIGINT (Ctrl+C) 
        torch.cuda.memory._record_memory_history() # https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
    main()
    simulation_app.close()
    
     
        

        
import os
from threading import Lock

from projects_root.utils.usd_utils import SimulationApp
os.environ.setdefault("MPLBACKEND", "Agg") # ?

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cfg", type=str, default="projects_root/experiments/benchmarks/experiments_cfg.yml", help="path to experiments configuration file")
args = parser.parse_args()

from curobo.util_file import  load_yaml
cfg = load_yaml(args.cfg)
SIMULATING = cfg["env"]["simulation"]["is_on"] # if False, then we are running the robot in real time (i.e. the robot will move as fast as the real time allows)
ENABLE_GPU_DYNAMICS = cfg["env"]["simulation"]["enable_gpu_dynamics"] # # GPU DYNAMICS - OPTIONAL (originally was disabled)# GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU. However, this will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. If enabling GPU dynamics slows down the simulation, it may be that your GPU is not able to handle the additional load. You can enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.# See: https://docs-prod.omniverse.nvidia.com/isaacsim/latest/reference_material/speedup_cheat_sheet.html?utm_source=chatgpt.com # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
RENDER_DT = cfg["env"]["simulation"]["render_dt"]
PHYSICS_STEP_DT = cfg["env"]["simulation"]["physics_dt"]

# CRITICAL: Isaac Sim must be imported FIRST before any other modules
try:
    import isaacsim
except ImportError:
    pass
from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics,make_world

# Init Isaac Sim app
simulation_app = init_app(cfg["env"]["simulation"]["init_app_settings"]) # SimulationApp
# Omniverse and IsaacSim modules
from omni.isaac.core import World

from dataclasses import dataclass
import time
from typing import Callable, Optional, Union
from omni.isaac.core import World
import omni.usd
import time
from typing import List
import torch
import os
import numpy as np
# Our modules
from projects_root.autonomous_arm import ArmMpc
from projects_root.utils.colors import npColors
# CuRobo modules
from curobo.geom.types import WorldConfig
from curobo.util.logger import setup_logger
setup_logger("warn") # curobo logger in warn mode
from curobo.util.usd_helper import UsdHelper
from projects_root.utils.world_model_wrapper import WorldModelWrapper
from threading import Thread, Event, Lock

# Prevent cuda out of memory errors. Backward competebility with curobo source code...
a = torch.zeros(4, device="cuda:0")

# @dataclass
# class EnvironmentConfig:
#     usd_file: str

# @dataclass
# class ChallengeConfig:
#     challenge_callback: Callable
#     metrics_collector: Callable

# @dataclass
# class ExperimentConfig:
#     environment: EnvironmentConfig
#     challenge: ChallengeConfig
#     timeout: float
#     pre_step_callback: Callable
#     post_step_callback: Callable

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
        collision_spheres_cfg = load_yaml(os.path.join('curobo/src/curobo/content/configs/robot', collision_spheres))
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
    print(sphere_count, extra_sphere_count)
    return sphere_count, extra_sphere_count

def init_ccheck_wcfg_in_sim(
        usd_help:UsdHelper, 
        robot_prim_path:str, 
        ignore_substrings:List[str])->WorldConfig:    
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








robot_cfg = load_yaml(experiment_cfg["robot"])["robot_cfg"]
auto_arm_cfg = experiment_cfg["auto_arm_cfg"]
debug_cfg = cfg["debug"]

visualize_spheres = debug_cfg["visualize_spheres"]["is_on"]
visualize_spheres_ts_delta = debug_cfg["visualize_spheres"]["ts_delta"]
# -----------------------------------------
# Scenario setup (simulation or real world)
# -----------------------------------------

X_robot = auto_arm_cfg["X_robot"]
X_target = auto_arm_cfg["X_target"]
target_color = npColors.red
X_robot_np = np.array(X_robot, dtype=np.float32)
X_target_R = list(np.array(X_target[:3]) - X_robot_np[:3]) + list(X_target[3:])  

# Calculate sphere counts for all robots BEFORE creating instances
_split = calculate_robot_sphere_count(robot_cfg) # split[0] = 'valid' (base, no extra), split[1] = extra
robot_sphere_counts = _split[0] + _split[1] # total (base + extra) sphere count (base + extra)
robot_sphere_counts_no_extra = _split[0] # valid (base only) sphere count (base only)

robot = ArmMpc(robot_cfg, my_world, usd_help, p_R=X_robot[:3], q_R=X_robot[3:],  p_T_R=np.array(X_target_R[:3]), q_T_R=np.array(X_target_R[3:]),  target_color=target_color, n_coll_spheres=robot_sphere_counts,  n_coll_spheres_valid=robot_sphere_counts_no_extra, use_col_pred=False)

if SIMULATING:
    # reset default prim to /World
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World")) # TODO: Try removing this and check if it breaks anything (if nothing breaks, remove it)    
    wait_for_playing(my_world, simulation_app, autoplay=True) 
if REAL:
    pass
    # TODO: needed?
    
# -----------------------------------------------------------------------------------------
# Publish "contexts" 
# (each robot context serves robot and other robots solver initialization)
# -----------------------------------------------------------------------------------------

# ----------------------------
# Initialize solvers
# ----------------------------
# Set robot in initial joint configuration (in curobo they call it  the "retract" config)
# print("debug")
print(robot.j_names)
_robot_idx_list = list(range(len(robot.j_names))) # [robot.robot.get_dof_index(x) for x in robot.j_names]
robot_idx_list = _robot_idx_list
assert _robot_idx_list is not None # Type assertion for linter    
if SIMULATING:
    robot.init_joints_in_sim(_robot_idx_list)
if REAL:
    pass 
    # TODO: INIT JOINT POSITIONS IN REAL WORLD
    # robot.init_joints(_robot_idx_list)


robot.init_solver(auto_arm_cfg["init_solver_cfg"])

# ----------------------------
# Start robot  loop
# ----------------------------
cu_world_never_add = ["/curobo", robot.target_prim_path ,robot.prim_path] # never treat these names as obstacles (add them to world model)
cu_world_never_update = ["/World/defaultGroundPlane"] # add here any substring of an onstacle you assume remains static throughout the entire simulation!



def async_robot_ctrl_loop_mpc_dec(
        robot_idx: int,
        stop_event: Event,
        get_t_idx,
        publish_plans:bool,
        t_lock: Lock,
        physx_lock: Lock,
        plans_lock: Lock,
        robots,
        col_pred_with,
        plans: List[Optional[Any]],
        tensor_args,
        cu_world_never_add:List[str],
        cu_world_never_update:List[str],
        usd_help:Optional[UsdHelper]=None,
        
):
    
    
        last_step = -1
        r: ArmMpc = robots[robot_idx]
        
        # Initialize collision check world configuration
        assert usd_help is not None # Type assertion for linter
        r.reset_wmw(init_ccheck_wcfg_in_sim(usd_help, r.prim_path, r.target_prim_path, cu_world_never_add)) 


        # Control loop
        while not stop_event.is_set():

            # wait for new time step
            with t_lock:
                cur_step = get_t_idx()
            if cur_step == last_step:
                time.sleep(1e-7)
                continue
            last_step = cur_step

            # publish 
            if publish_plans:
                sim_js = None
                real_js = None
                if SIMULATING:
                    sim_js = r.get_sim_joint_state(sync_new=False) # alreadysynced at sense()
                r.publish(plans, plans_lock,sim_js,real_js)
        
            # sense callbacks
            update_obs_callback = (r.update_obs_from_sim, {'usd_help':usd_help, 'ignore_list':cu_world_never_add + cu_world_never_update})
            update_target_callback = (r.update_target_from_sim, {})
            update_joint_state_callback = (r.update_cu_js_from_sim, {})
            # sense
            if r.use_col_pred:
                r.sense(update_obs_callback, update_target_callback, update_joint_state_callback, physx_lock, plans, col_pred_with[robot_idx], cur_step, tensor_args, robot_idx,plans_lock)
            else:
                r.sense(update_obs_callback, update_target_callback, update_joint_state_callback, physx_lock)

            # plan (our modified mpc planning, (torch-heavy, no PhysX))
            action = r.plan(max_attempts=2)
            
            # command 
            r.command(action, num_times=1, physx_lock=physx_lock)


class ExperimentConfig:
    def __init__(self,
                 num_robots:int,
                 actor_cfgs:list[dict],
                 auto_arm_cfg:dict,
                 debug_cfg:dict,
                 ):
        self.num_robots = num_robots
        self.robot_cfg = robot_cfg
        
class Simulation:
    def __init__(self, simulation_cfg):
        self.simulation_app = None
        self.stage:omni.usd.Stage
        self.usd_helper:UsdHelper
        self.world:World
        self.t_lock = Lock()
        self.physx_lock = Lock()
        
        self.reset_experiment(simulation_cfg['stage_file'], simulation_cfg['physics_dt'], simulation_cfg['render_dt'])
    
    def reset_experiment(self, stage_file:str, physics_dt:float, render_dt:float):
        # Initialize UsdHelper (helper for USD stage operations by curobo)
        # reset stage
        self.stage = omni.usd.get_context().new_stage() # clear all obstacles
        self.stage.DefinePrim("/curobo", "Xform")  # Transform for CuRobo-specific objects        
        # initialize usd helper and load stage file if provided
        self.usd_helper = UsdHelper()  
        self.usd_helper.load_stage(self.stage)
        if stage_file.endswith('.usd') or stage_file.endswith('.usda'):
            self.usd_helper.load_stage_from_file(stage_file) # set self.stage to the stage (self=usd_help)
        
        # initialize world        
        self.world = make_world(ground_plane=True, set_default_prim=True, to_Xform=True)        
        activate_gpu_dynamics(self.world)
        self.world.set_simulation_dt(physics_dt, render_dt)


class SimCuWorld:
    def __init__(self,
            cu_world_never_add:List[str]=["/curobo"],
            cu_world_never_update:List[str]=["/World/defaultGroundPlane"],
            ):
        self.cu_world_never_add = cu_world_never_add
        self.cu_world_never_update = cu_world_never_update    
    
    def init_ccheck_wcfg_in_sim(
        self,
        actor_prim_path:str,
        usd_help:UsdHelper, 
        )->WorldConfig:    
        """
        Make the initial collision check world configuration.
        This is the world configuration that will be used for collision checking.
        Note: must be collision check WorldConfig! not a regular WorldConfig.
        Also obstacles needs to be expressed in robot frame!
        """
        
        # Get obstacles from simulation and convert to WorldConfig (not yet collision check world WorldConfig!)
        cu_world_R = usd_help.get_obstacles_from_stage( 
            only_paths=['/World'], # look for obs only under the world prim path
            reference_prim_path=actor_prim_path, # obstacles are expressed in robot frame! (not world frame)
            ignore_substring=self.cu_world_never_add 
        )

        # Convert raw WorldConfig to collision check world WorldConfig! (Must!)
        cu_world_R = cu_world_R.get_collision_check_world()
        
        return cu_world_R


class Experiment:
    def __init__(self, n_actors):
        self.plans_lock = Lock()
        self.stop_event = Event()
        self.plans = [None for _ in range(n_actors)]

@dataclass
class SimActorCfg:
    broadcast_plans
    cu_world_never_add:List[str]
    cu_world_never_update:List[str]
    actor_prim_path: List[str]

class SimActor:
    def __init__(self,
                 cu_world_never_add,
                 cu_world_never_update,
                 actor_prim_path:str,
                 usd_help:UsdHelper,
                 target_ctrl_loop:Callable,
                 experiment:Experiment,
                 simulation:Simulation,
                 actor_cfg:ActorCfg,
                 ):
        
        
        # never treat these names as obstacles (add them to world model)
        self.sim_cu_world = SimCuWorld(cu_world_never_add,cu_world_never_update)
        self.sim_cu_world.init_ccheck_wcfg_in_sim(actor_prim_path, usd_help)
        
        self.thread = Thread(target=target_ctrl_loop, args=(simulation.t_lock, simulation.physx_lock, experiment.plans_lock,broadcast_plans))


 

# def launch_actor_threads():
    
#     # global simulation step index (# = world/sim clock ticks minus 1)
#     stop_event = Event()      # Event to signal robot threads to stop
#     t_lock = Lock()           # Protects access to shared t_idx
#     plans_lock = Lock()       # Protects access to shared plans list
#     physx_lock = Lock()       # Protects access to shared physx state (robot state, etc...)
#     plans: List[Optional[Any]] = [None for _ in range(len(robots))] # TODO: This is a hack to pass plans to the robot threads, but it is not the best approach so it'd be better to use ros topics or pass it as an argument when can, but it is the only one that works for now
    
#     cu_world_never_add = ["/curobo", *[robot.target_prim_path for robot in robots],*[robot.prim_path for robot in robots], "/World/ConveyorTrack", '/World/conveyor_cube'] # never treat these names as obstacles (add them to world model)
#     cu_world_never_update = ["/World/defaultGroundPlane", "/World/cv_approx"] # add here any substring of an onstacle you assume remains static throughout the entire simulation!
    
#     robot_threads = [Thread(target=async_ctrl_loop_robot,args=(idx, stop_event, lambda: t_idx, t_lock, physx_lock, plans_lock, robots, col_pred_with, plans, tensor_args, cu_world_never_add, cu_world_never_update, usd_help), daemon=True) for idx in range(len(robots))] # TODO: This is a hack to pass plans to the robot threads, but it is not the best approach so it'd be better to use ros topics or pass it as an argument when can, but it is the only one that works for now
#     for th in robot_threads:
#         th.start()

def async_sim():
    
    
    
    # point_visualzer_inputs = [] # empty list for draw_points() inputs
    step_batch_start_time = time.time()
    step_batch_size = 100
    if SIMULATING:
        assert simulation_app is not None # Type assertion for linter (it is not None when SIMULATING=True)
        while simulation_app.is_running():
            
            # wait_physx_start = time.time()
            with physx_lock: 
                # print(f"main-physx_lock_wait_time,{time.time() - wait_physx_start}")
                step_start_time = time.time()
                my_world.step(render=True)           
                step_end_time = time.time()
                print(f"main-world-step-time, {step_end_time - step_start_time}")
                        
            with t_lock:
                t_idx += 1
            
            if t_idx % step_batch_size == 0: # step batch size = 100
                step_batch_time = time.time() - step_batch_start_time
                print(f"ts: {t_idx}")
                print("num of actions planned by each robot:")
                print([robots[i].n_actions_planned for i in range(len(robots))])
                print(f"overall avg step time: {(step_batch_time/step_batch_size)*1000:.1f} ms")
                step_batch_start_time = time.time()
        
            
    # Clean up thread pool
    stop_event.set()
    for th in robot_threads:
        th.join()
    simulation_app.close() 
        

# Initialize collision check world configuration
robot.reset_wmw(init_ccheck_wcfg_in_sim(usd_help, robot.prim_path, robot.target_prim_path, cu_world_never_add)) 


def run_experiment(experiment_cfg):
    simulation = Simulation(experiment_cfg['simulation'], experiment_cfg['actors'])
    
    sim_actors = []
    for actor_cfg in experiment_cfg['actors']:
        sim_actors.append(SimActor(**actor_cfg))
    for sim_actor in sim_actors:
        sim_actor.start()




if __name__ == "__main__":

# Standard Library
main()



    
    


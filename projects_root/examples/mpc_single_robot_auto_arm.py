"""
async version of projects_root/examples/mpc_moving_obstacles_mpc_mpc.py
"""

# Force non-interactive matplotlib backend to avoid GUI operations from worker threads
import os
os.environ.setdefault("MPLBACKEND", "Agg") # ?

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mcfg", type=str, default="projects_root/projects/cfgs_by_robot/g1/mpc_single_arm/left_arm/meta_cfg.yml", help="path to meta-config-file configuration to load")
args = parser.parse_args()

from curobo.util_file import  load_yaml
meta_cfg = load_yaml(args.mcfg)


# Clean imports - no path setup needed!


SIMULATING = meta_cfg["env"]["simulation"]["is_on"] # if False, then we are running the robot in real time (i.e. the robot will move as fast as the real time allows)
REAL = meta_cfg["env"]["real"]["is_on"] # currently unsupported
ENABLE_GPU_DYNAMICS = meta_cfg["env"]["simulation"]["enable_gpu_dynamics"] # # GPU DYNAMICS - OPTIONAL (originally was disabled)# GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU. However, this will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. If enabling GPU dynamics slows down the simulation, it may be that your GPU is not able to handle the additional load. You can enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.# See: https://docs-prod.omniverse.nvidia.com/isaacsim/latest/reference_material/speedup_cheat_sheet.html?utm_source=chatgpt.com # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
RENDER_DT = meta_cfg["env"]["simulation"]["render_dt"]
PHYSICS_STEP_DT = meta_cfg["env"]["simulation"]["physics_dt"]

################### Imports and initiation ########################
if True: # imports and initiation (put it in an if statement to collapse it)
    
    if SIMULATING:
        # CRITICAL: Isaac Sim must be imported FIRST before any other modules
        try:
            import isaacsim
        except ImportError:
            pass
            
        from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics,make_world
        from projects_root.utils.usd_utils import load_usd_to_stage

        # Init Isaac Sim app
        simulation_app = init_app(meta_cfg["env"]["simulation"]["init_app_settings"]) # SimulationApp
        # Omniverse and IsaacSim modules
        from omni.isaac.core import World
        
        
        # Load USD to stage
        # stage = load_usd_to_stage("usd_collection/envs/cv_new.usd") # pxr.Usd.Stage
        
        # Init Isaac Sim world
        my_world:World = make_world(ground_plane=True, set_default_prim=True, to_Xform=True)
        
        # Enable GPU dynamics if needed
        if ENABLE_GPU_DYNAMICS:
            activate_gpu_dynamics(my_world)
        
        # Set simulation dt
        my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT)
        
        
        # from projects_root.utils.helper import add_extensions # available only after app initiation
        # add_extensions(simulation_app, meta_cfg["env"]["simulation"]["init_app_settings"]["headless"]) # in all of the examples of curobo it happens somwhere around here, before the simulation begins. I am not sure why, but I kept it as that. 
        # from omni.isaac.core.utils.physics import set_physics_threads
        

    if REAL:
        # TODO: figure out what we do when SIMULATING=False (should not be too hard)
        my_world = None
        usd_help = None
        stage = None
        tensor_args = None
        simulation_app = None
    
    # Third party modules (moved after Isaac Sim initialization)
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
    from curobo.util.usd_helper import UsdHelper
    
    from projects_root.utils.world_model_wrapper import WorldModelWrapper
    
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
    print("debug")
    print(sphere_count, extra_sphere_count)
    return sphere_count, extra_sphere_count




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


# def define_run_setup(meta_cfg):
#     X_robot = meta_cfg["auto_arm_cfg"]["X_robot"]
#     X_target = meta_cfg["auto_arm_cfg"]["X_target"]
#     target_colors = npColors.red  
#     X_robot_np = np.array(X_robot, dtype=np.float32)
#     X_target_R = list(np.array(X_target[:3]) - X_robot_np[:3]) + list(X_target[3:])  
#     return X_robot_np, X_target_R, target_colors




def main():

    # ------------
    # Curobo setup 
    # ------------
    setup_logger("warn") # curobo logger in warn mode
    if SIMULATING:   
        # Initialize UsdHelper (helper for USD stage operations by curobo)
        usd_help = UsdHelper()  
        stage = my_world.stage  # get the stage from the world
        usd_help.load_stage(stage) # set self.stage to the stage (self=usd_help)
        # set /World as Xform prim, and make it the default prim
        stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))
        # Make also /curobo as Xform prim
        _curobo_xform = stage.DefinePrim("/curobo", "Xform")  # Transform for CuRobo-specific objects
    
    if REAL:
        pass
        # TODO: init real world

    # ------------------
    # Config files setup
    # ------------------    
    
    robot_cfg = load_yaml(meta_cfg["robot"])["robot_cfg"]
    auto_arm_cfg = meta_cfg["auto_arm_cfg"]
    debug_cfg = meta_cfg["debug"]

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
    print("debug")
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
    
    # Initialize collision check world configuration
    if SIMULATING:
        assert usd_help is not None # Type assertion for linter
        robot.reset_wmw(init_ccheck_wcfg_in_sim(usd_help, robot.prim_path, robot.target_prim_path, cu_world_never_add)) 
    if REAL:
        robot.reset_wmw(init_ccheck_wcfg_in_real()) 

    if SIMULATING:
        t_idx = 0
        assert simulation_app is not None # Type assertion for linter (it is not None when SIMULATING=True)
        while simulation_app.is_running():   
            # sense
            robot.sense((robot.update_obs_from_sim, {'usd_help':usd_help, 'ignore_list':cu_world_never_add + cu_world_never_update}), (robot.update_target_from_sim, {}), (robot.update_cu_js_from_sim, {}))
            # plan
            action = robot.plan(max_attempts=2)    
            # command 
            robot.command(action, num_times=1)                    
            my_world.step(render=True)           
            t_idx += 1

            if visualize_spheres and t_idx % visualize_spheres_ts_delta == 0:
                robot.visualize_robot_as_spheres(robot.curobo_format_joints)
        simulation_app.close() 
        
    if REAL: # Real world (no simulation) # TODO add support for both or just one of them
        while True:
            # todo
            time.sleep(REAL_TIME_EXPECTED_CTRL_DT) # TODO: This is a hack to make the robot threads run at the same speed as the simulation, but it is not the best approach so it'd be better to use ros topics or pass it as an argument when can, but it is the only one that works for now
            



if __name__ == "__main__":

    # Standard Library
    main()
    
    
    
     
        


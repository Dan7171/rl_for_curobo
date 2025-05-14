
SIMULATING = True # if False, then we are running the robot in real time (i.e. the robot will move as fast as the real time allows)
REAL_TIME_EXPECTED_CTRL_DT = 0.03 #1 / (The expected control frequency in Hz). Set that to the avg time measurded between two consecutive calls to my_world.step() in real time. To print that time, use: print(f"Time between two consecutive calls to my_world.step() in real time, run with --print_ctrl_rate "True")
ENABLE_GPU_DYNAMICS = False # # GPU DYNAMICS - OPTIONAL (originally was disabled)# GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU. However, this will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. If enabling GPU dynamics slows down the simulation, it may be that your GPU is not able to handle the additional load. You can enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.# See: https://docs-prod.omniverse.nvidia.com/isaacsim/latest/reference_material/speedup_cheat_sheet.html?utm_source=chatgpt.com # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
VISUALIZE_PREDICTED_OBS_PATHS = True # If True, then the predicted paths of the dynamic obstacles will be rendered in the simulation.
VISUALIZE_ROBOT_COL_SPHERES = False # If True, then the robot collision spheres will be rendered in the simulation.
HIGHLIGHT_OBS = False # mark the predicted (or not predicted) dynamic obstacles in the simulation
DEBUG_GPU_MEM = False # If True, then the GPU memory usage will be printed on every call to my_world.step()
RENDER_DT = 0.03 # original 1/60. All details were moved to notes/all_dts_in_one_place_explained.txt
PHYSICS_STEP_DT = 0.03 # original 1/60. All details were moved to notes/all_dts_in_one_place_explained.txt
OBS_PREDICTION = True # same as MODIFY_MPC_COST_FN_FOR_DYNAMIC_OBS in past scripts using mpc...
DEBUG = True
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
    import threading
    # initialization of isaac sim
    from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics
    simulation_app = init_app({"headless": args.headless_mode is not None, "width": "1920", "height": "1080"}) # must happen before importing other isaac sim modules, or any other module which imports isaac sim modules.
    # optimization of isaac sim 
    # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
    # See: https://docs.isaacsim.omniverse.nvidia.com/latest/development_tools/carb_settings.html
    #import carb
    # settings = carb.settings.get_settings()
    # settings.set_int("/rtx/post/dlss/execMode", 0) # the most performant setting, reducing VRAM consumption and rendering time but decreasing render quality. This is the default value in Isaac Sim.
    # settings.set("rtx-transient/resourcemanager/texturestreaming/memoryBudget", 0.1) # educe the Texture Streaming Budget (0.6 is the default value))
    # settings.set("/rtx/rendermode", 0)  # Disable RTX
    # settings.set("/app/renderer/enableRayTracing", False)
    # # torch.set_default_dtype(torch.float32)

    # isaac sim modules
    from omni.isaac.core import World 
    from omni.isaac.core.utils.stage import add_reference_to_stage
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    # Our modules
    from projects_root.utils.helper import add_extensions
    from projects_root.autonomous_franka import FrankaCumotion
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
    a = torch.zeros(4, device="cuda:0") # prevent cuda out of memory errors (took from curobo examples)

    
######################### HELPER ##########################
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
    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType()  # Device configuration for tensor operations
    if ENABLE_GPU_DYNAMICS:
        activate_gpu_dynamics(my_world)
    
    # Adding two frankas to the scene
    # # Inspired by curobo/examples/isaac_sim/batch_motion_gen_reacher.py but this time at the same world (the batch motion gen reacher example is for multiple worlds)
    X_Robots = [
        np.array([0,0,0,1,0,0,0], dtype=np.float32),
        np.array([1.2,0,0,1,0,0,0], dtype=np.float32)
        ] # X_RobotOrigin (x,y,z,qw, qx,qy,qz) (expressed in world frame)
    n_robots = len(X_Robots)
    robots_cu_js: List[Optional[JointState]] =[None for _ in range(n_robots)]# for visualization of robot spheres
    robots_collision_caches = [{"obb": 100, "mesh": 100} for _ in range(n_robots)]
    robot_cfgs = [load_yaml(f"projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/franka{i}.yml")["robot_cfg"] for i in range(1,n_robots+1)]
    robot_idx_lists:List[Optional[List]] = [None for _ in range(n_robots)] 
    robot_world_models = [WorldConfig() for _ in range(n_robots)]
    X_Targets = [[0.6, 0, 0.2, 0, 1, 0, 0], [1.8, 0, 0.2, 0, 1, 0, 0]]# [[0.6, 0, 0.2, 0, 1, 0, 0] for _ in range(n_robots)]
    
    if OBS_PREDICTION:
        col_pred_with = [[1], []] # at each entry i, list of indices of robots that the ith robot will use for dynamic obs prediction
   
    collision_obstacles_cfg_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/collision_obstacles.yml"
    col_ob_cfg = load_yaml(collision_obstacles_cfg_path)
    env_obstacles = [] # list of obstacles in the world
    for obstacle in col_ob_cfg:
        obstacle = Obstacle(my_world, **obstacle)
        for i in range(len(robot_world_models)):
            if obstacle.cchecking_enabled:
                world_model_idx = obstacle.add_to_world_model(robot_world_models[i], X_Robots[i])#  usd_helper=usd_help) # inplace modification of the world model with the obstacle
                print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")            
        env_obstacles.append(obstacle) # add the obstacle to the list of obstacles
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    
    robots:List[FrankaCumotion] = []
    for i in range(n_robots):
        robots.append(FrankaCumotion(
            robot_cfgs[i], my_world, usd_help, 
            p_R=X_Robots[i][:3],q_R=X_Robots[i][3:], p_T=X_Targets[i][:3],
            q_T=X_Targets[i][3:], target_color=np.array([0,0.5,0]),
            dilation_factor=None,
            num_ik_seeds=500,
            evaluate_interpolated_trajectory=True
            # num_trajopt_seeds=12
            )
        )
    
    add_extensions(simulation_app, args.headless_mode) # in all of the examples of curobo it happens somwhere around here, before the simulation begins. I am not sure why, but I kept it as that. 
    wait_for_playing(my_world, simulation_app,args.autoplay) # wait for the play button to be pressed
    
    ################# SIM IS PLAYING ###################    
    # dynamic_obs_coll_predictors:List[DynamicObsCollPredictor] = []
    ccheckers = []
    # expected_ctrl_freq_at_mpc = 1 / step_dt_traj_mpc # This is what the mpc "thinks" the control frequency should be. It uses that to generate the rollouts.                
    
    # col_preds = []
    for i, robot in enumerate(robots):
        # Set robots in initial joint configuration (in curobo they call it  the "retract" config)
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot.init_joints(robot_idx_lists[i])
        
        # init dynamic obs coll predictors
        if OBS_PREDICTION:
            obs_groups_nspheres = [robots[obs_robot_idx].get_num_of_sphers() for obs_robot_idx in col_pred_with[i]]
            if len(col_pred_with[i]): # if we set to the robot at least one dynamic obstacle group to account for  
                robot.init_col_predictor(obs_groups_nspheres, cost_weight=10000, manually_express_p_own_in_world_frame=True)
        
        # init solver
        robot.init_solver(robot_world_models[i],robots_collision_caches[i], DEBUG)
        checker = robots[i].get_cchecker() # available only after init_solver
        ccheckers.append(checker)
        robots[i].robot._articulation_view.initialize() # new (isac 4.5) https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207

    # register ccheckers for environment obstacles
    for i in range(len(env_obstacles)):
        if env_obstacles[i].cchecking_enabled:
            env_obstacles[i].register_ccheckers(ccheckers)

    # initialize (activate) dynamic obs col predictors, each with its own obstacles       
    robots_spheres_s0 = [robot.get_current_spheres_state() for robot in robots]
    for i in range(n_robots):
        if hasattr(robots[i], 'dynamic_obs_col_pred'): 
            col_pred = robots[i].dynamic_obs_col_pred
            rad_spheresColwithS0 = torch.concat([robots_spheres_s0[other][1] for other in col_pred_with[i]])
            col_pred.set_obs_rads(rad_spheresColwithS0)
        
    t_idx = 0 # time step index in real world (not simulation) steps. This is the num of completed control steps (actions) in *played* simulation (after play button is pressed)
    ctrl_loop_start_time = time.time()
    
    while simulation_app.is_running():                 
        point_visualzer_inputs = []
        my_world.step(render=True)  
        # update obstacles poses in registed ccheckers (for environment (shared) obstacles)
        for i in range(len(env_obstacles)): 
            env_obstacles[i].update_registered_ccheckers()

        
        for i in range(n_robots):
            # get joint state
            sim_js = robots[i].get_sim_joint_state() # robot2.robot.get_joints_state() # reading current joint state from robot
            robots_cu_js[i] = robots[i].get_curobo_joint_state(sim_js) 
            if robots_cu_js[i] is None:
                continue
            # get real (most updated) position of target
            p_T, q_T = robots[i].target.get_world_pose() # print_rate_decorator(lambda: , args.print_ctrl_rate, "target.get_world_pose")() # goal pose        

            # update target and replan if needed
            if robots[i].set_new_target_for_solver(p_T, q_T, sim_js):
                robots[i].reset_command_plan(robots_cu_js[i]) # replanning a new global plan and setting robot2.cmd_plan to point the new plan.
        
        robots_plans_to_go = [robot.get_plan(plan_stage='optimized') for robot in robots] # arr[i] is None if no plan for robot i
        for i in range(n_robots):

            # update dynamic obs col predictor
            if hasattr(robots[i], 'dynamic_obs_col_pred'):
                col_pred = robots[i].dynamic_obs_col_pred 
                obs_groups_to_update = []
                p_obs_groups = []
                for j, obs_group_id in enumerate(col_pred_with[i]): # for each obstacle group j that the ith robot will use for dynamic obs prediction
                    if robots_plans_to_go[obs_group_id] is not None: # if obstacle group j has actions to go
                        obs_groups_to_update.append(j) # mark the jth obstacle group in the groups which are going to be updated
                        p_obs_group = robots_plans_to_go[obs_group_id][0] # spheres positions of the "plan to go" for the jth obstacle group
                        # if the plan to go is shorter than the robot horizon, repeat the last step of the plan to go
                        n_steps_obs_group_plan = p_obs_group.shape[0]
                        robot_horizon = robots[i].get_trajopt_horizon() # works for both mpc and cumotion
                        tail_len = robot_horizon - n_steps_obs_group_plan
                        if tail_len > 0:
                            tail = p_obs_group[-1].unsqueeze(0).repeat(tail_len, 1, 1) # repeat the last step of the obstacle plan
                            p_obs_group = torch.cat([p_obs_group, tail],dim=0) # now we have a prediction for H steps ahead as we wanted
                        p_obs_groups.append(p_obs_group)
                        if VISUALIZE_PREDICTED_OBS_PATHS:
                            global_plan_points = {'points': p_obs_group, 'color': 'green'}
                            point_visualzer_inputs.append(global_plan_points)       
                col_pred.update_obs_groups(p_obs_groups, obs_groups_to_update)


        for i in range(n_robots):
            # execute the next action in the plan if plan is not over
            if robots[i].cmd_plan is not None:
                art_action = robots[i].get_next_articulation_action(idx_list=robot_idx_lists[i])
                robots[i].apply_articulation_action(art_action)

                

        t_idx += 1 # num of completed control steps (actions) in *played* simulation (aft
        
        if t_idx % 100 == 0:
            print("t = ", t_idx)
            ctrl_loop_freq = t_idx / (time.time() - ctrl_loop_start_time) 
            print(f"Control loop frequency [HZ] = {ctrl_loop_freq}")

        if len(point_visualzer_inputs):
            draw_points(point_visualzer_inputs) # print_rate_decorator(lambda: draw_points(point_visualzer_inputs), args.print_ctrl_rate, "draw_points")() 
 
if __name__ == "__main__":
    if DEBUG_GPU_MEM:
        signal.signal(signal.SIGINT, handle_sigint_gpu_mem_debug) # register the signal handler for SIGINT (Ctrl+C) 
        torch.cuda.memory._record_memory_history() # https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
    main()
    simulation_app.close()
    
     
        

        
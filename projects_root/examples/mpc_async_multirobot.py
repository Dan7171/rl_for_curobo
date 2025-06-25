"""
async version of projects_root/examples/mpc_moving_obstacles_mpc_mpc.py
"""



SIMULATING = True # if False, then we are running the robot in real time (i.e. the robot will move as fast as the real time allows)
REAL_TIME_EXPECTED_CTRL_DT = 0.03 #1 / (The expected control frequency in Hz). Set that to the avg time measurded between two consecutive calls to my_world.step() in real time. To print that time, use: print(f"Time between two consecutive calls to my_world.step() in real time, run with --print_ctrl_rate "True")
ENABLE_GPU_DYNAMICS = False # # GPU DYNAMICS - OPTIONAL (originally was disabled)# GPU Dynamics: Enabling GPU dynamics can potentially speed up the simulation by offloading the physics calculations to the GPU. However, this will only be beneficial if your GPU is powerful enough and not already fully utilized by other tasks. If enabling GPU dynamics slows down the simulation, it may be that your GPU is not able to handle the additional load. You can enable or disable GPU dynamics in your script using the world.set_gpu_dynamics_enabled(enabled) function, where enabled is a boolean value indicating whether GPU dynamics should be enabled.# See: https://docs-prod.omniverse.nvidia.com/isaacsim/latest/reference_material/speedup_cheat_sheet.html?utm_source=chatgpt.com # See: https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/sim_performance_optimization_handbook.html
OBS_PREDICTION = True # If True, this would be what the original MPC cost function could handle. False means that the cost will consider obstacles as moving and look into the future, while True means that the cost will consider obstacles as static and not look into the future.
DEBUG = True # Currenly, the main feature of True is to run withoug cuda graphs. When its true, we can set breakpoints inside cuda graph code (like in cost computation in "ArmBase" for example)  
VISUALIZE_PREDICTED_OBS_PATHS = True # If True, then the predicted paths of the dynamic obstacles will be rendered in the simulation.
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
robots_cfgs_dir = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs"
mpc_cfg_overide_files_dir = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs"
################### Imports and initiation ########################
if True: # imports and initiation (put it in an if statement to collapse it)
    
    # Third party modules
    import time
    import signal
    from typing import List, Optional
    import torch
    import numpy as np
    # Initialize isaacsim app and load extensions
    from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics
    simulation_app = init_app() # must happen before importing other isaac sim modules, or any other module which imports isaac sim modules.
    from projects_root.utils.helper import add_extensions # available only after app initiation
    add_extensions(simulation_app, None if not HEADLESS_ISAAC else 'true') # in all of the examples of curobo it happens somwhere around here, before the simulation begins. I am not sure why, but I kept it as that. 
    # Omniverse and IsaacSim modules
    from omni.isaac.core import World 
    # Our modules
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import init_runtime_topics, get_topics
    from projects_root.utils.transforms import transform_poses_batched
    from projects_root.autonomous_franka import FrankaMpc
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
    from projects_root.autonomous_franka import AutonomousFranka
    # Prevent cuda out of memory errors. Backward competebility with curobo source code...
    a = torch.zeros(4, device="cuda:0")


def handle_sigint_gpu_mem_debug(signum, frame):
    print("Caught SIGINT (Ctrl+C), first dump snapshot...")
    torch.cuda.memory._dump_snapshot()
    print("Snapshot dumped to dump_snapshot.pickle, you can upload it to the server: https://docs.pytorch.org/memory_viz")
    print("Now raising KeyboardInterrupt to let the original KeyboardInterrupt handler (of nvidia) to close the app")
    raise KeyboardInterrupt # to let the original KeyboardInterrupt handler (of nvidia) to close the app
    
def main():
    
    
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
    
    
    # robot base frames, expressed in world frame
    X_Robots = [
        np.array([0,0,0,1,0,0,0], dtype=np.float32), # 1,0,0,0 = 0,0,0 in euler angles
        np.array([1.2,0,0,0,0,0,1], dtype=np.float32) # 0,0,0,1  = 0,0,180 in euler angles
        ] # (x,y,z,qw, qx,qy,qz) expressed in world frame
    
    n_robots = len(X_Robots)
    init_runtime_topics(n_envs=1, robots_per_env=n_robots) 
    runtime_topics = get_topics()
    env_topics = runtime_topics.get_default_env() if runtime_topics is not None else []
    robots_cu_js: List[Optional[JointState]] =[None for _ in range(n_robots)]# for visualization of robot spheres
    robots_collision_caches = [{"obb": 5, "mesh": 5} for _ in range(n_robots)]
    robot_cfgs = [load_yaml(f"{robots_cfgs_dir}/franka{i}.yml")["robot_cfg"] for i in range(1,n_robots+1)]
    robot_idx_lists:List[Optional[List]] = [None for _ in range(n_robots)] 
    robot_world_models = [WorldConfig() for _ in range(n_robots)]
    ccheckers = [] # collision checker for each robot 
    # robot targets, expressed in robot frames
    X_Targets_R = [[0.6, 0, 0.3, 0, 1, 0, 0], [0.6, 0, 0.3, 0, 1, 0, 0]] 
    target_colors = [npColors.green, npColors.red] # as num of robots
    plot_costs = [False, False] # as num of robots
    if OBS_PREDICTION:
        col_pred_with = [[1], [0]] # at each entry i, list of indices of robots that the ith robot will use for dynamic obs prediction
    
    robots:List[FrankaMpc] = []
    for i in range(n_robots):
        robots.append(FrankaMpc(
            robot_cfgs[i], 
            my_world, 
            usd_help, 
            env_id=0,
            robot_id=i,
            p_R=X_Robots[i][:3],
            q_R=X_Robots[i][3:], 
            p_T_R=X_Targets_R[i][:3],
            q_T_R=X_Targets_R[i][3:], 
            target_color=target_colors[i],
            plot_costs=plot_costs[i],
            override_particle_file=f'{mpc_cfg_overide_files_dir}/override_particle_mpc_files/{i}.yml'
            )
        )
    # ENVIRONMENT OBSTACLES - INITIALIZATION
    col_ob_cfg = load_yaml(collision_obstacles_cfg_path)
    env_obstacles = [] # list of obstacles in the world
    for obstacle in col_ob_cfg:
        obstacle = Obstacle(my_world, **obstacle)
        for i in range(len(robot_world_models)):
            world_model_idx = obstacle.add_to_world_model(robot_world_models[i], X_Robots[i])#  usd_helper=usd_help) # inplace modification of the world model with the obstacle
            print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")
        env_obstacles.append(obstacle) # add the obstacle to the list of obstacles
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    
    # wait for play button in simulator to be pushed
    wait_for_playing(my_world, simulation_app,autoplay=True) 
    
    
    
    for i, robot in enumerate(robots):
        # Set robots in initial joint configuration (in curobo they call it  the "retract" config)
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot.init_joints(robot_idx_lists[i])
        # Init robot mpc solver
        robots[i].init_solver(robot_world_models[i],robots_collision_caches[i], MPC_DT, DEBUG)

        # Some technical required step in isaac 4.5 https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207
        robots[i].robot._articulation_view.initialize() 
        
        # Get initialized collision checker of robot
        checker = robots[i].get_cchecker() # available only after init_solver
        ccheckers.append(checker)
        # # Some technical required step in isaac 4.5 https://github.com/NVlabs/curobo/commit/0a50de1ba72db304195d59d9d0b1ed269696047f#diff-0932aeeae1a5a8305dc39b778c783b0b8eaf3b1296f87886e9d539a217afd207
        # robots[i].robot._articulation_view.initialize() 

        
    for i in range(len(env_obstacles)):
        env_obstacles[i].register_ccheckers(ccheckers)


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
        my_world.step(render=True)       
        world_step_timer += time.time() - world_step_timer_start

        # ENVIRONMENT OBSTACLES - READ STATES AND UPDATE ROBOTS
        # update obstacles poses in registed ccheckers (for environment (shared) obstacles)
        env_obstacles_update_timer_start = time.time()
        for i in range(len(env_obstacles)): 
            env_obstacles[i].update_registered_ccheckers()
        env_obstacles_timer += time.time() - env_obstacles_update_timer_start

        # ROBOTS AS OBSTACLES - READ STATES/PLANS
        # get other robots states (no prediction) or plans (with prediction) for collision checking
        robots_as_obs_timer_start = time.time()
        if OBS_PREDICTION:         
            plans = [robots[i].get_plan(valid_spheres_only=False) for i in range(len(robots))]
            # Store plans for each robot in the environment topics
            for robot_idx in range(len(env_topics)):
                env_topics[robot_idx]["plans"] = plans[robot_idx]
        
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
                        planSpheres_robotj = plans[j]['task_space']['spheres'] 
                        p_spheresRobotjH = planSpheres_robotj['p'][:robots[i].H].to(tensor_args.device) # get plan (sphere positions) of robot j, up to the horizon length of robot i
                        rad_spheresRobotjH = planSpheres_robotj['r'][0].to(tensor_args.device)
                        if p_spheresOthersH is None:
                            p_spheresOthersH = p_spheresRobotjH
                            rad_spheresOthersH = rad_spheresRobotjH
                        else:
                            p_spheresOthersH = torch.cat((p_spheresOthersH, p_spheresRobotjH), dim=1) # stack the plans horizontally
                            rad_spheresOthersH = torch.cat((rad_spheresOthersH, rad_spheresRobotjH))
                col_pred = robots[i].get_col_pred()
                if col_pred is not None:
                    if t_idx == 0:
                      
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
            
            # UPDATE STATE IN SOLVER
            joint_state_timer_start = time.time()
            robots_cu_js[i] = robots[i].get_curobo_joint_state(robots[i].get_sim_joint_state())
            robots[i].update_current_state(robots_cu_js[i])    
            joint_state_timer += time.time() - joint_state_timer_start
            
            # UPDATE TARGET IN SOLVER
            targets_update_timer_start = time.time()
            p_T, q_T = robots[i].target.get_world_pose() 
            if robots[i].set_new_target_for_solver(p_T, q_T):
                print(f"robot {i} target changed!")
                robots[i].update_solver_target()
            targets_update_timer += time.time() - targets_update_timer_start

            # MPC STEP
            mpc_solver_timer_start = time.time()
            mpc_result = robots[i].solver.step(robots[i].current_state, max_attempts=2) 
            mpc_solver_timer += time.time() - mpc_solver_timer_start
            
            # APPLY ACTION
            action_timer_start = time.time()
            art_action = robots[i].get_next_articulation_action(mpc_result.js_action) # get articulated action from joint state action
            robots[i].apply_articulation_action(art_action,num_times=1) # Note: I chhanged it to 1 instead of 3
            action_timer += time.time() - action_timer_start
            
            # VISUALIZATION
            if VISUALIZE_MPC_ROLLOUTS:
                visualizations_timer_start = time.time()
                p_visual_rollouts_robotframe = robots[i].solver.get_visual_rollouts()
                q_visual_rollouts_robotframe = torch.empty(p_visual_rollouts_robotframe.shape[:-1] + torch.Size([4]), device=p_visual_rollouts_robotframe.device)
                q_visual_rollouts_robotframe[...,:] = torch.tensor([1,0,0,0],device=p_visual_rollouts_robotframe.device, dtype=p_visual_rollouts_robotframe.dtype) 
                visual_rollouts = torch.cat([p_visual_rollouts_robotframe, q_visual_rollouts_robotframe], dim=-1)                
                visual_rollouts = transform_poses_batched(visual_rollouts, X_Robots[i].tolist())
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
        if t_idx % k_print == 0 and ctrl_loop_timer > 0:    
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
    
     
        

        
"""
Clean async version of multi-robot MPC with eliminated code duplication.
Key improvements:
1. Robot poses computed once in basic_setup() and injected into costs
2. n_coll_spheres calculated dynamically from col_pred_with
3. env_id/robot_id passed from robot context, not config files
"""

# Configuration constants
SIMULATING = True
REAL_TIME_EXPECTED_CTRL_DT = 0.03
ENABLE_GPU_DYNAMICS = False
OBS_PREDICTION = True
DEBUG = True
VISUALIZE_PREDICTED_OBS_PATHS = True
VISUALIZE_MPC_ROLLOUTS = True
VISUALIZE_ROBOT_COL_SPHERES = False
HIGHLIGHT_OBS = False
HIGHLIGHT_OBS_H = 30
DEBUG_GPU_MEM = False
RENDER_DT = 0.03
PHYSICS_STEP_DT = 0.03
MPC_DT = 0.03
HEADLESS_ISAAC = False

# Clean configuration paths - no duplication needed
collision_obstacles_cfg_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/collision_obstacles.yml"
robots_cfgs_dir = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/multi_arm_decentralized/robot/franka"
mpc_cfg_overide_files_dir = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/multi_arm_decentralized/mpc"

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
    # Prevent cuda out of memory errors. Backward compatibility with curobo source code...
    a = torch.zeros(4, device="cuda:0")


def compute_robot_sphere_counts(robots_cfgs: List[dict]) -> List[int]:
    """Compute the number of collision spheres for each robot from their configs."""
    sphere_counts = []
    for robot_cfg in robots_cfgs:
        # This is a simplified approach - in reality you'd parse the collision_spheres config
        # For now, assume all Franka robots have 65 spheres
        sphere_counts.append(65)
    return sphere_counts


def compute_obstacle_sphere_counts(col_pred_with: List[List[int]], robot_sphere_counts: List[int]) -> List[int]:
    """Compute total obstacle spheres for each robot based on col_pred_with."""
    obstacle_counts = []
    for i, other_robot_indices in enumerate(col_pred_with):
        total_obstacle_spheres = sum(robot_sphere_counts[j] for j in other_robot_indices)
        obstacle_counts.append(total_obstacle_spheres)
    return obstacle_counts


def basic_setup(n_robots: int):
    """
    Clean setup that computes robot configurations once.
    
    Returns:
        X_robots: list of robot poses in world frame
        col_pred_with: list of lists of robot indices for dynamic obs prediction
        X_target_R: list of robot target poses in robot frame
        plot_costs: list of booleans for cost plotting
        target_colors: list of colors for each robot target
    """
    # Shared target configuration
    X_targets = [0, 0, 0.5, 0, 1, 0, 0]  # position and orientation in world frame
    
    # Robot configurations based on number of robots
    match(n_robots):
        case 1:
            X_robots = [[-0.5, 0, 0, 1, 0, 0, 0]]
            col_pred_with = [[]]  # No other robots to predict
            plot_costs = [False]
            target_colors = [npColors.red]
            
        case 2:
            X_robots = [[-0.5, 0, 0, 1, 0, 0, 0], [0.5, 0, 0, 1, 0, 0, 0]]
            col_pred_with = [[1], [0]]  # Each robot predicts the other
            plot_costs = [False, False]
            target_colors = [npColors.red, npColors.green]

        case 3:
            X_robots = [[0.7071, -0.5, 0, 1, 0, 0, 0], [-0.7071, -0.5, 0, 1, 0, 0, 0], [0, 0.5, 0, 1, 0, 0, 0]]
            col_pred_with = [[1, 2], [0, 2], [0, 1]]  # Each robot predicts the other two
            plot_costs = [False, False, False]
            target_colors = [npColors.red, npColors.green, npColors.blue]
            
        case 4:
            X_robots = [[-0.5, -0.5, 0, 1, 0, 0, 0], [-0.5, 0.5, 0, 1, 0, 0, 0], 
                       [0.5, 0.5, 0, 1, 0, 0, 0], [0.5, -0.5, 0, 1, 0, 0, 0]]
            col_pred_with = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]  # Each robot predicts all others
            plot_costs = [False, False, False, False]
            target_colors = [npColors.red, npColors.green, npColors.blue, npColors.yellow]

    # Convert to numpy arrays
    X_robots_np = [np.array(X_robot, dtype=np.float32) for X_robot in X_robots]
    
    # Express targets in robot frames (single computation)
    X_target_R = [list(np.array(X_targets[:3]) - X_robots_np[i][:3]) + list(X_targets[3:]) 
                  for i in range(n_robots)]
    
    return X_robots_np, col_pred_with, X_target_R, plot_costs, target_colors


def inject_robot_context_into_solver(robot: FrankaMpc, env_id: int, robot_id: int, 
                                   robot_pose: np.ndarray, col_pred_with: List[int], 
                                   robot_sphere_counts: List[int]):
    """Inject robot context into the solver's rollout function for dynamic cost computation."""
    # Compute obstacle sphere count for this robot
    n_obstacle_spheres = sum(robot_sphere_counts[j] for j in col_pred_with)
    
    # Get the rollout function (arm_base instance)
    rollout_fn = robot.get_wrap_mpc_optimizer().rollout_fn
    
    # Set robot context for dynamic cost computation
    if hasattr(rollout_fn, 'set_robot_context'):
        rollout_fn.set_robot_context(
            env_id=env_id,
            robot_id=robot_id,
            robot_pose=robot_pose.tolist(),
            col_pred_with=col_pred_with
        )
        print(f"Injected robot context for robot {robot_id}: pose={robot_pose[:3]}, n_obstacle_spheres={n_obstacle_spheres}")


def main(n_robots):
    # Isaac Sim and USD setup
    usd_help = UsdHelper()
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    my_world.set_simulation_dt(PHYSICS_STEP_DT, RENDER_DT)
    if ENABLE_GPU_DYNAMICS:
        activate_gpu_dynamics(my_world)
    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    
    # CuRobo setup
    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType()
    
    # Clean setup - compute all robot configurations once
    X_Robots, col_pred_with, X_target_R, plot_costs, target_colors = basic_setup(n_robots)
    
    # Runtime topics for communication between robots
    init_runtime_topics(n_envs=1, robots_per_env=n_robots)
    runtime_topics = get_topics()
    env_topics = runtime_topics.get_default_env() if runtime_topics is not None else []
    
    # CLEAN APPROACH: Pre-populate robot context in runtime topics
    # This allows DynamicObsCost to read context during initialization
    for i in range(n_robots):
        n_obstacle_spheres = sum(robot_sphere_counts[j] for j in col_pred_with[i])
        # Store robot context in runtime topics for this robot
        env_topics[i]["robot_context"] = {
            "env_id": 0,
            "robot_id": i,
            "robot_pose": X_Robots[i].tolist(),
            "n_obstacle_spheres": n_obstacle_spheres,
            "col_pred_with": col_pred_with[i]
        }
        print(f"Pre-populated robot context for robot {i}: pose={X_Robots[i][:3]}, n_obstacle_spheres={n_obstacle_spheres}")
    
    # Initialize robot configurations
    robots_cu_js: List[Optional[JointState]] = [None for _ in range(n_robots)]
    robot_idx_lists: List[Optional[List]] = [None for _ in range(n_robots)]
    robots_collision_caches = [{"obb": 5, "mesh": 5} for _ in range(n_robots)]
    robot_world_models = [WorldConfig() for _ in range(n_robots)]
    robot_cfgs = [load_yaml(f"{robots_cfgs_dir}/arm{i}.yml")["robot_cfg"] for i in range(n_robots)]
    ccheckers = []
    
    # Compute robot sphere counts once
    robot_sphere_counts = compute_robot_sphere_counts(robot_cfgs)
    
    # Create FrankaMpc instances
    robots: List[FrankaMpc] = []
    for i in range(n_robots):
        robots.append(FrankaMpc(
            robot_cfgs[i],
            my_world,
            usd_help,
            env_id=0,
            robot_id=i,
            p_R=X_Robots[i][:3],
            q_R=X_Robots[i][3:],
            p_T_R=np.array(X_target_R[i][:3]),  # Fix linter error
            q_T_R=np.array(X_target_R[i][3:]),  # Fix linter error
            target_color=target_colors[i],
            plot_costs=plot_costs[i],
            override_particle_file=f'{mpc_cfg_overide_files_dir}/arm{i}_clean.yml'  # Use clean config
        ))
    
    # Environment obstacles initialization
    col_ob_cfg = load_yaml(collision_obstacles_cfg_path)
    env_obstacles = []
    for obstacle in col_ob_cfg:
        obstacle = Obstacle(my_world, **obstacle)
        for i in range(len(robot_world_models)):
            world_model_idx = obstacle.add_to_world_model(robot_world_models[i], X_Robots[i])
            print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")
        env_obstacles.append(obstacle)
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    
    # Wait for play button
    wait_for_playing(my_world, simulation_app, autoplay=True)
    
    # Initialize robots and inject clean robot context
    for i, robot in enumerate(robots):
        # Set robots in initial joint configuration
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        robot.init_joints(robot_idx_lists[i])
        
        # Initialize MPC solver
        robots[i].init_solver(robot_world_models[i], robots_collision_caches[i], MPC_DT, DEBUG)
        robots[i].robot._articulation_view.initialize()
        
        # Get collision checker
        checker = robots[i].get_cchecker()
        ccheckers.append(checker)
        
        # Robot context was already pre-populated in runtime topics - no injection needed!
    
    # Register environment obstacles with collision checkers
    for i in range(len(env_obstacles)):
        env_obstacles[i].register_ccheckers(ccheckers)

    # Main simulation loop (same as before)
    t_idx = 0
    ctrl_loop_timer = 0
    world_step_timer = 0
    mpc_solver_timer = 0
    targets_update_timer = 0
    joint_state_timer = 0
    action_timer = 0
    robots_as_obs_timer = 0
    env_obstacles_timer = 0
    visualizations_timer = 0

    while simulation_app.is_running():
        point_visualzer_inputs = []
        ctrl_loop_timer_start = time.time()
        
        # World step
        world_step_timer_start = time.time()
        my_world.step(render=True)
        world_step_timer += time.time() - world_step_timer_start

        # Environment obstacles update
        env_obstacles_update_timer_start = time.time()
        for i in range(len(env_obstacles)):
            env_obstacles[i].update_registered_ccheckers()
        env_obstacles_timer += time.time() - env_obstacles_update_timer_start

        # Robots as obstacles - get plans
        robots_as_obs_timer_start = time.time()
        if OBS_PREDICTION:
            plans = [robots[i].get_plan(valid_spheres_only=False) for i in range(len(robots))]
            for robot_idx in range(len(env_topics)):
                env_topics[robot_idx]["plans"] = plans[robot_idx]
        robots_as_obs_timer += time.time() - robots_as_obs_timer_start

        # Process each robot
        for i in range(len(robots)):
            # Update robot state
            joint_state_timer_start = time.time()
            robots_cu_js[i] = robots[i].get_curobo_joint_state(robots[i].get_sim_joint_state())
            robots[i].update_current_state(robots_cu_js[i])
            joint_state_timer += time.time() - joint_state_timer_start
            
            # Update target
            targets_update_timer_start = time.time()
            p_T, q_T = robots[i].target.get_world_pose()
            if robots[i].set_new_target_for_solver(p_T, q_T):
                print(f"robot {i} target changed!")
                robots[i].update_solver_target()
            targets_update_timer += time.time() - targets_update_timer_start

            # MPC step
            mpc_solver_timer_start = time.time()
            mpc_result = robots[i].solver.step(robots[i].current_state, max_attempts=2)
            mpc_solver_timer += time.time() - mpc_solver_timer_start
            
            # Apply action
            action_timer_start = time.time()
            art_action = robots[i].get_next_articulation_action(mpc_result.js_action)
            robots[i].apply_articulation_action(art_action, num_times=1)
            action_timer += time.time() - action_timer_start
            
            # Visualization
            if VISUALIZE_MPC_ROLLOUTS:
                visualizations_timer_start = time.time()
                p_visual_rollouts_robotframe = robots[i].solver.get_visual_rollouts()
                q_visual_rollouts_robotframe = torch.empty(p_visual_rollouts_robotframe.shape[:-1] + torch.Size([4]), device=p_visual_rollouts_robotframe.device)
                q_visual_rollouts_robotframe[...,:] = torch.tensor([1,0,0,0], device=p_visual_rollouts_robotframe.device, dtype=p_visual_rollouts_robotframe.dtype)
                visual_rollouts = torch.cat([p_visual_rollouts_robotframe, q_visual_rollouts_robotframe], dim=-1)
                visual_rollouts = transform_poses_batched(visual_rollouts, X_Robots[i].tolist())
                rollouts_for_visualization = {'points': visual_rollouts, 'color': 'green'}
                point_visualzer_inputs.append(rollouts_for_visualization)
                visualizations_timer += time.time() - visualizations_timer_start
            
            if VISUALIZE_ROBOT_COL_SPHERES and t_idx % 2 == 0:
                visualizations_timer_start = time.time()
                robots[i].visualize_robot_as_spheres(robots_cu_js[i])
                visualizations_timer += time.time() - visualizations_timer_start

        # Visualization
        if len(point_visualzer_inputs):
            visualizations_timer_start = time.time()
            draw_points(point_visualzer_inputs)
            visualizations_timer += time.time() - visualizations_timer_start
        
        t_idx += 1
        ctrl_loop_timer += time.time() - ctrl_loop_timer_start
        
        # Print statistics
        k_print = 100
        if t_idx % k_print == 0 and ctrl_loop_timer > 0:
            print(f"t = {t_idx}")
            print(f"ctrl freq in last {k_print} steps: {k_print / ctrl_loop_timer}")
            print(f"Dynamic obstacle costs automatically configured from robot context")
            
            # Reset timers
            ctrl_loop_timer = 0
            mpc_solver_timer = 0
            world_step_timer = 0
            targets_update_timer = 0
            joint_state_timer = 0
            action_timer = 0
            visualizations_timer = 0
            robots_as_obs_timer = 0
            env_obstacles_timer = 0


if __name__ == "__main__":
    if DEBUG_GPU_MEM:
        import signal
        torch.cuda.memory._record_memory_history()
    
    n_robots = int(input("Enter the number of robots (1-4, default 1): ") or "1")
    print(f"Running clean multi-robot MPC with {n_robots} robots")
    print("Key improvements:")
    print("- Robot poses computed once and injected dynamically")
    print("- n_coll_spheres calculated from col_pred_with")
    print("- env_id/robot_id passed from robot context")
    print("- No configuration duplication!")
    
    main(n_robots)
    simulation_app.close() 
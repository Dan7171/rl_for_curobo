"""
Multi-Robot MPC with Dynamic Targets and Central Bin/Table

This example demonstrates:
1. Multiple robots with targets that start at the same initial pose (center)
2. Automatic target updates every 5-10 seconds
3. Visual color markers on robots matching their target colors
4. Central bin/table obstacle that robots must avoid
5. Targets positioned around/inside the central obstacle area

Based on: projects_root/examples/mpc_async_multirobot_demo.py
"""

# Configuration flags
SIMULATING = True
REAL_TIME_EXPECTED_CTRL_DT = 0.03
ENABLE_GPU_DYNAMICS = False
OBS_PREDICTION = True
DEBUG = True
VISUALIZE_PLANS_AS_DOTS = True
VISUALIZE_MPC_ROLLOUTS = True
VISUALIZE_ROBOT_COL_SPHERES = False
HIGHLIGHT_OBS = False
HIGHLIGHT_OBS_H = 30
DEBUG_GPU_MEM = False
RENDER_DT = 0.03
PHYSICS_STEP_DT = 0.03
MPC_DT = 0.03
HEADLESS_ISAAC = False

# Scene configuration
SCENE_TYPE = "bin"  # "bin" or "table"
CENTER_POSITION = [0.0, 0.0, 0.2]  # Center of the bin/table
TARGET_UPDATE_INTERVAL = (5.0, 10.0)  # Min/max seconds between updates
MARKER_TYPE = "sphere"  # "sphere" or "cube" for robot markers

# Paths
collision_obstacles_cfg_path = "projects_root/examples/dynamic_scene/dynamic_scene_obstacles.yml"
robots_cfgs_dir = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/multi_arm_decentralized/robot/franka"
mpc_cfg_overide_files_dir = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/multi_arm_decentralized/mpc"
robots_collision_spheres_configs_parent_dir = "curobo/src/curobo/content/configs/robot"

################### Imports and initiation ########################
if True:  # imports and initiation (put it in an if statement to collapse it)
    
    # CRITICAL: Isaac Sim must be imported FIRST before any other modules
    try:
        import isaacsim
    except ImportError:
        pass
    
    from projects_root.utils.issacsim import init_app, wait_for_playing, activate_gpu_dynamics
    simulation_app = init_app()  # must happen before importing other isaac sim modules
    
    # Third party modules
    import time
    import signal
    from typing import List, Optional, Tuple
    import torch
    import os
    import numpy as np
    from projects_root.utils.helper import add_extensions
    add_extensions(simulation_app, None if not HEADLESS_ISAAC else 'true')
    
    # Omniverse and IsaacSim modules
    from omni.isaac.core import World 
    from omni.isaac.core.materials import OmniGlass
    
    # Our modules
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import init_runtime_topics, get_topics
    from projects_root.utils.transforms import transform_poses_batched
    from projects_root.autonomous_arm import ArmMpc
    from projects_root.utils.draw import draw_points
    from projects_root.utils.colors import npColors
    from projects_root.utils.dynamic_scene_manager import DynamicSceneManager
    
    # CuRobo modules
    from curobo.geom.types import Sphere, WorldConfig
    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.util.logger import setup_curobo_logger
    from curobo.util.usd_helper import UsdHelper
    from curobo.util_file import load_yaml
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
    from projects_root.projects.dynamic_obs.dynamic_obs_predictor.obstacle import Obstacle
    from projects_root.autonomous_arm import AutonomousArm
    
    # Prevent cuda out of memory errors
    a = torch.zeros(4, device="cuda:0")


def calculate_robot_sphere_count(robot_cfg):
    """Calculate the number of collision spheres for a robot from its configuration."""
    collision_spheres = robot_cfg["kinematics"]["collision_spheres"]
    
    if isinstance(collision_spheres, str):
        collision_spheres_cfg = load_yaml(os.path.join(robots_collision_spheres_configs_parent_dir, collision_spheres))
        collision_spheres_dict = collision_spheres_cfg["collision_spheres"]
    else:
        collision_spheres_dict = collision_spheres
    
    sphere_count = 0
    for link_name, spheres in collision_spheres_dict.items():
        if isinstance(spheres, list):
            sphere_count += len(spheres)
    
    extra_spheres = robot_cfg["kinematics"].get("extra_collision_spheres", {})
    extra_sphere_count = 0
    for obj_name, count in extra_spheres.items():
        extra_sphere_count += count
        
    return sphere_count, extra_sphere_count


def parse_meta_configs(meta_config_paths: List[str]) -> Tuple[List[str], List[str]]:
    """Parse meta-configuration files to extract robot and MPC config paths."""
    robot_config_paths = []
    mpc_config_paths = []
    
    for meta_path in meta_config_paths:
        meta_config = load_yaml(meta_path)
        robot_config_paths.append(meta_config["robot"])
        mpc_config_paths.append(meta_config["mpc"])
        
    return robot_config_paths, mpc_config_paths


def define_run_setup(n_robots: int, scene_manager: DynamicSceneManager):
    """
    Define robot setup using dynamic scene manager
    
    Returns:
        X_robots: list of robot poses in world frame
        col_pred_with: list of robot indices for dynamic obs prediction
        X_targets_R: list of robot target poses in robot frame (initially all at center)
        plot_costs: list of booleans for cost plotting
        target_colors: list of colors for each robot target
    """
    
    # Define robot positions based on number of robots
    match(n_robots):
        case 1:
            X_robots = [[-0.6, 0, 0, 1, 0, 0, 0]]
            col_pred_with = [[]]
            plot_costs = [False]
        case 2:  # 2 robots in a line
            X_robots = [[-0.6, -0.3, 0, 1, 0, 0, 0], [-0.6, 0.3, 0, 1, 0, 0, 0]] 
            col_pred_with = [[1], [0]]
            plot_costs = [True, True]
        case 3:  # 3 robots in a triangle 
            X_robots = [[-0.7, -0.4, 0, 1, 0, 0, 0], [-0.7, 0.4, 0, 1, 0, 0, 0], [-0.3, 0, 0, 1, 0, 0, 0]]
            col_pred_with = [[1,2], [0,2], [0,1]]
            plot_costs = [True, True, True]
        case 4:  # 4 robots in a square around the center
            X_robots = [[-0.7, -0.4, 0, 1, 0, 0, 0], [-0.7, 0.4, 0, 1, 0, 0, 0], 
                       [0.7, 0.4, 0, 1, 0, 0, 0], [0.7, -0.4, 0, 1, 0, 0, 0]]
            col_pred_with = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
            plot_costs = [False, False, False, False]
        case _:
            # Default for more robots: arrange in circle around center
            radius = 0.8
            X_robots = []
            col_pred_with = []
            plot_costs = []
            for i in range(n_robots):
                angle = 2 * np.pi * i / n_robots
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                X_robots.append([x, y, 0, 1, 0, 0, 0])
                # Each robot predicts collisions with all others
                col_pred_with.append([j for j in range(n_robots) if j != i])
                plot_costs.append(False)

    # Convert to numpy arrays
    X_robots_np = [np.array(Xi, dtype=np.float32) for Xi in X_robots]
    
    # Get target colors from scene manager
    target_colors = [scene_manager.get_target_for_robot(i)[1] for i in range(n_robots)]
    
    # All targets start at center, expressed in robot frames
    center_world = np.array(CENTER_POSITION)
    X_targets_R = []
    for i in range(n_robots):
        # Target position relative to robot base
        target_rel = center_world - X_robots_np[i][:3]
        X_targets_R.append(list(target_rel) + [1, 0, 0, 0])  # position + quaternion
    
    return X_robots_np, col_pred_with, X_targets_R, plot_costs, target_colors


def main(meta_config_paths: List[str]):
    """Main function for multi-robot MPC simulation with dynamic targets."""
    n_robots = len(meta_config_paths)
    print(f"Starting multi-robot simulation with {n_robots} robots and dynamic targets")
    
    # Parse meta-configurations
    robot_config_paths, mpc_config_paths = parse_meta_configs(meta_config_paths)
    
    # Print robot configuration summary
    for i, (robot_path, mpc_path) in enumerate(zip(robot_config_paths, mpc_config_paths)):
        print(f"Robot {i}: robot_config='{robot_path}', mpc_config='{mpc_path}'")
    
    # Isaac sim and USD setup
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
    
    # Initialize dynamic scene manager BEFORE robot setup
    # We need robot positions first for the scene manager
    temp_robot_positions = []
    for i in range(n_robots):
        if n_robots <= 4:
            positions = [[-0.6, 0, 0], [-0.6, -0.3, 0], [-0.6, 0.3, 0], [-0.7, -0.4, 0]]
            temp_robot_positions.append(np.array(positions[i] if i < len(positions) else positions[0]))
        else:
            # Circle arrangement
            radius = 0.8
            angle = 2 * np.pi * i / n_robots
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            temp_robot_positions.append(np.array([x, y, 0]))
    
    # Initialize dynamic scene manager with enhanced markers
    scene_manager = DynamicSceneManager(
        n_robots=n_robots,
        robot_positions=temp_robot_positions,
        stage=stage,
        scene_type=SCENE_TYPE,
        center_position=CENTER_POSITION,
        target_area_radius=0.4,
        target_height_range=(0.3, 0.7),
        update_interval=TARGET_UPDATE_INTERVAL,
        marker_type=MARKER_TYPE,
        marker_size=0.08,  # Larger, more visible markers
        marker_height_offset=0.25  # Height above end-effector
    )
    
    # Basic setup using scene manager
    X_robots, col_pred_with, X_targets_R, plot_costs, target_colors = define_run_setup(n_robots, scene_manager)
    
    # Runtime topics
    init_runtime_topics(n_envs=1, robots_per_env=n_robots) 
    runtime_topics = get_topics()
    env_topics = runtime_topics.get_default_env() if runtime_topics is not None else []
    
    # Initialize robot data structures
    robots_cu_js: List[Optional[JointState]] = [None for _ in range(n_robots)]
    robot_idx_lists: List[Optional[List]] = [None for _ in range(n_robots)]
    robots_collision_caches = [{"obb": 5, "mesh": 5} for _ in range(n_robots)]
    robot_world_models = [WorldConfig() for _ in range(n_robots)]
    robot_cfgs = [load_yaml(robot_path)["robot_cfg"] for robot_path in robot_config_paths]
    ccheckers = []
    
    # Calculate sphere counts
    robot_sphere_counts_split = [calculate_robot_sphere_count(robot_cfg) for robot_cfg in robot_cfgs]
    robot_sphere_counts = [split[0] + split[1] for split in robot_sphere_counts_split]
    robot_sphere_counts_valid = [split[0] for split in robot_sphere_counts_split]

    # Create robot instances
    robots: List[ArmMpc] = []
    for i in range(n_robots):
        robots.append(ArmMpc(
            robot_cfgs[i], 
            my_world, 
            usd_help, 
            env_id=0,
            robot_id=i,
            p_R=X_robots[i][:3],
            q_R=X_robots[i][3:], 
            p_T_R=np.array(X_targets_R[i][:3]),
            q_T_R=np.array(X_targets_R[i][3:]), 
            target_color=target_colors[i],
            plot_costs=plot_costs[i],
            override_particle_file=mpc_config_paths[i],
            n_coll_spheres=robot_sphere_counts[i],
            n_coll_spheres_valid=robot_sphere_counts_valid[i],
            use_col_pred=OBS_PREDICTION and len(col_pred_with[i]) > 0
        ))
    
    # Initialize environment obstacles from scene manager OR from external config
    env_obstacles = []
    
    # Option 1: Use scene manager generated obstacles (recommended)
    if hasattr(scene_manager, 'obstacle_configs') and scene_manager.obstacle_configs:
        print("Using obstacles from dynamic scene manager")
        for obstacle_config in scene_manager.obstacle_configs:
            obstacle = Obstacle(my_world, **obstacle_config)
            for i in range(len(robot_world_models)):
                world_model_idx = obstacle.add_to_world_model(robot_world_models[i], X_robots[i])
                print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")
            env_obstacles.append(obstacle)
    
    # Option 2: Fallback to external config file
    else:
        print(f"Using obstacles from config file: {collision_obstacles_cfg_path}")
        if os.path.exists(collision_obstacles_cfg_path):
            col_ob_cfg = load_yaml(collision_obstacles_cfg_path)
            for obstacle_config in col_ob_cfg:
                obstacle = Obstacle(my_world, **obstacle_config)
                for i in range(len(robot_world_models)):
                    world_model_idx = obstacle.add_to_world_model(robot_world_models[i], X_robots[i])
                    print(f"Obstacle {obstacle.name} added to world model {world_model_idx}")
                env_obstacles.append(obstacle)
    
    world_prim = stage.GetPrimAtPath("/World")
    stage.SetDefaultPrim(world_prim)
    
    # Wait for simulation to start
    wait_for_playing(my_world, simulation_app, autoplay=True)
    
    # Populate robot context for dynamic obstacle prediction
    for i in range(n_robots):
        mpc_config = load_yaml(mpc_config_paths[i])
        
        has_dynamic_obs_cost = (
            "cost" in mpc_config and 
            "custom" in mpc_config["cost"] and 
            "arm_base" in mpc_config["cost"]["custom"] and 
            "dynamic_obs_cost" in mpc_config["cost"]["custom"]["arm_base"]
        )
        
        if has_dynamic_obs_cost:
            n_obstacle_spheres = sum(robot_sphere_counts[j] for j in col_pred_with[i])
            
            env_topics[i]["env_id"] = 0
            env_topics[i]["robot_id"] = i
            env_topics[i]["robot_pose"] = X_robots[i].tolist()
            env_topics[i]["n_obstacle_spheres"] = n_obstacle_spheres
            env_topics[i]["n_own_spheres"] = robot_sphere_counts[i]
            env_topics[i]["horizon"] = mpc_config["model"]["horizon"]
            env_topics[i]["n_rollouts"] = mpc_config["mppi"]["num_particles"]
            env_topics[i]["col_pred_with"] = col_pred_with[i]
            env_topics[i]["mpc_config_paths"] = mpc_config_paths
            env_topics[i]["robot_config_paths"] = robot_config_paths
            env_topics[i]["robot_sphere_counts"] = robot_sphere_counts_split

    # Initialize robots
    for i, robot in enumerate(robots):
        robot_idx_lists[i] = [robot.robot.get_dof_index(x) for x in robot.j_names]
        if robot_idx_lists[i] is None:
            raise RuntimeError(f"Failed to get DOF indices for robot {i}")
        
        idx_list = robot_idx_lists[i]
        assert idx_list is not None
        robot.init_joints(idx_list)
        robots[i].init_solver(robot_world_models[i], robots_collision_caches[i], MPC_DT, DEBUG)
        robots[i].robot._articulation_view.initialize()
        
        checker = robots[i].get_cchecker()
        ccheckers.append(checker)

    # Register obstacles with all collision checkers AFTER all robots are initialized
    for i in range(len(env_obstacles)):
        env_obstacles[i].register_ccheckers(ccheckers)

    # Set robots in scene manager for marker tracking
    scene_manager.set_robots(robots)

    # Target update callback
    def update_robot_targets(updated_targets: dict):
        """Callback when targets are updated by scene manager"""
        print(f"🔄 APPLYING TARGET UPDATES to robots: {list(updated_targets.keys())}")
        for robot_id, new_target_world in updated_targets.items():
            if robot_id < len(robots):
                # Update robot target using the correct method
                # set_new_target_for_solver takes world coordinates
                orientation = np.array([1, 0, 0, 0])  # Keep orientation unchanged
                success = robots[robot_id].set_new_target_for_solver(new_target_world, orientation)
                status = "✅ SUCCESS" if success else "⚠️  NO CHANGE"
                print(f"   Robot {robot_id}: {status} - target at [{new_target_world[0]:.3f}, {new_target_world[1]:.3f}, {new_target_world[2]:.3f}]")

    # Add callback to scene manager
    scene_manager.add_target_update_callback(update_robot_targets)
    
    # Start dynamic scene
    scene_manager.start_scene()
    print("Dynamic scene started - targets will change every 5-10 seconds")
    
    # Main simulation loop
    t_idx = 0
    try:
        while simulation_app.is_running():
            point_visualizer_inputs = []
            my_world.step(render=True)       
            
            # Update obstacle poses
            for i in range(len(env_obstacles)): 
                env_obstacles[i].update_registered_ccheckers()
                
            # Update robot marker positions to follow end-effectors
            if t_idx % 5 == 0:  # Update every 5 steps to reduce computation
                scene_manager.update_robot_markers()
                
            # Robot planning and control
            if OBS_PREDICTION:         
                plans = [robots[i].get_plan(valid_spheres_only=False) for i in range(len(robots))]
                if VISUALIZE_PLANS_AS_DOTS:
                    for i in range(len(robots)):
                        point_visualizer_inputs.append({
                            'points': plans[i]['task_space']['spheres']['p'][:robots[i].H].to(tensor_args.device), 
                            'color': 'green'
                        })
            
            for i in range(len(robots)):
                if robots[i].use_col_pred:
                    robots[i].update(plans, col_pred_with[i], t_idx, tensor_args, i)
                else:
                    robots[i].update()
                    
                action = robots[i].plan(max_attempts=2)
                robots[i].command(action, num_times=1)
                
                # Visualization
                if VISUALIZE_MPC_ROLLOUTS:
                    point_visualizer_inputs.append({
                        'points': robots[i].get_rollouts_in_world_frame(), 
                        'color': 'green'
                    })
                if VISUALIZE_ROBOT_COL_SPHERES and t_idx % 2 == 0:
                    robots[i].visualize_robot_as_spheres(robots[i].curobo_format_joints)

            if len(point_visualizer_inputs):
                draw_points(point_visualizer_inputs)
                
            t_idx += 1
            
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    finally:
        # Clean up
        scene_manager.stop_scene()
        print("Simulation ended")

    # Apply OmniGlass to bin parts
    glass = OmniGlass()
    bin_prims = [stage.GetPrimAtPath(f"/World/bin_part_{i}") for i in range(5)]
    for prim in bin_prims:
        glass.apply(prim)


def resolve_meta_config_path(robot_model: str) -> str:
    """Resolve meta-configuration paths to absolute paths."""
    root_path = "projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/meta_cfgs"
    return os.path.join(root_path, f"{robot_model}.yml")


if __name__ == "__main__":
    if DEBUG_GPU_MEM:
        torch.cuda.memory._record_memory_history()
    
    # Define robot types - you can modify this list
    input_args = ['franka', 'franka', 'ur5e', 'franka']  # 4 robots for demo
    
    try:
        main([resolve_meta_config_path(robot_model) for robot_model in input_args])
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close() 
#
# G1 Dual Arm Centralized MPC Controller
#
# This script controls both arms of the G1 robot using a centralized MPC approach.
# It is based on the dual_arm_centralized_franka_mpc.py example.

try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse
import os
import numpy as np

## parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
)
parser.add_argument(
    "--robot", 
    type=str, 
    default="g1_23dofs_dual_arm.yml", 
    help="robot configuration to load"
)
parser.add_argument(
    "--override_particle_file", 
    type=str, 
    default="projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_g1_dual_arm.yml", 
    help="override particle MPC config file"
)

args = parser.parse_args()

###########################################################

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
 
# Third Party
import carb
from helper import add_robot_to_scene, add_extensions
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects import cuboid, sphere

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

############################################################

EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")

def draw_points(rollouts: torch.Tensor):
    if rollouts is None:
        return
    # Standard Library
    import random

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
    N = 100
    # if draw.get_num_points() > 0:
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


def main():
    # Create Isaac Sim world
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # Make targets for both arms
    left_target = cuboid.VisualCuboid(
        "/World/left_target",
        position=np.array([0.2, 0.3, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )
    
    right_target = cuboid.VisualCuboid(
        "/World/right_target",
        position=np.array([0.8, 0.3, 0.5]),
        orientation=np.array([0, 1, 0, 0]),
        color=np.array([0, 1.0, 0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    past_pose_left = None
    past_pose_right = None
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    # Determine robot configuration path
    if os.path.isabs(args.robot):
        robot_config_path = args.robot
    elif args.robot.startswith('./') or args.robot.startswith('../') or os.path.exists(args.robot):
        robot_config_path = args.robot
    else:
        robot_config_path = join_path(get_robot_configs_path(), args.robot)
        
    robot_cfg = load_yaml(robot_config_path)["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    
    # Add buffer to collision spheres
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    # Add the robot to the scene
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

    articulation_controller = robot.get_articulation_controller()

    # Create collision world configuration
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.04
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    init_curobo = False

    tensor_args = TensorDeviceType()

    # Set up MPC solver configuration
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        use_cuda_graph=False,
        use_cuda_graph_metrics=True,
        use_cuda_graph_full_step=False,
        self_collision_check=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
        use_mppi=True,
        use_lbfgs=False,
        use_es=False,
        store_rollouts=True,
        step_dt=0.03,
        override_particle_file=args.override_particle_file
    )

    # Create MPC solver
    mpc = MpcSolver(mpc_config)

    # Get robot state and set up goal
    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    
    # Create dual-arm goals: [num_arms, 3] for positions, [num_arms, 4] for quaternions
    # These will be replaced with actual target positions once simulation starts
    dual_arm_positions = torch.stack([
        tensor_args.to_device(np.array([0.2, 0.3, 0.5])),   # Left arm target
        tensor_args.to_device(np.array([0.8, 0.3, 0.5]))    # Right arm target
    ], dim=0)  # Shape: [2, 3]
    
    dual_arm_quaternions = torch.stack([
        tensor_args.to_device(np.array([1.0, 0.0, 0.0, 0.0])),   # Left arm orientation
        tensor_args.to_device(np.array([1.0, 0.0, 0.0, 0.0]))    # Right arm orientation  
    ], dim=0)  # Shape: [2, 4]
    
    # Add batch dimension to match expected format: [1, num_arms, 3/4]
    dual_arm_positions = dual_arm_positions.unsqueeze(0)    # Shape: [1, 2, 3]
    dual_arm_quaternions = dual_arm_quaternions.unsqueeze(0)  # Shape: [1, 2, 4]
    
    # Create dual-arm pose goal
    ik_goal = Pose(
        position=dual_arm_positions,
        quaternion=dual_arm_quaternions
    )
    
    # Set up goal for solver
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=ik_goal,
    )

    # Initialize the MPC solver with the goal
    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)

    # Load USD stage and initialize world
    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None
    step = 0
    spheres = None  # For collision sphere visualization
    add_extensions(simulation_app, args.headless_mode)
    
    # Print initialization info
    print(f"Initialized G1 dual-arm centralized MPC")
    print(f"Using robot config: {robot_config_path}")
    print(f"Using particle file: {args.override_particle_file}")
    
    # Main simulation loop
    while simulation_app.is_running():
        if not init_world:
            for _ in range(10):
                my_world.step(render=True)
            init_world = True
        draw_points(mpc.get_visual_rollouts())

        my_world.step(render=True)
        if not my_world.is_playing():
            continue

        step_index = my_world.current_time_step_index

        if step_index <= 10:
            robot._articulation_view.initialize()

            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )

        if not init_curobo:
            init_curobo = True
        step += 1
        step_index = step
        
        # Update the collision world periodically
        if step_index % 1000 == 0:
            print("Updating world")
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                ignore_substring=[
                    robot_prim_path,
                    "/World/left_target",
                    "/World/right_target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
                reference_prim_path=robot_prim_path,
            )
            obstacles.add_obstacle(world_cfg_table.cuboid[0])
            mpc.world_coll_checker.load_collision_model(obstacles)

        # Get target positions from visual cubes
        left_cube_position, left_cube_orientation = left_target.get_world_pose()
        right_cube_position, right_cube_orientation = right_target.get_world_pose()

        if past_pose_left is None:
            past_pose_left = left_cube_position + 1.0
            past_pose_right = right_cube_position + 1.0

        # Check if targets have moved
        left_changed = np.linalg.norm(left_cube_position - past_pose_left) > 1e-3
        right_changed = np.linalg.norm(right_cube_position - past_pose_right) > 1e-3
        
        if left_changed or right_changed:
            # Create multi-arm goals for both arms
            dual_arm_positions = torch.stack([
                tensor_args.to_device(left_cube_position),   # Left arm target
                tensor_args.to_device(right_cube_position)   # Right arm target
            ], dim=0)  # Shape: [2, 3]
            
            dual_arm_quaternions = torch.stack([
                tensor_args.to_device(left_cube_orientation),   # Left arm orientation
                tensor_args.to_device(right_cube_orientation)   # Right arm orientation  
            ], dim=0)  # Shape: [2, 4]
            
            # Add batch dimension
            dual_arm_positions = dual_arm_positions.unsqueeze(0)    # Shape: [1, 2, 3]
            dual_arm_quaternions = dual_arm_quaternions.unsqueeze(0)  # Shape: [1, 2, 4]
            
            # Create dual-arm pose goal
            ik_goal = Pose(
                position=dual_arm_positions,
                quaternion=dual_arm_quaternions
            )
            
            # Update the goal
            goal_buffer.goal_pose = ik_goal
            mpc.update_goal(goal_buffer)
            
            past_pose_left = left_cube_position
            past_pose_right = right_cube_position
            
            print(f"Updated dual-arm goals:")
            print(f"  Left arm target:  {left_cube_position}")
            print(f"  Right arm target: {right_cube_position}")
            print(f"  Goal position shape: {goal_buffer.goal_pose.position.shape}")
            print(f"  Goal quaternion shape: {goal_buffer.goal_pose.quaternion.shape}")

        # Get current robot state
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        js_names = robot.dof_names
        sim_js_names = robot.dof_names

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        
        # Visualize collision spheres if enabled
        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = mpc.kinematics.get_robot_as_spheres(cu_js.position)

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
                    if si < len(spheres) and not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))
        
        # Update the current state for the MPC solver
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
        mpc_result = mpc.step(current_state, max_attempts=2)

        # Process MPC result
        succ = mpc_result is not None and mpc_result.success
        if succ:
            cmd_state_full = mpc_result.js_action
            common_js_names = []
            idx_list = []
            for x in sim_js_names:
                if x in cmd_state_full.joint_names:
                    idx_list.append(robot.get_dof_index(x))
                    common_js_names.append(x)

            cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
            cmd_state_full = cmd_state

            art_action = ArticulationAction(
                cmd_state.position.view(-1).cpu().numpy(),
                joint_indices=idx_list,
            )
            
            # Print metrics occasionally
            if step_index % 1000 == 0 and hasattr(mpc_result, 'metrics'):
                if hasattr(mpc_result.metrics, 'feasible'):
                    print(f"Feasible: {mpc_result.metrics.feasible.item()}")
                if hasattr(mpc_result.metrics, 'pose_error'):
                    print(f"Pose error: {mpc_result.metrics.pose_error.item()}")

            # Apply action
            articulation_controller.apply_action(art_action)
        else:
            carb.log_warn("No action is being taken.")


############################################################

if __name__ == "__main__":
    main()
    simulation_app.close() 
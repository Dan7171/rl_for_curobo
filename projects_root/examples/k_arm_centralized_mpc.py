#!/usr/bin/env python3
#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

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
from typing import List, Dict, Tuple, Optional
import numpy as np

## import curobo:

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

parser.add_argument("--robot", type=str, default="franka_dual_arm.yml", help="robot configuration to load")
parser.add_argument("--num_arms", type=int, default=2, help="number of arms in the system")
parser.add_argument("--override_particle_file", type=str, default=None, help="override particle MPC config file")
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
 
# Standard Library
import os

# Third Party
import carb
import numpy as np
from helper import add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects import cuboid, sphere

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

############################################################

EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")

# Standard Library
from typing import Optional

# Third Party
from helper import add_extensions, add_robot_to_scene

# CuRobo
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


class KArmTargetManager:
    """Manages target cubes and poses for K-arm system."""
    
    def __init__(self, num_arms: int, my_world: World, tensor_args: TensorDeviceType):
        self.num_arms = num_arms
        self.my_world = my_world
        self.tensor_args = tensor_args
        self.targets = []
        self.past_poses = []
        
        # Generate colors for targets
        self.colors = self._generate_arm_colors(num_arms)
        
        # Create target cubes
        self._create_targets()
        
    def _generate_arm_colors(self, num_arms: int) -> List[List[float]]:
        """Generate distinct colors for each arm's target."""
        colors = []
        for i in range(num_arms):
            # Generate colors in HSV space for better distribution
            hue = (i * 360 / num_arms) % 360
            # Convert HSV to RGB (simplified)
            if hue < 120:
                r, g, b = 1.0, hue/120, 0.0
            elif hue < 240:
                r, g, b = (240-hue)/120, 1.0, 0.0
            else:
                r, g, b = 0.0, (360-hue)/120, 1.0
                
            colors.append([r, g, b])
        return colors
        
    def _create_targets(self):
        """Create target visualization cubes for each arm."""
        # Default positions in a line
        for i in range(self.num_arms):
            x_pos = 0.3 + i * 0.5  # Space arms 0.5m apart
            y_pos = 0.3
            z_pos = 0.5
            
            target = cuboid.VisualCuboid(
                f"/World/arm_{i}_target",
                position=np.array([x_pos, y_pos, z_pos]),
                orientation=np.array([0, 1, 0, 0]),
                color=np.array(self.colors[i]),
                size=0.05,
            )
            self.targets.append(target)
            self.past_poses.append(np.array([x_pos, y_pos, z_pos]) + 1.0)  # Initialize with changed state
            
    def check_targets_changed(self) -> bool:
        """Check if any target has moved significantly."""
        changed = False
        for i, target in enumerate(self.targets):
            current_pos, _ = target.get_world_pose()
            if np.linalg.norm(current_pos - self.past_poses[i]) > 1e-3:
                changed = True
                self.past_poses[i] = current_pos
        return changed
        
    def get_multi_arm_pose_goal(self) -> Pose:
        """Get current target poses for all arms in multi-arm format."""
        positions = []
        orientations = []
        
        for target in self.targets:
            pos, quat = target.get_world_pose()
            positions.append(self.tensor_args.to_device(pos))
            orientations.append(self.tensor_args.to_device(quat))
            
        # Stack into [num_arms, 3] and [num_arms, 4] tensors
        multi_arm_positions = torch.stack(positions, dim=0)  # [num_arms, 3]
        multi_arm_quaternions = torch.stack(orientations, dim=0)  # [num_arms, 4]
        
        return Pose(
            position=multi_arm_positions,
            quaternion=multi_arm_quaternions
        )
        
    def print_target_update(self):
        """Print debug information about target updates."""
        print(f"Updated {self.num_arms}-arm goals:")
        for i, target in enumerate(self.targets):
            pos, _ = target.get_world_pose()
            print(f"  Arm {i} target: {pos}")


def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    setup_curobo_logger("warn")
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    # Handle both absolute paths and standard config names
    if os.path.isabs(args.robot):
        # Absolute path provided
        robot_config_path = args.robot
    elif args.robot.startswith('./') or args.robot.startswith('../') or os.path.exists(args.robot):
        # Relative path provided or file exists in current directory
        robot_config_path = args.robot
    else:
        # Standard config name, look in CuRobo configs
        robot_config_path = join_path(get_robot_configs_path(), args.robot)
    
    robot_cfg = load_yaml(robot_config_path)["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    
    # Initialize target manager for K arms
    target_manager = KArmTargetManager(args.num_arms, my_world, tensor_args)

    articulation_controller = robot.get_articulation_controller()

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

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].pose[2] = -10.0

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

    # Determine override particle file based on number of arms
    if args.override_particle_file is None:
        if args.num_arms == 2:
            override_particle_file = 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_dual_arm.yml'
        elif args.num_arms == 3:
            override_particle_file = 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_triple_arm.yml'
        elif args.num_arms == 4:
            override_particle_file = 'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_quad_arm.yml'
        else:
            # Generate config on the fly for arbitrary K
            from projects_root.utils.multi_arm_config_generator import MultiArmConfigGenerator
            generator = MultiArmConfigGenerator("tmp_configs")
            override_particle_file = generator.generate_particle_mpc_config(args.num_arms, f"k_arm_{args.num_arms}")
            print(f"Generated particle MPC config for {args.num_arms} arms: {override_particle_file}")
    else:
        override_particle_file = args.override_particle_file

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
        step_dt=0.02,
        override_particle_file=override_particle_file
    )

    mpc = MpcSolver(mpc_config)

    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    joint_names = mpc.rollout_fn.joint_names

    state = mpc.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg, joint_names=joint_names)
    )
    current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    
    # Create initial goal with multi-arm poses
    initial_multi_arm_pose = target_manager.get_multi_arm_pose_goal()
    goal = Goal(
        current_state=current_state,
        goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        goal_pose=initial_multi_arm_pose,
    )

    goal_buffer = mpc.setup_solve_single(goal, 1)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state, max_attempts=2)

    usd_help.load_stage(my_world.stage)
    init_world = False
    cmd_state_full = None
    step = 0
    spheres = None  # For collision sphere visualization
    add_extensions(simulation_app, args.headless_mode)
    
    print(f"Initialized {args.num_arms}-arm centralized MPC system")
    print(f"Using robot config: {args.robot}")
    print(f"Using particle file: {override_particle_file}")
    
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
        
        if step_index % 1000 == 0:
            print("Updating world")
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                ignore_substring=[
                    robot_prim_path,
                    "/World/defaultGroundPlane",
                    "/curobo",
                ] + [f"/World/arm_{i}_target" for i in range(args.num_arms)],  # Ignore all arm targets
                reference_prim_path=robot_prim_path,
            )
            obstacles.add_obstacle(world_cfg_table.cuboid[0])
            mpc.world_coll_checker.load_collision_model(obstacles)

        # Check if any targets have moved
        if target_manager.check_targets_changed():
            # Update multi-arm goal
            multi_arm_pose = target_manager.get_multi_arm_pose_goal()
            
            # IMPORTANT: Direct assignment instead of copy_() to preserve multi-arm structure
            goal_buffer.goal_pose = multi_arm_pose
            mpc.update_goal(goal_buffer)
            
            target_manager.print_target_update()
            print(f"  Goal position shape: {goal_buffer.goal_pose.position.shape}")
            print(f"  Goal quaternion shape: {goal_buffer.goal_pose.quaternion.shape}")

        # Get robot current state:
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
        
        # Collision sphere visualization
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
                    if not np.isnan(s.position[0]):
                        spheres[si].set_world_pose(position=np.ravel(s.position))
                        spheres[si].set_radius(float(s.radius))
        
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

        mpc_result = mpc.step(current_state, max_attempts=2)

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

        art_action = ArticulationAction(
            cmd_state.position.view(-1).cpu().numpy(),
            joint_indices=idx_list,
        )
        
        if step_index % 1000 == 0:
            print(mpc_result.metrics.feasible.item(), mpc_result.metrics.pose_error.item())

        if succ:
            # Set desired joint angles obtained from MPC:
            for _ in range(1):
                articulation_controller.apply_action(art_action)
        else:
            carb.log_warn("No action is being taken.")


############################################################

if __name__ == "__main__":
    main()
    simulation_app.close() 
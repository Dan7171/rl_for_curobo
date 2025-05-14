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
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library

# Standard Library
import argparse

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

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
args = parser.parse_args()
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
import numpy as np
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.kit import SimulationApp

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

def tmp_helper_get_full_js(robot_list, tensor_args):
    sim_js_names = robot_list[0].dof_names
    sim_js = robot_list[0].get_joints_state()
    
    full_js = JointState(
        position=tensor_args.to_device(sim_js.positions).view(1, -1),
        velocity=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
        acceleration=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
        jerk=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
        joint_names=sim_js_names,
    )
    for i in range(1, len(robot_list)):
        sim_js = robot_list[i].get_joints_state()
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions).view(1, -1),
            velocity=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities).view(1, -1) * 0.0,
            joint_names=sim_js_names,
        )
        full_js = full_js.stack(cu_js)
    
    return full_js
def main():

    mode_debug= 'mpc' # 'motion_gen'

    usd_help = UsdHelper()
    act_distance = 0.2

    n_envs = 2
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    # Make a target to follow
    target_list = []
    target_material_list = []
    offset_y = 2.5
    radius = 0.1
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot_list = []
    for i in range(n_envs):
        if i > 0:
            pose.position[0, 1] += offset_y
        usd_help.add_subroot("/World", "/World/world_" + str(i), pose)

        target = cuboid.VisualCuboid(
            "/World/world_" + str(i) + "/target",
            position=np.array([0.5, 0, 0.5]) + pose.position[0].cpu().numpy(),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        target_list.append(target)
        r = add_robot_to_scene(
            robot_cfg,
            my_world,
            "/World/world_" + str(i) + "/",
            robot_name="robot_" + str(i),
            position=pose.position[0].cpu().numpy(),
            initialize_world=False,
        )
        robot_list.append(r[0])
    setup_curobo_logger("warn")
    my_world.initialize_physics()

    # warmup curobo instance

    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"

    world_file = ["collision_test.yml", "collision_thin_walls.yml"]
    world_cfg_list = []
    for i in range(n_envs):
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file[i]))
        )  # .get_mesh_world()
        world_cfg.objects[0].pose[2] -= 0.02
        world_cfg.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
        usd_help.add_world_to_stage(world_cfg, base_frame="/World/world_" + str(i))
        world_cfg_list.append(world_cfg)

    if mode_debug == 'motion_gen':
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg_list,
            tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.03,
            collision_cache={"obb": 10, "mesh": 10},
            collision_activation_distance=0.025,
            maximum_trajectory_dt=0.25,
        )
        motion_gen = MotionGen(motion_gen_config)
        
        print("warming up...")

    else:
        
        """
        https://curobo.org/_api/curobo.wrap.reacher.mpc.html#curobo.wrap.reacher.mpc.MpcSolver._update_batch_size
        High-level interface for Model Predictive Control (MPC).
        MPC can reach Cartesian poses and joint configurations while avoiding obstacles. The solver uses Model Predictive Path Integral (MPPI) optimization as the solver. MPC only optimizes locally so the robot can get stuck near joint limits or behind obstacles. To generate global trajectories, use MotionGen.
        See Model Predictive Control (MPC) for an example. This MPC solver implementation can be used in the following steps:
        Create a Goal object with the target pose or joint configuration.
        Create a goal buffer for the problem type using setup_solve_single, setup_solve_goalset, setup_solve_batch, setup_solve_batch_goalset, setup_solve_batch_env, or setup_solve_batch_env_goalset. Pass the goal object from the previous step to this function. This function will update the internal solve state of MPC and also the goal for MPC. An augmented goal buffer is returned.
        Call step with the current joint state to get the next action.
        To change the goal, create a Pose object with new pose or JointState object with new joint configuration. Then copy the target into the augmented goal buffer using goal_buffer.goal_pose.copy_(new_pose) or goal_buffer.goal_state.copy_(new_state).
        Call update_goal with the augmented goal buffer to update the goal for MPC.
        Call step with the current joint state to get the next action.
        To dynamically change the type of goal reached between pose and joint configuration targets, create the goal object in step 1 with both targets and then use enable_cspace_cost and enable_pose_cost to enable or disable reaching joint configuration cost and pose cost.
        Initializes the MPC solver.
        """
        mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg_list,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            self_collision_check=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={"obb": 30, "mesh": 10},
            use_mppi=True,
            use_lbfgs=False,
            use_es=False,
            store_rollouts=True,
            step_dt=0.02,
            n_collision_envs=n_envs, # n_collision_envs – Number of collision environments to create for batched planning across different environments. Only used for MpcSolver.setup_solve_batch_env and MpcSolver.setup_solve_batch_env_goalset.
    )
        mpc = MpcSolver(mpc_config)
        add_extensions(simulation_app, args.headless_mode)
        
        # 1 x 7
        retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        # n robots x 7
        retract_cfg_batch = retract_cfg.repeat(n_envs, 1)
        # 7
        joint_names = mpc.rollout_fn.joint_names
        from curobo.rollout.rollout_base import Goal

        # n robots x 7
        
        current_state_batch = JointState.from_position(retract_cfg_batch, joint_names=joint_names) # tmp_helper_get_full_js(robot_list, tensor_args)         # # n robots x 9
        current_state_batch.position = current_state_batch.position.contiguous()
        current_state_batch.velocity = current_state_batch.velocity.contiguous()
        current_state_batch.acceleration = current_state_batch.acceleration.contiguous()
        current_state_batch.jerk = current_state_batch.jerk.contiguous()
        
        # n robots x 7
        state = mpc.rollout_fn.compute_kinematics(current_state_batch)

        # n robots x 3
        retract_pose_batch = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        
        

    """
    # Goal # https://curobo.org/_api/curobo.rollout.rollout_base.html#curobo.rollout.rollout_base.Goal

    Goal data class used to update optimization target.
    #NOTE: We can parallelize Goal in two ways: 
    # 1. Solve for current_state, pose pair in same environment 
    # 2. Solve for current_state, pose pair in different environment 
    # For case (1),
    #   we use batch_pose_idx to find the memory address of the current_state, 
    #   pose pair while keeping batch_world_idx = [0] 
    # For case (2), 
    #   we add a batch_world_idx[0,1,2..].
            
    
    # types of setup_solve_ :
    setup_solve_single: Creates a goal buffer to solve for a robot to reach target pose or joint configuration.
    setup_solve_goalset: Creates a goal buffer to solve for a robot to reach a pose in a set of poses.
    setup_solve_batch:Creates a goal buffer to solve for a batch of robots to reach targets.
    setup_solve_batch_goalset: Creates a goal buffer to solve for a batch of robots to reach a set of poses.
    setup_solve_batch_env: Creates a goal buffer to solve for a batch of robots in different collision worlds.
    setup_solve_batch_env_goalset: Creates a goal buffer to solve for a batch of robots in different collision worlds.
    """
    different_env_for_each_robot = True 
    num_seeds = 1 # todo shold be 1?
    
    # n robots x 7
    goal_state_batch = JointState.from_position(retract_cfg_batch, joint_names=joint_names)

    if different_env_for_each_robot:
        goal = Goal(
            current_state=current_state_batch, # https://curobo.org/_api/curobo.rollout.rollout_base.html#curobo.rollout.rollout_base.Goal
            goal_state=goal_state_batch, 
            goal_pose=retract_pose_batch,
            batch_world_idx=list(range(n_envs)), # optional
        )
        goal_buffer = mpc.setup_solve_batch_env(goal, num_seeds)
    else:
        goal = Goal(
            current_state=current_state_batch, # https://curobo.org/_api/curobo.rollout.rollout_base.html#curobo.rollout.rollout_base.Goal
            goal_state=JointState.from_position(retract_cfg, joint_names=joint_names), 
            goal_pose=retract_pose_batch
        )
        goal_buffer = mpc.setup_solve_batch(goal, num_seeds)
    mpc.update_goal(goal_buffer)
    mpc_result = mpc.step(current_state_batch, max_attempts=2)


    # START 174 - 190 curobo/examples/isaac_sim/batch_motion_gen_reacher.py
    config = RobotWorldConfig.load_from_config(
        robot_file, world_cfg_list, collision_activation_distance=act_distance
    )
    model = RobotWorld(config)
    i = 0
    max_distance = 0.5
    x_sph = torch.zeros((n_envs, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius
    env_query_idx = torch.arange(n_envs, device=tensor_args.device, dtype=torch.int32)
    plan_config = MotionGenPlanConfig(
        enable_graph=False, max_attempts=2, enable_finetune_trajopt=True
    )
    prev_goal = None
    cmd_plan = [None, None]
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    cmd_idx = 0
    past_goal = None
    # 174 - 190 END
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
        step_index = my_world.current_time_step_index

        if step_index <= 10:
            # my_world.reset()
            for robot in robot_list:
                robot._articulation_view.initialize()
                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)

                robot._articulation_view.set_max_efforts(
                    values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                )
        if step_index < 20:
            continue
        sp_buffer = []
        sq_buffer = []
        for k in target_list:
            sph_position, sph_orientation = k.get_local_pose()
            sp_buffer.append(sph_position)
            sq_buffer.append(sph_orientation)

        ik_goal = Pose(
            position=tensor_args.to_device(sp_buffer),
            quaternion=tensor_args.to_device(sq_buffer),
        )
        if prev_goal is None:
            prev_goal = ik_goal.clone()
        if past_goal is None:
            past_goal = ik_goal.clone()
        
        # prev_distance = ik_goal.distance(prev_goal)
        # past_distance = ik_goal.distance(past_goal)

        goal_buffer.goal_pose.copy_(ik_goal)
        mpc.update_goal(goal_buffer)

        current_state_batch = tmp_helper_get_full_js(robot_list, tensor_args) # full_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)
        current_state_batch = current_state_batch.get_ordered_joint_state(mpc.kinematics.joint_names)
        result = mpc.step(current_state_batch, max_attempts=2)

        for i  in range(len(robot_list)):
            sim_js_names = robot_list[i].dof_names # 9
            cmd_state_full = result.js_action[i] # 9
            common_js_names = [] # 9
            idx_list = [] # 9
            for x in sim_js_names:
                if x in cmd_state_full.joint_names:
                    idx_list.append(robot_list[i].get_dof_index(x))
                    common_js_names.append(x)
            # 9 (Joint state)
            cmd_state = cmd_state_full.get_ordered_joint_state(common_js_names)
            cmd_state_full = cmd_state

            art_action = ArticulationAction(
                cmd_state.position.view(-1).cpu().numpy(),
                joint_indices=idx_list,
            )
            art_controllers[i].apply_action(art_action)
        for _ in range(len(robot_list)):
            my_world.step(render=False)


if __name__ == "__main__":
    main()

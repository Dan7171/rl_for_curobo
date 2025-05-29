"""
This script was built based on the curobo/examples/isaac_sim/batch_motion_gen_reacher.py
and modified to use MPC instead of MotionGen (global planner).
"""
DEBUG = True
COL_PRED = True
try:
    # Third Party
    import isaacsim
except ImportError:
    pass

# Third Party
from typing import Optional
import torch
a = torch.zeros(4, device="cuda:0")

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
import time
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.kit import SimulationApp

# CuRobo
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.rollout.rollout_base import Goal
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictorBatch
from scipy.spatial.transform import Rotation as R






def get_batch_js(robot_list, tensor_args) -> JointState:
    """
    Get the common joint state for a batch of robots.
    """
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

def transform_pose_between_frames(pose:list[float], frame_expressed_in:Optional[list[float]]=None, frame_express_at:Optional[list[float]]=None):
    """
    Transforms pose expressed in some frame F1 (frame_expressed_in), from frame F1 to frame F2 (frame_express_at).
    Both frame_expressed_in and frame_express_at are expressed in the world frame!

    If frame_expressed_in or frame_express_at is not provided, it is assumed that it is the world frame.
    Meaning, that if frame_expressed_in is not provided, it is assumed that the pose is a pose in the world frame.
    And if frame_express_at is not provided, it is assumed that we want to transform the pose to the world frame.
    
    Parameters:
    - pose: [px, py, pz, qw, qx, qy, qz] expressed in F1
    - frame_expressed_in: [px, py, pz, qw, qx, qy, qz] pose of F1 in world (the frame the pose is expressed in)
    - frame_express_at: [px, py, pz, qw, qx, qy, qz] pose of F2 in world (the frame the pose is transformed to)
    
    Returns:
    - pose_in_f2: [px, py, pz, qw, qx, qy, qz] expressed in F2
    """
    
    def decompose(pose):
        pos = np.array(pose[:3])
        quat = np.array(pose[3:])
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # convert qw,qx,qy,qz -> x,y,z,w
        return pos, rot

    def compose(pos, rot):
        quat = rot.as_quat()  # x,y,z,w
        return np.concatenate([pos, [quat[3], quat[0], quat[1], quat[2]]])  # to qw,qx,qy,qz

    world_pose = [0.0, 0, 0, 1, 0, 0, 0]
    if frame_expressed_in is None:
        frame_expressed_in = world_pose
    if frame_express_at is None:
        frame_express_at = world_pose
    # Decompose all poses
    p_f1, r_f1 = decompose(frame_expressed_in)
    p_f2, r_f2 = decompose(frame_express_at)
    p_rel, r_rel = decompose(pose)

    # Pose in world frame: T_world = T_f1 * T_rel
    p_world = r_f1.apply(p_rel) + p_f1
    r_world = r_f1 * r_rel

    # Inverse of F2 pose
    r_f2_inv = r_f2.inv()
    p_f2_inv = -r_f2_inv.apply(p_f2)

    # Transform to F2: T_f2 = T_f2_inv * T_world
    p_in_f2 = r_f2_inv.apply(p_world) + p_f2_inv
    r_in_f2 = r_f2_inv * r_world

    return compose(p_in_f2, r_in_f2)



def main(
        world_files = ["collision_test.yml", "collision_thin_walls.yml"],
        offset_y = 2.5,
        # X_targetsInitial_envFrame = ([0.5, 0, 0.5], [0, 1, 0, 0]),
        ):
    """
    Main function to run the batch MPC example.
    Args:

        world_files: list of world files to load (each environment has a different world file, can use the same world file for all environments)
        offset_y: distance between adjacent environments on the y axis. TODO: modify to include offset_x, offset_z, etc.
        # X_targetsInitial_enviFrame: Initial pose of the robot targets expressed in the i'th environment (same for all i)  frame (each pose is a tuple of [position, orientation])

    """
    setup_curobo_logger("warn") # not sure if I want this

    # number of environments
    n_envs = len(world_files)

    # envs frames poses (expressed in the world frame)
    X_envs  = []
    for i in range(n_envs):
        X_envs.append(Pose.from_list([0, i*offset_y, 0, 1, 0, 0, 0]))

    # robot base poses expressed in the i'th environment frame 
    # (M[i][j] is the j'th robot frame expressed in the i'th environment frame)
    Xlist_rbase_Fenv = [
        [[0, 0, 0, 1, 0, 0, 0], [1.2, 0, 0, 1, 0, 0, 0]], 
        [[0, 0, 0, 1, 0, 0, 0]]
    ]

    # robot base poses expressed in the world frame
    # (M[i][j] is the j'th robot frame expressed in the world frame)
    Xlist_rbase = [] 
    for i in range(n_envs):
        Xlist_rbase.append([])
        for j in range(len(Xlist_rbase_Fenv[i])):
            X_rbase = transform_pose_between_frames(Xlist_rbase_Fenv[i][j], frame_expressed_in=X_envs[i].tolist())
            Xlist_rbase[i].append(X_rbase)

    # target poses expressed in the i'th environment frame. 
    # (M[i][j] is the j'th robot's target pose expressed in the i'th environment frame)
    Xlist_target_Fenv = [
        [[0.5, 0, 0.5, 0, 1, 0, 0], [0.5, 0.5, 0.5, 0, 1, 0, 0]], 
        [[0.5, 0, 0.5, 0, 1, 0, 0]]
    ]
    

    n_robots_envwise = [len(Xlist_rbase_Fenv[i]) for i in range(n_envs)] # number of robots in each environment
    assert len(n_robots_envwise) == n_envs
    n_robots_total = sum(n_robots_envwise) # all robots in all environments
    
    # list of env idx for each robot in the total robot count
    robot_to_env_idx = [env_idx for env_idx in range(n_envs) for _ in range(n_robots_envwise[env_idx])]
    
    # curobo wrapper on torch tensor device type
    tensor_args = TensorDeviceType()
    
    # set up the stage and world:
    # curobo util to make USD operations easier
    usd_help = UsdHelper() 
    # assuming obstacles are in objects_path:
    # world with ground plane
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    stage = my_world.stage
    
    # Prepering robots deployment in different environments:

    
    # load robot config
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    # joint names
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    # initial joint config when simulation starts (modifiable, length = 9, example:[0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0, 0.04, 0.04])
    default_config:list[float] = robot_cfg["kinematics"]["cspace"]["retract_config"]
    
    
    # LOOP OVER ENVIRONMENTS:
    # robots in different environments 
    robot_list = []
    robot_list_by_env = []
    target_list = []
    target_list_by_env = []
    for i in range(n_envs):
        robot_list_by_env.append([])
        target_list_by_env.append([])

        # create stage subroot for the i'th environment, at X_env_i (pose in the world frame)
        world_i_subroot = "/World/world_" + str(i)
        usd_help.add_subroot("/World", world_i_subroot, X_envs[i])

        for j in range(n_robots_envwise[i]):
            # create stage subroot for the i'th environment, at X_env_i (pose in the world frame)
            # usd_help.add_subroot("/World", "/World/world_" + str(i), X_envs[i])

            # create target in the i-th environment
            target = cuboid.VisualCuboid(
                world_i_subroot + "/target" + str(j),
                position=np.array(Xlist_target_Fenv[i][j][:3]) + X_envs[i].position[0].cpu().numpy(),
                orientation=np.array(Xlist_target_Fenv[i][j][3:]),
                color=np.array([1.0, 0, 0]),
                size=0.05,
            )
            target_list.append(target)
            target_list_by_env[i].append(target)

            # add robot to the i-th environment
            r = add_robot_to_scene(
                robot_cfg,
                my_world,
                world_i_subroot + "/",
                robot_name="robot_" + str(i) + "_" + str(j),
                # position=Xlist_rbase_Fenv[i][j].position[0].cpu().numpy(),  # X_env_i.position[0].cpu().numpy(),
                position=Pose.from_list(Xlist_rbase_Fenv[i][j]).position[0].cpu().numpy() + X_envs[i].position[0].cpu().numpy(), # position passed here expressed in the world frame, so its shifted from by X_env_i
                initialize_world=False,
            )
            robot_list.append(r[0])
            robot_list_by_env[i].append(r[0])

    my_world.initialize_physics() # seems to be needed

    # load world config of each environment to the stage
    world_cfg_list = []
    for i in range(n_envs):
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_files[i]))
        )  # .get_mesh_world()
        world_cfg.objects[0].pose[2] -= 0.02
        world_cfg.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
        usd_help.add_world_to_stage(world_cfg, base_frame="/World/world_" + str(i))
        for j in range(n_robots_envwise[i]):
            world_cfg_list.append(world_cfg)
         
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

    
    # TODO: currently not used, has some bug RuntimeError: shape '[1, 1, 95, 30, 7]' is invalid for input of size 19740
    # overide_cfg =  'projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_batch.yml'
    # with open(overide_cfg, 'r') as f:
    #     cfg = load_yaml(overide_cfg)
    #     b = cfg['mppi']['num_particles']
    #     H = cfg['model']['horizon']

    # init mpc *batch* solver config
    b = 400 # TODO: change to n_rollouts of each robot
    H = 30 # TODO: change to H of the mpc solver
    
    if COL_PRED:
        dynamic_obs_checker=DynamicObsCollPredictorBatch(
            tensor_args=tensor_args,
            n_envs=n_envs,
            b=b,
            H=H, 
            n_robot_spheres=65,
            n_robots_envwise=n_robots_envwise,
            spheres_to_ignore= list(range(50)),# [0,1,2,3,4,5,6,7], # spheres to ignore for collision checking, see projects_root/utils/ FRANKA_COL_SPHERES.png
            attached_objects_spheres=[61,62,63,64] # spheres to treat as ignored when robot is not picking up the object, but only when the robot is not attached to the object.
        )
    else:
        dynamic_obs_checker = None
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg_list, # although its a list and lists are not shown in the valid inputs, it is valid input and there is no error (like in the motion gen example).
        use_cuda_graph=not DEBUG,
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
        n_collision_envs= n_envs, # len(world_cfg_list), # n_envs, # n_collision_envs – Number of collision environments to create for batched planning across different environments. Only used for MpcSolver.setup_solve_batch_env and MpcSolver.setup_solve_batch_env_goalset.
        dynamic_obs_checker=dynamic_obs_checker,
        # override_particle_file='projects_root/projects/dynamic_obs/dynamic_obs_predictor/cfgs/particle_mpc_batch.yml'
    )
    
    # init mpc solver for batch of robots

    mpc = MpcSolver(mpc_config) # batch solver, solve (n robots) mpc problem in parallel on every .step() call
    joint_names = mpc.rollout_fn.joint_names # 7
    
    add_extensions(simulation_app, args.headless_mode)
    
    # get retract (initial) config for each robot (currently just duplicated, assuming all robots start at the same cfg but it should be unique)
    
    # 1 x 7
    retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    
    # for each robot, repeat its retract configuration (initial jointconfig)
    retract_cfg_batch = retract_cfg.repeat(n_robots_total, 1) # n robots x 7
    

    # get current state for each robot
    # n robots x 7
    current_state_batch = JointState.from_position(retract_cfg_batch, joint_names=joint_names) # get_batch_js(robot_list, tensor_args)         # # n robots x 9
    # make contiguous (for memory efficiency, like cuda graphs...)
    current_state_batch.position = current_state_batch.position.contiguous()
    current_state_batch.velocity = current_state_batch.velocity.contiguous()
    current_state_batch.acceleration = current_state_batch.acceleration.contiguous()
    current_state_batch.jerk = current_state_batch.jerk.contiguous()
    
    # n robots x 7
    state = mpc.rollout_fn.compute_kinematics(current_state_batch) # FK probably

    # n robots x 3
    retract_pose_batch = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        
        

    """
    # Goal # https://curobo.org/_api/curobo.rollout.rollout_base.html#curobo.rollout.rollout_base.Goal

    Goal data class used to update optimization target.
    # NOTE: We can parallelize Goal in two ways: 
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
    num_seeds = 1 # TODO: should stay 1 as in the setup_solve_single case for 1 mpc robot?
    
    # n robots x 7
    goal_state_batch = JointState.from_position(retract_cfg_batch, joint_names=joint_names)

    if different_env_for_each_robot:
        goal = Goal(
            # batch_pose_idx=torch.tensor(list(range(n_envs)), device=tensor_args.device), # optional
            current_state=current_state_batch, # https://curobo.org/_api/curobo.rollout.rollout_base.html#curobo.rollout.rollout_base.Goal
            goal_state=goal_state_batch, 
            goal_pose=retract_pose_batch,
            batch_world_idx=torch.tensor(robot_to_env_idx, device=tensor_args.device), # optional
            batch_pose_idx=torch.tensor(robot_to_env_idx, device=tensor_args.device), # optional
            batch_enable_idx=torch.tensor(robot_to_env_idx, device=tensor_args.device), # optional

        )
        goal_buffer = mpc.setup_solve_batch_env_custom(goal, num_seeds, n_envs)

    else:
        goal = Goal(
            current_state=current_state_batch, # https://curobo.org/_api/curobo.rollout.rollout_base.html#curobo.rollout.rollout_base.Goal
            goal_state=JointState.from_position(retract_cfg, joint_names=joint_names), 
            goal_pose=retract_pose_batch
        )
        goal_buffer = mpc.setup_solve_batch(goal, num_seeds)
    mpc.update_goal(goal_buffer)
    # mpc.update_goal(goal)
    _ = mpc.step(current_state_batch, max_attempts=2)

    # START 174 - 190 curobo/examples/isaac_sim/batch_motion_gen_reacher.py
    # config = RobotWorldConfig.load_from_config(
    #     robot_file, world_cfg_list, 
    #     collision_activation_distance=act_distance
    # )
    # model = RobotWorld(config)
    i = 0

    # 174 - 190 END
    art_controllers = [r.get_articulation_controller() for r in robot_list]
    
    t = 0
    ctrl_loop_timer = 0
    world_step_timer = 0
    mpc_solver_timer = 0
    targets_update_timer = 0
    joint_state_timer = 0
    action_timer = 0
    while simulation_app.is_running():
        ctrl_loop_timer_start = time.time()
        world_step_timer_start = time.time()
        my_world.step(render=True)
        world_step_timer += time.time() - world_step_timer_start
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

        # get target poses from simulation
        
        targets_update_timer_start = time.time()
        
        # for each robot, get target pose in robot frame (required by mpc solver)
        solver_target_batch_p = []
        solver_target_batch_q = []
        for env in range(n_envs):
            for robot_idx in range(len(target_list_by_env[env])):
                target = target_list_by_env[env][robot_idx]
                # get target in environment frame
                p_target_envFrame, q_target_envFrame = target.get_local_pose() 
                
                # print("local frame ",env,robot_idx, p_target_envFrame)
                # transform to robot frame
                X_target_envFrame = p_target_envFrame.tolist() + q_target_envFrame.tolist()
                X_solverTarget = transform_pose_between_frames(
                    X_target_envFrame, 
                    frame_expressed_in=X_envs[env].tolist(), 
                    frame_express_at=Xlist_rbase[env][robot_idx] # Xlist_rbase_Fenv[env][robot_idx]
                )
                p_solverTarget, q_solverTarget = X_solverTarget[:3], X_solverTarget[3:]
                solver_target_batch_p.append(np.array(p_solverTarget))
                solver_target_batch_q.append(np.array(q_solverTarget))
                # print("solver ",env,robot_idx, p_solverTarget)
                
        # make a batch of new target poses for batch solver 
        ik_goal = Pose(
            position=tensor_args.to_device(solver_target_batch_p),
            quaternion=tensor_args.to_device(solver_target_batch_q),
        )
   
        
        # update solver with new target poses from simulation
        goal_buffer.goal_pose.copy_(ik_goal)
        mpc.update_goal(goal_buffer)
        targets_update_timer += time.time() - targets_update_timer_start
        
        
        # get batch of current joint states from simulation
        joint_state_timer_start = time.time()
        # n robots x 9 
        current_state_batch:JointState = get_batch_js(robot_list, tensor_args) # full_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)        
        # n robots x 7 or n robots x 9? to check
        current_state_batch:JointState = current_state_batch.get_ordered_joint_state(mpc.kinematics.joint_names)
        joint_state_timer += time.time() - joint_state_timer_start
        
        # solve mpc problem for batch of robots 
        mpc_solver_time_start = time.time()
        result = mpc.step(current_state_batch, max_attempts=2)
        mpc_solver_timer += time.time() - mpc_solver_time_start
        
        # 
        # create action for each robot 
        action_timer_start = time.time()
        for i in range(len(robot_list)):
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
        action_timer += time.time() - action_timer_start
        # for _ in range(len(robot_list)):
        #     my_world.step(render=False)
        ctrl_loop_timer += time.time() - ctrl_loop_timer_start
        t += 1
        if t % 100 == 0:
            print(f"t = {t}")
            print(f"avg ctrl freq in last 100 steps:  {100 / ctrl_loop_timer:.2f}")
            print(f"mpc solver freq in last 100 steps: {100 / mpc_solver_timer:.2f}")
            print(f"world step freq in last 100 steps: {100 / world_step_timer:.2f}")
            print(f"targets update freq in last 100 steps: {100 / targets_update_timer:.2f}")
            print(f"joint state freq in last 100 steps: {100 / joint_state_timer:.2f}")
            print(f"action freq in last 100 steps: {100 / action_timer:.2f}")
        
            total_time_measured = mpc_solver_timer + world_step_timer + targets_update_timer + joint_state_timer + action_timer
            total_time_actual = ctrl_loop_timer
            delta = total_time_measured - total_time_actual
            print(f"total time measured: {total_time_measured:.2f}")
            print(f"total time actual: {total_time_actual:.2f}")
            print(f"delta: {delta:.2f}")
            print("In percentage %:")
            print(f"mpc solver: {100 * mpc_solver_timer / total_time_actual:.2f}")
            print(f"world step: {100 * world_step_timer / total_time_actual:.2f}")
            print(f"targets update: {100 * targets_update_timer / total_time_actual:.2f}")
            print(f"joint state: {100 * joint_state_timer / total_time_actual:.2f}")
            print(f"action: {100 * action_timer / total_time_actual:.2f}")
            # reset timers
            ctrl_loop_timer = 0
            mpc_solver_timer = 0
            world_step_timer = 0
            targets_update_timer = 0
            joint_state_timer = 0
            action_timer = 0
        
if __name__ == "__main__":
    
    main()

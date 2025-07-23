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
from collections.abc import Callable
import copy
from typing_extensions import Union
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library
import argparse

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

parser.add_argument("--robot", type=str, default="franka.yml", help="robot configuration to load")
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
# Enable the layers and stage windows in the UI
# Standard Library
import os

# Third Party
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper

############################################################


########### OV #################;;;;;


###########
EXT_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__))))
DATA_DIR = os.path.join(EXT_DIR, "data")
########### frame prim #################;;;;;


# Standard Library
from typing import Optional

# Third Party
from projects_root.examples.helper import add_extensions, add_robot_to_scene

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
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

           
class MpcPlanner:
    def __init__(self, solver: MpcSolver, solver_config: MpcSolverConfig):
        self.solver = solver
        self.solver_config = solver_config
        self.cmd_state_full = None
        # self.past_pose = None
        self.plan_goals:dict[str, Pose] = {} 
        self.ee_link_name:str = self.solver.kinematics.ee_link # end effector link name, based on the robot config
        self.constrained_links_names:list[str] = copy.copy(self.solver.kinematics.link_names) # all links that we can set goals for (except ee link), based on the robot config
        if self.ee_link_name in self.constrained_links_names: # ee link should not be in extra links, so we remove it
            self.constrained_links_names.remove(self.ee_link_name)
        # self.constrained_links_names:list[str] = copy.copy(self.solver.kinematics.link_names) # all links that we can set goals for (except ee link), based on the robot config



        retract_cfg = self.solver.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        joint_names = self.solver.rollout_fn.joint_names
        state = self.solver.rollout_fn.compute_kinematics(
            JointState.from_position(retract_cfg, joint_names=joint_names)
        )
        
        self.current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
        _initial_ee_target_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        _initial_constrained_links_target_poses = {name: state.link_poses[name] for name in self.constrained_links_names}
        goal = Goal(
            current_state=self.current_state,
            goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
            goal_pose=_initial_ee_target_pose,
            links_goal_pose=_initial_constrained_links_target_poses
        )
        
        self.plan_goals = {self.ee_link_name: _initial_ee_target_pose}
        for link_name in self.constrained_links_names:
            self.plan_goals[link_name] = _initial_constrained_links_target_poses[link_name]
        
        self.goal_buffer = self.solver.setup_solve_single(goal, 1)
        self.solver.update_goal(self.goal_buffer)
        mpc_result = self.solver.step(self.current_state, max_attempts=2)

 
       

    def _outdated_plan_goals(self, goals:dict[str, Pose]):
        """
        check if the current plan goals are outdated
        """
        for link_name, goal in goals.items():
            if link_name not in self.plan_goals or torch.norm(self.plan_goals[link_name].position - goal.position) > 1e-3 or torch.norm(self.plan_goals[link_name].quaternion - goal.quaternion) > 1e-3:
                print(f"plan goals are outdated for link {link_name}")
                return True
        return False
    
    def yield_action(self, goals:dict[str, Pose]):

        if self._outdated_plan_goals(goals):
            self.plan_goals = goals
            self.goal_buffer.goal_pose = goals[self.ee_link_name]
            for link_name in self.constrained_links_names:
                if link_name in goals:
                    self.goal_buffer.links_goal_pose[link_name] = goals[link_name]
            self.solver.update_goal(self.goal_buffer)
            # self.past_pose = goals[self.ee_link_name]


        # if self.past_pose is None \
        #     or torch.norm(goals[self.ee_link_name].position - self.past_pose.position) > 1e-3 \
        #     or torch.norm(goals[self.ee_link_name].quaternion - self.past_pose.quaternion) > 1e-3: 
            
            self.goal_buffer.goal_pose.copy_(goals[self.ee_link_name])
            self.solver.update_goal(self.goal_buffer)
            # self.past_pose = goals[self.ee_link_name]


        mpc_result = self.solver.step(self.current_state, max_attempts=2)
        return mpc_result.js_action
    
    def convert_action_to_isaac(self, sim_js_names:list[str], order_finder:Callable)->ArticulationAction:
        """
        A utility function to convert curobo action to isaac sim action (ArticulationAction).
        """
        # get only joint names that are in both:
        common_js_names = []
        idx_list = []
        for x in sim_js_names:
            if x in self.cmd_state_full.joint_names:
                idx_list.append(order_finder(x))
                common_js_names.append(x)

        cmd_state = self.cmd_state_full.get_ordered_joint_state(common_js_names)
        self.cmd_state_full = cmd_state

        art_action = ArticulationAction(
            cmd_state.position.view(-1).cpu().numpy(),
            # cmd_state.velocity.cpu().numpy(),
            joint_indices=idx_list,
        )
        return art_action

    def update_state(self, cu_js:JointState):
        if self.cmd_state_full is None:
            self.current_state.copy_(cu_js)
        else:
            current_state_partial = self.cmd_state_full.get_ordered_joint_state(
                self.solver.rollout_fn.joint_names
            )
            self.current_state.copy_(current_state_partial)
            self.current_state.joint_names = current_state_partial.joint_names
            # current_state = current_state.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        # common_js_names = []
        self.current_state.copy_(cu_js)

def main():
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()
    stage = my_world.stage

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")
    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()

    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

    # Make a target to follow
    # target = cuboid.VisualCuboid(
    #     "/World/target",
    #     position=np.array([0.5, 0, 0.5]),
    #     orientation=np.array([0, 1, 0, 0]),
    #     color=np.array([1.0, 0, 0]),
    #     size=0.05,
    # )
    
    setup_curobo_logger("warn")
    n_obstacle_cuboids = 30
    n_obstacle_mesh = 10

    # warmup curobo instance
    usd_help = UsdHelper()
    target_pose = None

    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]

    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] += 0.02

    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)

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

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    
    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].pose[2] = -10.0

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]

    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        # use_cuda_graph=True,
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
    )

    planner = MpcPlanner(MpcSolver(mpc_config), mpc_config)
    mpc = planner.solver
    
    ee_target_prim_path = "/World/target"
    ee_retract_pose = planner.plan_goals[planner.ee_link_name]
    print(f"ee_retract_pose: {ee_retract_pose}")
    _initial_ee_target_pose = np.ravel(ee_retract_pose.to_list()) # set initial ee target pose to the current ee pose
    ee_target = cuboid.VisualCuboid(
        ee_target_prim_path,
        position=_initial_ee_target_pose[:3],
        orientation=_initial_ee_target_pose[3:],
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    # create target prims for constrained links (optional):
    
    constr_link_name_to_target_prim = {}
    constr_links_targets_prims_paths = []
    for link_name in planner.constrained_links_names:
        if link_name != planner.ee_link_name:
            target_path = "/World/target_" + link_name
            constrained_link_retract_pose = np.ravel(planner.plan_goals[link_name].to_list())
            _initial_constrained_link_target_pose = constrained_link_retract_pose # set initial constrained link target pose to the current link pose
            
            color = np.random.randn(3) * 0.2
            color[0] += 0.5
            color[1] = 0.5
            color[2] = 0.0
            constr_link_name_to_target_prim[link_name] = cuboid.VisualCuboid(
                target_path,
                position=np.array(_initial_constrained_link_target_pose[:3]),
                orientation=np.array(_initial_constrained_link_target_pose[3:]),
                color=color,
                size=0.05,
            )
            constr_links_targets_prims_paths.append(target_path)
    

    # retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
    # joint_names = mpc.rollout_fn.joint_names
    # state = mpc.rollout_fn.compute_kinematics(
    #     JointState.from_position(retract_cfg, joint_names=joint_names)
    # )
    # current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
    # retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
    # goal = Goal(
    #     current_state=current_state,
    #     goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
    #     goal_pose=retract_pose,
    # )
    # goal_buffer = mpc.setup_solve_single(goal, 1)
    
    
    # mpc.update_goal(goal_buffer)
    # mpc_result = mpc.step(current_state, max_attempts=2)

    usd_help.load_stage(my_world.stage)
    init_world = False
    # cmd_state_full = None
    step = 0
    add_extensions(simulation_app, args.headless_mode)
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
            # my_world.reset()
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
        if step_index % 100 == 0:
            print("Updating world")
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                ignore_substring=[
                    robot_prim_path,
                    "/World/target",
                    "/World/defaultGroundPlane",
                    "/curobo",
                ],
                reference_prim_path=robot_prim_path,
            )
            obstacles.add_obstacle(world_cfg_table.cuboid[0])
            mpc.world_coll_checker.load_collision_model(obstacles)



        # get robot current state:
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        # js_names = robot.dof_names
        sim_js_names = robot.dof_names

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(mpc.rollout_fn.joint_names)
        planner.update_state(cu_js)

        # position and orientation of target virtual cube:
        # cube_position, cube_orientation = target.get_world_pose()
        p_ee_target, q_ee_target = ee_target.get_world_pose()
        ee_goal = Pose(
            position=tensor_args.to_device(p_ee_target),
            quaternion=tensor_args.to_device(q_ee_target),
        )
        
        # read poses of the constrained links targets (if exist) from isaac sim:
        links_goal_poses = {}
        for link_name in constr_link_name_to_target_prim.keys():
            c_p, c_rot = constr_link_name_to_target_prim[link_name].get_world_pose()
            links_goal_poses[link_name] = Pose(
                position=tensor_args.to_device(c_p),
                quaternion=tensor_args.to_device(c_rot),
            )
        # set goals for the planner:
        goals = {planner.ee_link_name: ee_goal,}
        for link_name in constr_link_name_to_target_prim.keys():
            goals[link_name] = links_goal_poses[link_name]
        
        # target_pose = Pose(tensor_args.to_device(cube_position), tensor_args.to_device(cube_orientation))
        # goals = {planner.ee_link_name: target_pose}
        planner.cmd_state_full = planner.yield_action(goals)
        art_action = planner.convert_action_to_isaac(sim_js_names, robot.get_dof_index)
        articulation_controller.apply_action(art_action)

        

############################################################

if __name__ == "__main__":
    main()
    simulation_app.close()

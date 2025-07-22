
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
from copy import copy, deepcopy
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

parser.add_argument(
    "--robot", type=str, default="dual_ur10e.yml", help="robot configuration to load"
)
args = parser.parse_args()

############################################################

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

from projects_root.examples.helper import add_extensions, add_robot_to_scene
add_extensions(simulation_app, args.headless_mode)

import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

############################################################


########### OV #################;;;;;


############################################################

            


class Plan:
    def __init__(self, cmd_idx=0, cmd_plan:Optional[JointState]=None):
        self.cmd_idx = cmd_idx
        self.cmd_plan = cmd_plan
    
    def _is_finished(self):
        return self.cmd_idx >= len(self.cmd_plan.position) - 1
    
    def consume_action(self):
        if self.cmd_plan is None or self._is_finished():
            self.cmd_idx = 0
            self.cmd_plan = None
            return None
        else:
            cmd = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1
            return cmd
            


class CuPlanner:
    
    
            
    def __init__(self,
                 motion_gen_config:MotionGenConfig, 
                 plan_config:MotionGenPlanConfig, 
                 warmup_config:dict
                ):
        self.motion_gen_config = motion_gen_config
        self.plan_config = plan_config
        self.warmup_config = warmup_config
        self.motion_gen = MotionGen(self.motion_gen_config)
        print("warming up...")
        self.motion_gen.warmup(**self.warmup_config)
        self.plan = Plan()
        
        # all constrained links that we can set goals for (ee + optional extra links):
        self.ee_link_name:str = self.motion_gen.kinematics.ee_link # end effector link name, based on the robot config
        self.constrained_links_names:list[str] = copy(self.motion_gen.kinematics.link_names) # all links that we can set goals for (except ee link), based on the robot config
        if self.ee_link_name in self.constrained_links_names: # ee link should not be in extra links, so we remove it
            self.constrained_links_names.remove(self.ee_link_name)
    
        self.plan_goals:dict[str, Pose] = {}

            
            
            
            
            
            
            
    def _plan_new(self, 
                  cu_js:JointState,
                  new_goals:dict[str, Pose],
                  )->bool:
        """
        Making a new plan. return True if success, False otherwise
        """
        ee_goal = new_goals[self.ee_link_name]
        extra_links_goals = {link_name:new_goals[link_name] for link_name in self.constrained_links_names}
        result = self.motion_gen.plan_single(
            cu_js.unsqueeze(0), ee_goal, self.plan_config.clone(), link_poses=extra_links_goals
        )
        succ = result.success.item()  # ik_result.success.item()
        if succ:
            print("planned successfully, resetting plan...")
            self.plan.cmd_plan = result.get_interpolated_plan()
            self.plan.cmd_idx = 0
            self.plan_goals = new_goals
            return True
        else:
            carb.log_warn("Plan did not converge to a solution: " + str(result.status))
            return False
    
    def _outdated_plan_goals(self, goals:dict[str, Pose]):
        """
        check if the current plan goals are outdated
        """
        for link_name, goal in goals.items():
            if link_name not in self.plan_goals or torch.norm(self.plan_goals[link_name].position - goal.position) > 1e-3 or torch.norm(self.plan_goals[link_name].quaternion - goal.quaternion) > 1e-3:
                return True
        return False
    
    def _in_move(self, joint_velocities:np.ndarray):
        """
        check if the joints are in move
        """
        # print(f"joint_velocities={joint_velocities}")
        # print(f"max(abs(joint_velocities))={np.max(np.abs(joint_velocities))}")
        return np.max(np.abs(joint_velocities)) > 0.5
    
    def yield_action(self, goals:dict[str, Pose], cu_js:JointState, joint_velocities:np.ndarray):
        """
        goals: dict of link names (both end effector link and extra links) and their updated goal poses.
        cu_js: current curobo joint state of the robot.
        joint_velocities: current joint velocities of the robot, as measured by the robot (from simulation, sensors or other sources).
        returns:
            action: Union[ArticulationAction, None]
        """
        
        PLAN_NEW = 0 # REPLAN NEXT ACTION SEQUENCE (JOINT POSITIONS)
        STOP_IN_PLACE = 1 # SEND STOP COMMAND TO JOINT CONTROLLER (VELOCICY 0)
        CONSUME_FROM_PLAN = 2 # CONTINUE THE CURRENT ACTION SEQUENCE


        if self._outdated_plan_goals(goals):
            if self._in_move(joint_velocities):
                code = STOP_IN_PLACE
            else:
                code = PLAN_NEW
        else:
            code = CONSUME_FROM_PLAN
        consume = True
        if code == PLAN_NEW:
            # print(f'planning...')
            _success = self._plan_new(cu_js, goals)
            
        elif code == STOP_IN_PLACE:
            action = JointState(
                position=cu_js.position,
                velocity=cu_js.velocity * 0.0,
                joint_names=cu_js.joint_names,
            )
            # print(f'stopping robot...')
            consume = False

        elif code == CONSUME_FROM_PLAN:
            # print(f'consuming current plan...')
            pass
        else:
            raise ValueError(f"Invalid code: {code}")
        
        if consume:
            action = self.plan.consume_action() # returns None if no more actions to consume
        
        return action
     
    def convert_plan_to_isaac(self, sim_js_names:list[str], get_dof_index:Callable):
        """
        convert curobo plan to isaac sim plan
        returns:
            isaac_sim_plan: Plan
            articulation_action_idx_list: list[int]
        """
        full_js_plan = deepcopy(self.motion_gen.get_full_js(self.plan.cmd_plan))
        # get only joint names that are in both:
        articulation_action_idx_list = []
        common_js_names = []
        for x in sim_js_names:
            if x in full_js_plan.joint_names:
                articulation_action_idx_list.append(get_dof_index(x))
                common_js_names.append(x)
        
        isaac_cmd_plan = full_js_plan.get_ordered_joint_state(common_js_names)
        return Plan(cmd_plan=isaac_cmd_plan), articulation_action_idx_list
        
    # print("Curobo is Ready")
        
    def convert_action_to_isaac(
            self, 
            action:JointState, 
            sim_js_names:list[str], 
            order_finder:Callable
        )->ArticulationAction:

        """
        convert curobo action to isaac sim action
        """
        full_js_action = deepcopy(self.motion_gen.get_full_js(action))
        # get only joint names that are in both:
        art_action_idx_list = []
        common_js_names = []
        for x in sim_js_names:
            if x in full_js_action.joint_names:
                art_action_idx_list.append(order_finder(x))
                common_js_names.append(x)
    
        full_ordered_js_action = full_js_action.get_ordered_joint_state(common_js_names)
        articulation_action = ArticulationAction(
            full_ordered_js_action.position.cpu().numpy(),
            full_ordered_js_action.velocity.cpu().numpy(),
            joint_indices=art_action_idx_list,
        )
        return articulation_action
        

def main():
    setup_curobo_logger("warn")

    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    
    usd_help = UsdHelper()
    usd_help.load_stage(my_world.stage)

    tensor_args = TensorDeviceType()
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), args.robot))["robot_cfg"]
    j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot, robot_prim_path = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = robot.get_articulation_controller()

    world_cfg_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_cfg_table.cuboid[0].pose[2] -= 0.02

    world_cfg1 = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_mesh_world()
    world_cfg1.mesh[0].name += "_mesh"
    world_cfg1.mesh[0].pose[2] = -10.5

    world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)
    usd_help.add_world_to_stage(world_cfg, base_frame="/World")
    my_world.scene.add_default_ground_plane()
    

    _motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        collision_checker_type=CollisionCheckerType.MESH,
        use_cuda_graph=True,
        interpolation_dt=0.03,
        collision_cache={"obb": 30, "mesh": 10},
        collision_activation_distance=0.025,
        fixed_iters_trajopt=True,
        maximum_trajectory_dt=0.5,
        ik_opt_iters=500,
    )
    _plan_config = MotionGenPlanConfig(
        enable_graph=False,
        enable_graph_attempt=4,
        max_attempts=10,
        time_dilation_factor=0.5,
    )
    _warmup_config = {'enable_graph':True, 'warmup_js_trajopt':False}
    planner = CuPlanner(_motion_gen_config, _plan_config, _warmup_config)

    # i = 0

    
    # link_names = planner.motion_gen.kinematics.link_names 
    ee_link_name = planner.motion_gen.kinematics.ee_link
    
    # get link poses at retract configuration:
    retract_kinematics_state = planner.motion_gen.kinematics.get_state(planner.motion_gen.get_retract_config().view(1, -1))
    links_retract_poses = retract_kinematics_state.link_pose
    ee_retract_pose = retract_kinematics_state.ee_pose
    
    ee_target_prim_path = "/World/target"
    _initial_ee_target_pose = np.ravel(ee_retract_pose.to_list()) # set initial ee target pose to the current ee pose
    ee_target = cuboid.VisualCuboid(
        ee_target_prim_path,
        position=_initial_ee_target_pose[:3],
        orientation=_initial_ee_target_pose[3:],
        color=np.array([1.0, 0, 0]),
        size=0.05,
    )

    # create new targets for new links:
    # ee_idx = link_names.index(ee_link_name)
    constr_link_name_to_target_prim = {}
    constr_links_targets_prims_paths = []
    for link_name in planner.constrained_links_names:
        if link_name != ee_link_name:
            target_path = "/World/target_" + link_name
            constrained_link_retract_pose = np.ravel(links_retract_poses[link_name].to_list())
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
    
    i = 0
    spheres = None
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        step_index = my_world.current_time_step_index
        if step_index <= 10:
            # my_world.reset()
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(x) for x in j_names]
            robot.set_joint_positions(default_config, idx_list)
            robot.set_joint_velocities(np.zeros_like(default_config), idx_list)

            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        if step_index == 50 or step_index % 1000 == 0.0:
            print("Updating world, reading w.r.t.", robot_prim_path)
            obstacles = usd_help.get_obstacles_from_stage(
                only_paths=["/World"],
                reference_prim_path=robot_prim_path,
                ignore_substring=[
                    robot_prim_path,
                    ee_target_prim_path,
                    "/World/defaultGroundPlane",
                    "/curobo",
                ]
                + constr_links_targets_prims_paths,
            ).get_collision_check_world()

            planner.motion_gen.update_world(obstacles)
            print("Updated World")
            carb.log_info("Synced CuRobo world from stage.")

        # position and orientation of target virtual cube:
        # cube_position, cube_orientation = ee_target.get_world_pose()

  
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(planner.motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = planner.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

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
                    spheres[si].set_world_pose(position=np.ravel(s.position))
                    spheres[si].set_radius(float(s.radius))
        
        p_ee_target, q_ee_target = ee_target.get_world_pose()
        ee_goal = Pose(
            position=tensor_args.to_device(p_ee_target),
            quaternion=tensor_args.to_device(q_ee_target),
        )
        
        # add link poses:
        links_goal_poses = {}
        for link_name in constr_link_name_to_target_prim.keys():
            c_p, c_rot = constr_link_name_to_target_prim[link_name].get_world_pose()
            links_goal_poses[link_name] = Pose(
                position=tensor_args.to_device(c_p),
                quaternion=tensor_args.to_device(c_rot),
            )
        
        goals = {planner.ee_link_name: ee_goal,}
        for link_name in constr_link_name_to_target_prim.keys():
            goals[link_name] = links_goal_poses[link_name]
        
        action = planner.yield_action(goals, cu_js, sim_js.velocities)
        
        if action is not None:
            isaac_action = planner.convert_action_to_isaac(action, sim_js_names, robot.get_dof_index)
            articulation_controller.apply_action(isaac_action)
         

                
    simulation_app.close()


if __name__ == "__main__":
    main()


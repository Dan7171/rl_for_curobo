 

try:
    # Third Party
    import isaacsim
except ImportError:
    pass


# Third Party
from collections.abc import Callable
from copy import copy, deepcopy
import dataclasses
from typing import Optional
from typing_extensions import List
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
import torch
from warp.context import Union

from projects_root.utils.usd_pose_helper import get_stage_poses, list_relevant_prims
from projects_root.utils.world_model_wrapper import WorldModelWrapper

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
    def __init__(self, solver:Union[MotionGen, MpcSolver], solver_config:Union[MotionGenConfig, MpcSolverConfig]):
        self.solver = solver
        self.solver_config = solver_config
        
        # setup all "goal-constrained" links (links that we can set goals for): ee (musts) + (optional) extra links:
        self.ee_link_name:str = self.solver.kinematics.ee_link # end effector link name, based on the robot config
        self.constrained_links_names:list[str] = copy(self.solver.kinematics.link_names) # all links that we can set goals for (except ee link), based on the robot config
        if self.ee_link_name in self.constrained_links_names: # ee link should not be in extra links, so we remove it
            self.constrained_links_names.remove(self.ee_link_name)
        
        # buffer for the state of the current planning goals (for ee link + extra constrained links)
        self.plan_goals:dict[str, Pose] = {} 

    def _set_goals_to_retract_state(self):
        """
        init the plan goals to be as the retract state
        """

        # get the current state of the robot (at retract configuration) :
        retract_kinematics_state = self.solver.kinematics.get_state(self.solver.get_retract_config().view(1, -1))
        links_retract_poses = retract_kinematics_state.link_pose
        ee_retract_pose = retract_kinematics_state.ee_pose

        # update the plan goals buffer accordingly:
        self.plan_goals = {self.ee_link_name: ee_retract_pose}
        for link_name in self.constrained_links_names:
            self.plan_goals[link_name] = links_retract_poses[link_name]

    def yield_action(self, **kwargs)->Optional[JointState]:
        pass
    

    def _outdated_plan_goals(self, goals:dict[str, Pose]):
        """
        check if the current plan goals are outdated
        """
        for link_name, goal in goals.items():
            if link_name not in self.plan_goals or torch.norm(self.plan_goals[link_name].position - goal.position) > 1e-3 or torch.norm(self.plan_goals[link_name].quaternion - goal.quaternion) > 1e-3:
                print(f"plan goals are outdated for link {link_name}")
                return True
        return False
    
    def convert_action_to_isaac(
            self, 
            full_js_action:JointState, 
            sim_js_names:list[str], 
            order_finder:Callable
        )-> ArticulationAction:

        """
        A utility function to convert curobo action to isaac sim action (ArticulationAction).
        """
        # get only joint names that are in both:
        art_action_idx_list = []
        common_js_names = []
        for x in sim_js_names:
            if x in full_js_action.joint_names:
                art_action_idx_list.append(order_finder(x))
                common_js_names.append(x)
    
        full_ordered_js_action = full_js_action.get_ordered_joint_state(common_js_names)
        articulation_action = ArticulationAction(
            full_ordered_js_action.position.view(-1).cpu().numpy(),
            # full_ordered_js_action.velocity.cpu().numpy(),
            joint_indices=art_action_idx_list,
        )
        return articulation_action
    

class MpcPlanner(CuPlanner):
    def __init__(self, mpc_config:MpcSolverConfig):
        self._cmd_state_full = None
        self._current_js = None

        super().__init__(MpcSolver(mpc_config), mpc_config)
        self.solver:MpcSolver = self.solver # only for linter
        self._set_goals_to_retract_state()
        
    def yield_action(self,goals:dict[str, Pose], cu_js:JointState)->Optional[JointState]:
        
        if self._outdated_plan_goals(goals):
            # update the plan goals:
            self.plan_goals = goals
            self._goal_buffer.goal_pose = goals[self.ee_link_name]
            for link_name in self.constrained_links_names:
                self._goal_buffer.links_goal_pose[link_name] = goals[link_name]
            # update the goal in solver:
            # planner_input_goals_pos = []
            # planner_input_goals_quat = []

            # for goal_name, goal_pose in goals.items():
            #     planner_input_goals_pos.append(self.solver.tensor_args.to_device(goal_pose.position))
            #     planner_input_goals_quat.append(self.solver.tensor_args.to_device(goal_pose.quaternion))    
            
            # multi_arm_positions = torch.stack(planner_input_goals_pos, dim=0).unsqueeze(0)   # [num_arms, 3]
            # multi_arm_quaternions = torch.stack(planner_input_goals_quat, dim=0).unsqueeze(0)  # [num_arms, 4]
            # ik_goal = Pose(
            #     position=multi_arm_positions,     # [1, num_arms, 3] tensor for all arms
            #     quaternion=multi_arm_quaternions  # [1, num_arms, 4] tensor for all arms
            # )
            
            # IMPORTANT: Direct assignment instead of copy_() to preserve multi-arm structure
            
            # self._goal_buffer.goal_pose = ik_goal
            self.solver.update_goal(self._goal_buffer)
            
        
        mpc_result = self.solver.step(cu_js, max_attempts=2)
        self._cmd_state_full = mpc_result.js_action
        return self._cmd_state_full
    
    def _set_goals_to_retract_state(self):
        
        
        # get the current state of the robot (at retract configuration) :
        assert isinstance(self.solver, MpcSolver) # only for linter
        retract_cfg = self.solver.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        joint_names = self.solver.rollout_fn.joint_names
        state = self.solver.rollout_fn.compute_kinematics(
            JointState.from_position(retract_cfg, joint_names=joint_names)
        )
        self._current_js = JointState.from_position(retract_cfg, joint_names=joint_names)
        
        _initial_ee_target_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq) # set target to retract
        _initial_constrained_links_target_poses = {name: state.link_poses[name] for name in self.constrained_links_names} # set target to retract
        goal = Goal(
            current_state=self._current_js,
            goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
            goal_pose=_initial_ee_target_pose,
            # links_goal_pose=_initial_constrained_links_target_poses,
        )
        self._goal_buffer = self.solver.setup_solve_single(goal, 1)
        self.solver.update_goal(self._goal_buffer)
        mpc_result = self.solver.step(self._current_js, max_attempts=2)

        # update the plan goals buffer accordingly:
        self.plan_goals = {self.ee_link_name: _initial_ee_target_pose}
        for link_name in self.constrained_links_names:
            self.plan_goals[link_name] = _initial_constrained_links_target_poses[link_name]
        
        self._cmd_state_full = None

    def convert_action_to_isaac(self, full_js_action:JointState, sim_js_names:list[str], order_finder:Callable)->ArticulationAction:
        """
        A utility function to convert curobo action to isaac sim action (ArticulationAction).
        """
        # get only joint names that are in both:
        art_action_idx_list = []
        common_js_names = []
        for x in sim_js_names:
            if x in full_js_action.joint_names:
                art_action_idx_list.append(order_finder(x))
                common_js_names.append(x)

        full_ordered_js_action = full_js_action.get_ordered_joint_state(common_js_names)
        self._cmd_state_full = full_ordered_js_action

        articulation_action = ArticulationAction(
            full_ordered_js_action.position.view(-1).cpu().numpy(),
            # full_ordered_js_action.velocity.cpu().numpy(),
            joint_indices=art_action_idx_list,
        )
        return articulation_action
    
class CumotionPlanner(CuPlanner):
                
    def __init__(self,
                 motion_gen_config:MotionGenConfig, 
                 plan_config:MotionGenPlanConfig, 
                 warmup_config:dict
                ):
        """
        Cumotion planning kit. Can accept goals for end effector and optional extra links (e.g. "constrained" links).
        To use with multi arm, pass inputs (robot config, urdf, etc) as in this example: curobo/examples/isaac_sim/multi_arm_reacher.py
        robot config for example: curobo/src/curobo/content/configs/robot/dual_ur10e.yml

        To use with single arm, pass inputs (robot config, urdf, etc) as in this example: curobo/examples/isaac_sim/motion_gen_reacher.py
        robot config for example: curobo/src/curobo/content/configs/robot/ur10e.yml or franka.yml
        """
        super().__init__(MotionGen(motion_gen_config), motion_gen_config)
        self.plan_config = plan_config
        self.warmup_config = warmup_config
        self.solver:MotionGen = self.solver # only for linter 
        print("warming up...")
        self.solver.warmup(**self.warmup_config)
        self.plan = Plan()
        self._set_goals_to_retract_state()
    

            
    def _plan_new(self, 
                  cu_js:JointState,
                  new_goals:dict[str, Pose],
                  )->bool:
        """
        Making a new plan. return True if success, False otherwise
        """
        ee_goal = new_goals[self.ee_link_name]
        extra_links_goals = {link_name:new_goals[link_name] for link_name in self.constrained_links_names}
        result = self.solver.plan_single(
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

        if action is not None:
            action = deepcopy(self.solver.get_full_js(action))
        
        return action
         
    
class CuAgent:
    def __init__(self, 
                planner:CuPlanner,
                cu_world_wrapper_cfg:dict,
                base_pos=[0.0,0.0,0.0], 
                base_quat=[1,0,0,0]):
        
        self.planner = planner        
        self.base_pos = base_pos
        self.base_quat = base_quat 
        self.base_pose = base_pos + base_quat


        if "verbosity" in cu_world_wrapper_cfg:
            verbosity = cu_world_wrapper_cfg["verbosity"] 
        else:
            verbosity = 0
        # See wrapper's docstring to understand the motivation for the wrapper.
        _solver_wm = self.planner.solver.world_coll_checker.world_model
        assert isinstance(_solver_wm, WorldConfig) # only for linter
        self.cu_world_wrapper = WorldModelWrapper(
            world_config=_solver_wm,
            X_robot_W=np.array(self.base_pose), # robot base frame in world frame
            verbosity=verbosity
        )
        
    def reset_col_model_from_isaac_sim(self, usd_help:UsdHelper, robot_prim_path:str, ignore_substrings:List[str]):

        # Get world config from simulation
        isaac_cu_world_R = usd_help.get_obstacles_from_stage( 
            only_paths=['/World'], # look for obs only under the world prim path
            reference_prim_path=robot_prim_path, # obstacles are expressed in robot frame! (not world frame). That's why we marked it with 'R'
            ignore_substring=ignore_substrings
        )
        return self.reset_col_model(isaac_cu_world_R)

    def update_col_model_from_isaac_sim(self, 
            robot_prim_path:str,
            usd_help:UsdHelper, 
            ignore_list:List[str]=['/World/defaultGroundPlane'],
            paths_to_search_obs_under:List[str]=['/World']
            ):
        """
        Sensing obs from simulation and updating the world model
        """

        # get poses of all obstacles in the world (in world frame)
        pose_dict = get_stage_poses(
            usd_helper=usd_help,
            only_paths=paths_to_search_obs_under,
            reference_prim_path='/World', # poses are expressed in world frame
            ignore_substring=ignore_list,
        )
        
        # Fast pose update (only pose updates, no world-model re-initialization as before)
        self.cu_world_wrapper.update_from_pose_dict(pose_dict)
        current_paths = set(
            list_relevant_prims(usd_help, paths_to_search_obs_under, ignore_list)
        )

        new_paths = current_paths - self.cu_world_wrapper.get_known_prims()
        
        if new_paths: # Here we are being LAZY! we call expensive get_obstacled_from_stage() only if there are new obstacles!
            # print(f"[NEW OBSTACLES] {new_paths}")
            
            # Get basic (non-collision check) WorldConfig from stage
            new_world_cfg:WorldConfig = usd_help.get_obstacles_from_stage(
                only_paths=list(new_paths),
                reference_prim_path=robot_prim_path,
                ignore_substring=ignore_list,
            )

            # Convert to collision check world
            new_world_cfg = new_world_cfg.get_collision_check_world()

            if new_world_cfg.objects:  # add only if we got actual obstacles
                self.cu_world_wrapper.add_new_obstacles_from_cu_world(
                    cu_world_R=new_world_cfg,
                    silent=False,
                )
                # Track real obstacle names so future pose updates work
                for obj in new_world_cfg.objects:
                    self.cu_world_wrapper.add_prim_to_known(obj.name)
    
    def reset_col_model(self, cu_world_R:WorldConfig):
        """
        Reset the collision model (with obstacles) from an input world config.
        To use not in simulation, pass your custom world config from real world.
        To use in isaac sim, see reset_col_model_from_isaac_sim()
        Args:
            cu_world_R: WorldConfig object that contains the current obstacle poses.
        """
        
        # Convert raw WorldConfig to collision check world WorldConfig! (Must!)
        cu_col_world_R = self.cu_world_wrapper.initialize_from_cu_world(cu_world_R)        
        # Update MPC world collision checker with the initialized world
        assert self.planner.solver.world_coll_checker is not None # only for linter (it's not None) 
        self.planner.solver.world_coll_checker.load_collision_model(cu_col_world_R) 
        
        # Set the collision checker reference in the wrapper
        self.cu_world_wrapper.set_collision_checker(self.planner.solver.world_coll_checker)
        
        # Record the prims that are currently considered obstacles (for easy lookup later when checking if update is needed)
        self.cu_world_wrapper.set_known_prims()

        print("WorldModelWrapper reset finished successfully!")
        print(f"Known prims in collision world: {self.cu_world_wrapper.get_known_prims()}")        



def main(meta_cfg_path):
    
    meta_cfg = load_yaml(meta_cfg_path)
    agent_cfg = meta_cfg["agents"][0]
    planner_type = agent_cfg["planner"] 
    robot_cfg_path = agent_cfg["robot"] if "robot" in agent_cfg else join_path(get_robot_configs_path(), args.robot)
    robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]

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
    
    if planner_type == 'cumotion':
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
        
        planner = CumotionPlanner(_motion_gen_config, _plan_config, _warmup_config)
    
    else:
        if "link_names" in robot_cfg["kinematics"]:
            num_constrained_links = len(robot_cfg["kinematics"]["link_names"])
        else:
            num_constrained_links = 0
        num_arms = num_constrained_links + 1 # +1 for the end effector link
        meta_cfg["general"]["solver_cfgs"]["mpc"]["num_arms"] = num_arms
        _mpc_config = MpcSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            **meta_cfg["general"]["solver_cfgs"]["mpc"],
            # use_cuda_graph=True,
            # use_cuda_graph=False,
            # use_cuda_graph_metrics=True,
            # use_cuda_graph_full_step=False,
            # self_collision_check=True,
            # collision_checker_type=CollisionCheckerType.MESH,
            # collision_cache={"obb": 30, "mesh": 10},
            # use_mppi=True,
            # use_lbfgs=False,
            # use_es=False,
            # store_rollouts=True,
            # step_dt=0.02,
            # plot_costs=True,
        )

        planner = MpcPlanner(_mpc_config)

    cu_agent = CuAgent(planner, cu_world_wrapper_cfg={"verbosity":0})

    # # get link poses at retract configuration:
    # retract_kinematics_state = planner.solver.kinematics.get_state(planner.solver.get_retract_config().view(1, -1))
    # links_retract_poses = retract_kinematics_state.link_pose
    # ee_retract_pose = retract_kinematics_state.ee_pose
    
    ee_target_prim_path = "/World/target"
    ee_retract_pose = planner.plan_goals[planner.ee_link_name]
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
    
    cu_world_never_add = [
        robot_prim_path,
        ee_target_prim_path,
        "/World/defaultGroundPlane",
        "/curobo", 
        *constr_links_targets_prims_paths
        ]
    cu_agent.reset_col_model_from_isaac_sim(usd_help, robot_prim_path, ignore_substrings=cu_world_never_add)
    cu_world_never_update = [] # objects which we assume that are added in reset_col_model_from_isaac_sim, but we don't want to update them (e.g. because they are static)
    
    i = 0
    spheres = None
    cmd_state_full = None # mpc only
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

        cu_agent.update_col_model_from_isaac_sim(robot_prim_path, 
                                                 usd_help, 
                                                 ignore_list=cu_world_never_add+cu_world_never_update, 
                                                 paths_to_search_obs_under=["/World"]
                                                 )

  
        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        sim_js_names = robot.dof_names
        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities), # * 0.0? 
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=sim_js_names,
        )
        if isinstance(planner, CumotionPlanner):
            cu_js = cu_js.get_ordered_joint_state(planner.solver.kinematics.joint_names)
        
        elif isinstance(planner, MpcPlanner): # todo: try for both mpc and cumotion as in cumotion (if)
            cu_js = cu_js.get_ordered_joint_state(planner.solver.rollout_fn.joint_names)
            if planner._cmd_state_full is None:
                planner._current_js.copy_(cu_js)
            else:
                current_state_partial = planner._cmd_state_full.get_ordered_joint_state(
                    planner.solver.rollout_fn.joint_names
                )
                planner._current_js.copy_(current_state_partial)
                planner._current_js.joint_names = current_state_partial.joint_names
            common_js_names = []
            planner._current_js.copy_(cu_js)

        if args.visualize_spheres and step_index % 2 == 0:
            sph_list = planner.solver.kinematics.get_robot_as_spheres(cu_js.position)

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
        
        # read pose of the ee link's target (if exist) from isaac sim:
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
        
        # yield action from the planner:
        if isinstance(planner, CumotionPlanner):
            action = planner.yield_action(goals, cu_js, sim_js.velocities)
        
        elif isinstance(planner, MpcPlanner):
            action = planner.yield_action(goals, cu_js)
        
        if action is not None:
            
            isaac_action = planner.convert_action_to_isaac(action, sim_js_names, robot.get_dof_index)
            
            articulation_controller.apply_action(isaac_action)
        
         

                
    simulation_app.close()


if __name__ == "__main__":
    main(meta_cfg_path='projects_root/experiments/benchmarks/cfgs/meta_cfg.yml')
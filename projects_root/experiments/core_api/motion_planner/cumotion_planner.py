from collections.abc import Callable
import time
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
import numpy as np
import torch
from projects_root.experiments.core_api.motion_planner.motion_planner import MotionPlanner



class CumotionPlanner(MotionPlanner):
    PLAN_NEW = 0 # REPLAN NEXT ACTION SEQUENCE (JOINT POSITIONS)
    STOP_IN_PLACE = 1 # SEND STOP COMMAND TO JOINT CONTROLLER (VELOCICY 0)
    KEEP_WITH_PLAN = 2 # CONTINUE THE CURRENT ACTION SEQUENCE
    def __init__(self, 
                 motion_gen_config:MotionGenConfig, 
                 motion_gen_plan_config:MotionGenPlanConfig,
                 warmup_config:dict={}
                 ):
        
        super().__init__(motion_gen_config.tensor_args)
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(**warmup_config)
        print(f"debug: motion_gen warmup done")
        self.motion_gen_plan_config = motion_gen_plan_config
        
        self.current_plan = []
        self._next_action_idx = 0

        self._idx_list = [] # some internal variables for post processing the plan
        self._common_js_names = [] # some internal variables for post processing the plan
        
        # here we put all the names of links which we set a goal pose for (the ee link + all the links in the kinematics). See example in curobo/examples/isaac_sim/multi_arm_reacher.py
        self._ee_link_name = self.motion_gen.kinematics.ee_link
        self._extra_link_names_with_goal_pose = [name for name in self.motion_gen.kinematics.link_names if name != self._ee_link_name]
        self.links_with_goals = [self._ee_link_name] + self._extra_link_names_with_goal_pose
        
        self.current_solver_goal_poses = {name:np.array([0,0,0,1,0,0,0]) for name in self.links_with_goals} # set initial default pose
        print(f"debug: this is a {len(self.links_with_goals)}-target centralized motion planner")

    

    def _consume_action_from_plan(self) -> JointState:
        a = self.current_plan[self._next_action_idx]
        print(f"debug: consuming action from plan: {a.position}")
        self._next_action_idx += 1
        return a
    
    def yield_action(
            self, 
            force_plan:bool,
            force_stop:bool,  
            curobo_joint_state:JointState, 
            current_joint_velocities:np.ndarray,
            new_target_poses:dict[str, np.ndarray], 
            sim_js_names:list[str],
            get_dof_index:Callable[[str], int],
        )-> JointState | None:

        # pre check 
        code = self.KEEP_WITH_PLAN
        if force_plan or self._next_action_idx >= len(self.current_plan):
            code = self.PLAN_NEW
 
        elif force_stop:
            code = self.STOP_IN_PLACE
        else:
            for link_name, new_goal_pose in new_target_poses.items():
                goal_changed = np.linalg.norm(new_goal_pose - self.current_solver_goal_poses[link_name]) > 1e-3
                if goal_changed:
                    effectively_stopped = np.max(np.abs(current_joint_velocities)) < 0.5
                    if effectively_stopped:
                        code = self.PLAN_NEW
                    else:
                        code = self.STOP_IN_PLACE
                    break

        match code:
            
            case self.PLAN_NEW:
                
                # solver input poses
                self.current_solver_goal_poses = {name:new_target_poses[name] for name in self.links_with_goals}
                _ee_goal_pose = self._convert_np_to_Pose(self.current_solver_goal_poses[self._ee_link_name])
                _extra_link_goal_poses = {name:self._convert_np_to_Pose(self.current_solver_goal_poses[name]) for name in self._extra_link_names_with_goal_pose}
                
                # plan
                result = self.motion_gen.plan_single(
                    curobo_joint_state.unsqueeze(0), 
                    _ee_goal_pose, 
                    self.motion_gen_plan_config.clone(), 
                    link_poses=_extra_link_goal_poses
                )
                succ = result.success.item()  # ik_result.success.item()
                print(f"debug: plan successful?: {succ}")
                
                if succ:
                    
                    #########
                    # post process the plan (dont get why yet)
                    cmd_plan = result.get_interpolated_plan()
                    cmd_plan = self.motion_gen.get_full_js(cmd_plan)
                    # get only joint names that are in both:
                    self._idx_list = []
                    self._common_js_names = []
                    for x in sim_js_names:
                        if x in cmd_plan.joint_names:
                            self._idx_list.append(get_dof_index(x))
                            self._common_js_names.append(x)
                    self.current_plan = cmd_plan.get_ordered_joint_state(self._common_js_names)
                    #########

                    self._next_action_idx = 0
                    return self._consume_action_from_plan()
                
                else:
                    return None

            case self.STOP_IN_PLACE:
                print(f"debug: warning!!! stopping in place")
                # return JointState(position=curobo_joint_state.position, velocity=[0.0 for _ in range(len(curobo_joint_state))], joint_names=curobo_joint_state.joint_names)
                return None
            
            case self.KEEP_WITH_PLAN:
                return self._consume_action_from_plan()
            
            case _:
                raise ValueError(f"Invalid code: {code}")



    
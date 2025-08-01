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
# Standard Library
from dataclasses import dataclass
import datetime
import os
from typing import Any, Dict, List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
# CuRobo
from curobo.geom.sdf.world import WorldCollision
from curobo.rollout.cost.cost_base import CostConfig
from curobo.rollout.cost.dist_cost import DistCost, DistCostConfig
from curobo.rollout.cost.pose_cost import PoseCost, PoseCostConfig, PoseCostMetric
from curobo.rollout.cost.pose_cost_multi_arm import PoseCostMultiArm, PoseCostMultiArmConfig
from curobo.rollout.cost.straight_line_cost import StraightLineCost
from curobo.rollout.cost.zero_cost import ZeroCost
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from curobo.rollout.rollout_base import Goal, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.tensor import T_BValue_float, T_BValue_int
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import cat_max
from curobo.util.torch_utils import get_torch_jit_decorator
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import get_topics
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor

# Local Folder
from .arm_base import ArmBase, ArmBaseConfig, ArmCostConfig
from scipy.spatial.transform import Rotation as R


@dataclass
class ArmReacherMetrics(RolloutMetrics):
    cspace_error: Optional[T_BValue_float] = None
    position_error: Optional[T_BValue_float] = None
    rotation_error: Optional[T_BValue_float] = None
    pose_error: Optional[T_BValue_float] = None
    goalset_index: Optional[T_BValue_int] = None
    null_space_error: Optional[T_BValue_float] = None

    def __getitem__(self, idx):
        d_list = [
            self.cost,
            self.constraint,
            self.feasible,
            self.state,
            self.cspace_error,
            self.position_error,
            self.rotation_error,
            self.pose_error,
            self.goalset_index,
            self.null_space_error,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return ArmReacherMetrics(*idx_vals)

    def clone(self, clone_state=False):
        if clone_state:
            raise NotImplementedError()
        return ArmReacherMetrics(
            cost=None if self.cost is None else self.cost.clone(),
            constraint=None if self.constraint is None else self.constraint.clone(),
            feasible=None if self.feasible is None else self.feasible.clone(),
            state=None if self.state is None else self.state,
            cspace_error=None if self.cspace_error is None else self.cspace_error.clone(),
            position_error=None if self.position_error is None else self.position_error.clone(),
            rotation_error=None if self.rotation_error is None else self.rotation_error.clone(),
            pose_error=None if self.pose_error is None else self.pose_error.clone(),
            goalset_index=None if self.goalset_index is None else self.goalset_index.clone(),
            null_space_error=(
                None if self.null_space_error is None else self.null_space_error.clone()
            ),
        )


@dataclass
class ArmReacherCostConfig(ArmCostConfig):
    pose_cfg: Optional[PoseCostConfig] = None
    cspace_cfg: Optional[DistCostConfig] = None
    straight_line_cfg: Optional[CostConfig] = None
    zero_acc_cfg: Optional[CostConfig] = None
    zero_vel_cfg: Optional[CostConfig] = None
    zero_jerk_cfg: Optional[CostConfig] = None
    link_pose_cfg: Optional[PoseCostConfig] = None
    # custom_cfg is inherited from ArmCostConfig

    @staticmethod
    def _get_base_keys():
        base_k = ArmCostConfig._get_base_keys()
        # add new cost terms:
        new_k = {
            "pose_cfg": PoseCostConfig,  # Revert to single config
            "cspace_cfg": DistCostConfig,
            "straight_line_cfg": CostConfig,
            "zero_acc_cfg": CostConfig,
            "zero_vel_cfg": CostConfig,
            "zero_jerk_cfg": CostConfig,
            "link_pose_cfg": PoseCostConfig,
        }
        new_k.update(base_k)
        return new_k

    @staticmethod
    def from_dict(
        data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        enable_auto_discovery: bool = False,
    ):
        k_list = ArmReacherCostConfig._get_base_keys()
        
        # Handle multi-arm pose configuration conditionally
        if "pose_cfg" in data_dict and isinstance(data_dict["pose_cfg"], dict):
            # Check if num_arms is already set in pose_cfg (from external detection)
            num_arms_from_cfg = data_dict["pose_cfg"].get("num_arms", 1)
            
            if num_arms_from_cfg > 1:
                # Use PoseCostMultiArmConfig for multi-arm setup
                k_list = k_list.copy()  # Make a copy to avoid modifying the original
                k_list["pose_cfg"] = PoseCostMultiArmConfig
                print(f"Using multi-arm pose cost for {num_arms_from_cfg}-arm system")
        
        data = ArmCostConfig._get_formatted_dict(
            data_dict,
            k_list,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        
        # Handle custom costs with auto-discovery (inherited from ArmCostConfig)
        custom_dict = data_dict.get("custom", {})
        data["custom_cfg"] = ArmCostConfig._parse_custom_costs(
            custom_dict, 
            tensor_args, 
            enable_auto_discovery=enable_auto_discovery
        )
        
        return ArmReacherCostConfig(**data)


@dataclass
class ArmReacherConfig(ArmBaseConfig):
    cost_cfg: ArmReacherCostConfig
    constraint_cfg: ArmReacherCostConfig
    convergence_cfg: ArmReacherCostConfig

    @staticmethod
    def cost_from_dict(
        cost_data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        enable_auto_discovery: bool = False,
    ):
        return ArmReacherCostConfig.from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
            enable_auto_discovery=enable_auto_discovery,
        )


@get_torch_jit_decorator()
def _compute_g_dist_jit(rot_err_norm, goal_dist):
    # goal_cost = goal_cost.view(cost.shape)
    # rot_err_norm = rot_err_norm.view(cost.shape)
    # goal_dist = goal_dist.view(cost.shape)
    g_dist = goal_dist.unsqueeze(-1) + 10.0 * rot_err_norm.unsqueeze(-1)
    return g_dist


class ArmReacher(ArmBase, ArmReacherConfig):
    """
    .. inheritance-diagram:: curobo.rollout.arm_reacher.ArmReacher
    """

    @profiler.record_function("arm_reacher/init")
    def __init__(self, config: Optional[ArmReacherConfig] = None):
        if config is not None:
            ArmReacherConfig.__init__(self, **vars(config))
        ArmBase.__init__(self)

        # self.goal_state = None
        # self.goal_ee_pos = None
        # self.goal_ee_rot = None
        # self.goal_ee_quat = None
        self._compute_g_dist = False
        self._n_goalset = 1

        if self.cost_cfg.cspace_cfg is not None:
            self.cost_cfg.cspace_cfg.dof = self.d_action
            # self.cost_cfg.cspace_cfg.update_vec_weight(self.dynamics_model.cspace_distance_weight)
            self.dist_cost = DistCost(self.cost_cfg.cspace_cfg)
        if self.cost_cfg.pose_cfg is not None:
            self.cost_cfg.pose_cfg.waypoint_horizon = self.horizon
            
            # Check if this is a multi-arm setup from pose config
            num_arms = getattr(self.cost_cfg.pose_cfg, 'num_arms', 1)
            if num_arms > 1:
                # Use PoseCostMultiArm for multi-arm setup (config should already be PoseCostMultiArmConfig)
                self.goal_cost = PoseCostMultiArm(self.cost_cfg.pose_cfg)
                print(f"Using multi-arm pose cost for {num_arms}-arm system")
            else:
                self.goal_cost = PoseCost(self.cost_cfg.pose_cfg)
                
            if self.cost_cfg.link_pose_cfg is None:
                log_info(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.cost_cfg.link_pose_cfg = self.cost_cfg.pose_cfg
        self._link_pose_costs = {}

        if self.cost_cfg.link_pose_cfg is not None:
            # For link pose costs, always use regular PoseCostConfig even if main pose cost is multi-arm
            link_pose_config = self.cost_cfg.link_pose_cfg
            if hasattr(link_pose_config, 'num_arms'):
                # Convert multi-arm config to regular config for individual link poses
                link_pose_config = PoseCostConfig(
                    cost_type=link_pose_config.cost_type,
                    use_metric=link_pose_config.use_metric,
                    project_distance=link_pose_config.project_distance,
                    run_vec_weight=getattr(link_pose_config, 'run_vec_weight', None),
                    use_projected_distance=link_pose_config.use_projected_distance,
                    offset_waypoint=getattr(link_pose_config, 'offset_waypoint', [0, 0, 0, 0, 0, 0]),
                    offset_tstep_fraction=link_pose_config.offset_tstep_fraction,
                    waypoint_horizon=link_pose_config.waypoint_horizon,
                    weight=link_pose_config.weight,
                    vec_weight=link_pose_config.vec_weight,
                    vec_convergence=link_pose_config.vec_convergence,
                    run_weight=link_pose_config.run_weight,
                    return_loss=link_pose_config.return_loss,
                    terminal=link_pose_config.terminal,
                    tensor_args=link_pose_config.tensor_args
                )
            
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_costs[i] = PoseCost(link_pose_config)
        if self.cost_cfg.straight_line_cfg is not None:
            self.straight_line_cost = StraightLineCost(self.cost_cfg.straight_line_cfg)
        if self.cost_cfg.zero_vel_cfg is not None:
            self.zero_vel_cost = ZeroCost(self.cost_cfg.zero_vel_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_vel_cost.hinge_value is not None:
                self._compute_g_dist = True
        if self.cost_cfg.zero_acc_cfg is not None:
            self.zero_acc_cost = ZeroCost(self.cost_cfg.zero_acc_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_acc_cost.hinge_value is not None:
                self._compute_g_dist = True

        if self.cost_cfg.zero_jerk_cfg is not None:
            self.zero_jerk_cost = ZeroCost(self.cost_cfg.zero_jerk_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_jerk_cost.hinge_value is not None:
                self._compute_g_dist = True

        self.z_tensor = torch.tensor(
            0, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self._link_pose_convergence = {}

        if self.convergence_cfg.pose_cfg is not None:
            self.pose_convergence = PoseCost(self.convergence_cfg.pose_cfg)
            if self.convergence_cfg.link_pose_cfg is None:
                log_warn(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.convergence_cfg.link_pose_cfg = self.convergence_cfg.pose_cfg

        if self.convergence_cfg.link_pose_cfg is not None:
            # For convergence link pose costs, always use regular PoseCostConfig even if main pose cost is multi-arm
            conv_link_pose_config = self.convergence_cfg.link_pose_cfg
            if hasattr(conv_link_pose_config, 'num_arms'):
                # Convert multi-arm config to regular config for individual link poses
                conv_link_pose_config = PoseCostConfig(
                    cost_type=conv_link_pose_config.cost_type,
                    use_metric=conv_link_pose_config.use_metric,
                    project_distance=conv_link_pose_config.project_distance,
                    run_vec_weight=getattr(conv_link_pose_config, 'run_vec_weight', None),
                    use_projected_distance=conv_link_pose_config.use_projected_distance,
                    offset_waypoint=getattr(conv_link_pose_config, 'offset_waypoint', [0, 0, 0, 0, 0, 0]),
                    offset_tstep_fraction=conv_link_pose_config.offset_tstep_fraction,
                    waypoint_horizon=conv_link_pose_config.waypoint_horizon,
                    weight=conv_link_pose_config.weight,
                    vec_weight=conv_link_pose_config.vec_weight,
                    vec_convergence=conv_link_pose_config.vec_convergence,
                    run_weight=conv_link_pose_config.run_weight,
                    return_loss=conv_link_pose_config.return_loss,
                    terminal=conv_link_pose_config.terminal,
                    tensor_args=conv_link_pose_config.tensor_args
                )
                
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_convergence[i] = PoseCost(conv_link_pose_config)
        if self.convergence_cfg.cspace_cfg is not None:
            self.convergence_cfg.cspace_cfg.dof = self.d_action
            self.cspace_convergence = DistCost(self.convergence_cfg.cspace_cfg)

        # check if g_dist is required in any of the cost terms:
        self.update_params(Goal(current_state=self._start_state))

        # Initialize custom costs for arm_reacher
        self._custom_arm_reacher_costs = {}
        if (hasattr(self.cost_cfg, 'custom_cfg') and 
            self.cost_cfg.custom_cfg is not None and 
            "arm_reacher" in self.cost_cfg.custom_cfg):
            for cost_name, cost_info in self.cost_cfg.custom_cfg["arm_reacher"].items():
                try:
                    cost_class = cost_info["cost_class"]
                    cost_config = cost_info["cost_config"]
                    cost_instance = cost_class(cost_config)
                    self._custom_arm_reacher_costs[cost_name] = cost_instance
                    log_info(f"Initialized custom arm_reacher cost: {cost_name}")
                except Exception as e:
                    log_error(f"Failed to initialize custom arm_reacher cost {cost_name}: {e}")

    def cost_fn(self, state: KinematicModelState, action_batch=None):
        """
        Compute cost given that state dictionary and actions


        :class:`curobo.rollout.cost.PoseCost`
        :class:`curobo.rollout.cost.DistCost`

        """
        state_batch = state.state_seq
        with profiler.record_function("cost/base"):
            # get ArmBase cost
            cost_dict:dict = super(ArmReacher, self).cost_fn(state, action_batch, return_dict=True)
        
        # cost_list = list(cost_dict.values())
        ee_pos_batch, ee_quat_batch = state.ee_pos_seq, state.ee_quat_seq
        g_dist = None
        with profiler.record_function("cost/pose"):
            if (
                self._goal_buffer.goal_pose.position is not None
                and self.cost_cfg.pose_cfg is not None
                and self.goal_cost.enabled
            ):
                # Check if this is a multi-arm pose cost that needs special handling
                from curobo.rollout.cost.pose_cost_multi_arm import PoseCostMultiArm
                is_multi_arm_pose_cost = isinstance(self.goal_cost, PoseCostMultiArm)
                
                # Format end-effector data for multi-arm pose cost if needed
                if is_multi_arm_pose_cost:
                    # For multi-arm pose cost, construct 4D tensors from link poses
                    num_arms = getattr(self.goal_cost, 'num_arms', 2)  # Default to 2 for dual-arm
                    multi_arm_ee_pos, multi_arm_ee_quat = self._format_multi_arm_ee_data(state, num_arms)
                    ee_pos_for_cost = multi_arm_ee_pos
                    ee_quat_for_cost = multi_arm_ee_quat
                    
                    # Minimal debug output (very reduced frequency)
                    if not hasattr(self, '_debug_ee_poses_count'):
                        self._debug_ee_poses_count = 0
                    self._debug_ee_poses_count += 1
                    
                    if self._debug_ee_poses_count % 5000 == 0:  # Very reduced frequency
                        batch_size, horizon, num_arms_tensor, _ = multi_arm_ee_pos.shape
                        print(f"Multi-arm EE debug: {num_arms_tensor} arms, shapes pos={multi_arm_ee_pos.shape}, quat={multi_arm_ee_quat.shape}")
                else:
                    # For single-arm pose cost, use the regular ee_pos_batch and ee_quat_batch
                    ee_pos_for_cost = ee_pos_batch
                    ee_quat_for_cost = ee_quat_batch
                
                if self._compute_g_dist:
                    goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward_out_distance(
                        ee_pos_for_cost,
                        ee_quat_for_cost,
                        self._goal_buffer,
                    )

                    g_dist = _compute_g_dist_jit(rot_err_norm, goal_dist)
                else:
                    goal_cost = self.goal_cost.forward(
                        ee_pos_for_cost, ee_quat_for_cost, self._goal_buffer
                    )
                
                if goal_cost is not None:                    
                    goal_cost[torch.isinf(goal_cost)] = 100000.0
                    # goal_cost = torch.maximum(goal_cost, t)
                # cost_list.append(goal_cost)
                cost_dict["goal"] = goal_cost
                
                
                # Set priorities using dynamic obs cost
                modified_dyn_obs_cost = False
                if 'dynamic_obs_cost' in self._custom_arm_base_costs.keys() and 'dynamic_obs_cost' in cost_dict:
                    if self._custom_arm_base_costs['dynamic_obs_cost'].prior_rule != 'none':
                        dyn_cost = self._custom_arm_base_costs['dynamic_obs_cost']
                        old_dyn_obs_cost = cost_dict['dynamic_obs_cost']
                        
                        
                        robot_id = dyn_cost.robot_id
                        robot_context = get_topics().get_default_env()[robot_id]
                        if dyn_cost.prior_rule == 'random':
                            raise NotImplementedError("Random prioritization rule not implemented")
                        else:
                            if dyn_cost.prior_rule == 'pose':    
                                pos_err = 0
                                rot_err = 0
                                
                                for link_name in robot_context['link_name_to_pose'].keys():
                                    link_pose = robot_context['link_name_to_pose'][link_name]
                                    target_name = robot_context['name_link_to_target'][link_name]
                                    target_pose = robot_context['target_name_to_pose'][target_name]
                                    if dyn_cost.prior_pos_to_rot_ratio > 0:
                                        link_pos_err = np.linalg.norm(link_pose[0] - target_pose[0])
                                        pos_err += link_pos_err
                                    if dyn_cost.prior_pos_to_rot_ratio < 1:
                                        rot_link = R.from_quat(link_pose[1]).as_euler('xyz', degrees=True)
                                        rot_target = R.from_quat(target_pose[1]).as_euler('xyz', degrees=True)
                                        rot_diff = rot_target - rot_link
                                        rot_diff_scaled = (rot_diff + 180) % 360 - 180 # err in range [-180, 180]                                
                                        rot_err += sum(np.abs(rot_diff_scaled)) / 3 # absolute error mean over roll pitch yaw (3 axis)
                                
                                
                                n_links = len(robot_context['link_name_to_pose'].keys())
                                if n_links != 0:
                                    pos_err = pos_err / n_links # euclidean distance (link to target)

                                    pos_err_norm = pos_err / dyn_cost.prior_p_err_impact_rad # normalized to [0,1]
                                    pos_err_norm_clipped = min(1, pos_err_norm) 

                                    rot_err = rot_err / n_links # mean angular error per axis (roll pitch yaw)                                    
                                    rot_err_norm = rot_err / dyn_cost.prior_rot_err_impact_angle
                                    rot_err_norm_clipped = min(1, rot_err_norm) # normalize to [0,1]
                                    
                                    
                                    pose_err_norm = dyn_cost.prior_pos_to_rot_ratio * pos_err_norm_clipped + (1 - dyn_cost.prior_pos_to_rot_ratio) * rot_err_norm_clipped # 0 <= pose_err_norm <= 1
                                    
                                   
                                    b = 0.5 
                                    w = 1 - (1-pose_err_norm) ** b # 0 <= w <=1 (for any b > 0). b effects the shape of the function.
                                    w = max(w, dyn_cost.prior_keep_lower_bound)
                                    assert w >= 0 and w <= 1
                                    new_dyn_obs_cost = w * old_dyn_obs_cost

                                    
                                    cost_dict['dynamic_obs_cost'] = new_dyn_obs_cost        
                                    modified_dyn_obs_cost = True
                      


        with profiler.record_function("cost/link_poses"):
            if self._goal_buffer.links_goal_pose is not None and self.cost_cfg.pose_cfg is not None:
                link_poses = state.link_pose

                for k in self._goal_buffer.links_goal_pose.keys():
                    if k != self.kinematics.ee_link:
                        current_fn = self._link_pose_costs[k]
                        if current_fn.enabled:
                            # get link pose
                            current_pose = link_poses[k].contiguous()
                            current_pos = current_pose.position
                            current_quat = current_pose.quaternion

                            c = current_fn.forward(current_pos, current_quat, self._goal_buffer, k)
                            # cost_list.append(c)
                            cost_dict[f"link_pose_{k}"] = c

        if (
            self._goal_buffer.goal_state is not None
            and self.cost_cfg.cspace_cfg is not None
            and self.dist_cost.enabled
        ):

            joint_cost = self.dist_cost.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state_batch.position,
                self._goal_buffer.batch_goal_state_idx,
            )
            # cost_list.append(joint_cost)            cost_dict["joint_cost"] = joint_cost
        if self.cost_cfg.straight_line_cfg is not None and self.straight_line_cost.enabled:
            st_cost = self.straight_line_cost.forward(ee_pos_batch)
            # cost_list.append(st_cost)
            cost_dict["straight_line"] = st_cost

        if (
            self.cost_cfg.zero_acc_cfg is not None
            and self.zero_acc_cost.enabled
            # and g_dist is not None
        ):
            z_acc = self.zero_acc_cost.forward(
                state_batch.acceleration,
                g_dist,
            )

            # cost_list.append(z_acc)
            cost_dict["zero_acc"] = z_acc
        if self.cost_cfg.zero_jerk_cfg is not None and self.zero_jerk_cost.enabled:
            z_jerk = self.zero_jerk_cost.forward(
                state_batch.jerk,
                g_dist,
            )
            # cost_list.append(z_jerk)
            cost_dict["zero_jerk"] = z_jerk

        if self.cost_cfg.zero_vel_cfg is not None and self.zero_vel_cost.enabled:
            z_vel = self.zero_vel_cost.forward(
                state_batch.velocity,
                g_dist,
            )
            # cost_list.append(z_vel)
            cost_dict["zero_vel"] = z_vel
        
        # Execute custom arm_reacher costs
        if hasattr(self, '_custom_arm_reacher_costs'):
            for cost_name, cost_instance in self._custom_arm_reacher_costs.items():
                if cost_instance.enabled:
                    with profiler.record_function(f"cost/custom_arm_reacher/{cost_name}"):
                        custom_cost = cost_instance.forward(state)
                        # cost_list.append(custom_cost)
                        cost_dict[cost_name] = custom_cost
                        # try:
                        #     custom_cost = cost_instance.forward(state)
                        #     cost_list.append(custom_cost)
                        # except Exception as e:
                        #     log_error(f"Error computing custom arm_reacher cost {cost_name}: {e}")
        
        # Add live plotting support for ArmReacher - plot all costs in one comprehensive view


        # spectral_concentration_score = 0.0
        # entropy_score = 0.0
        # # if not hasattr(self, 'last_total_costs'):
        # #     self.last_total_costs = [cat_sum_reacher(list(cost_dict.values()))]
        # # else:
        #     # self.last_total_costs.append(cat_sum_reacher(list(cost_dict.values())))
        #     # if len(self.last_total_costs) > 100:
        #     #     self.last_total_costs.pop(0)
        
        #     #     last_total_cost = cat_sum_reacher(self.last_total_costs)
        #     #     X = np.fft.fft(last_total_cost.cpu().numpy())
        #     #     power_spectrum = np.abs(X)**2
        #     #     power_spectrum[0] = 0 # remove DC component
        #     #     power_spectrum /= np.sum(power_spectrum) # normalize
        #     #     spectral_concentration_score = np.max(power_spectrum)

        #     #     entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-12))  # Avoid log(0)
        #     #     max_entropy = np.log2(len(power_spectrum))
        #     #     entropy_score = 1 - (entropy / max_entropy)
                

        if getattr(self, '_enable_live_plotting', False):
            dict_to_plot =  {'total': cat_sum_reacher(list(cost_dict.values()))}
            # dict_to_plot['spectral_concentration_score'] = torch.tensor(spectral_concentration_score)
            # dict_to_plot['entropy_score'] = torch.tensor(entropy_score)
            
            if modified_dyn_obs_cost:
                dict_to_plot['debug_dynamic_obs_cost_before(debug)'] = old_dyn_obs_cost
                # dict_to_plot['dynamic obs diff'] = new_dyn_obs_cost - old_dyn_obs_cost
            for k, v in cost_dict.items():
                if k not in dict_to_plot:
                    dict_to_plot[k] = v
            
            self._update_live_plot(dict_to_plot)

        
        cost_list = list(cost_dict.values())
        with profiler.record_function("cat_sum"):
            if self.sum_horizon:
                cost = cat_sum_horizon_reacher(cost_list)
            else:
                cost = cat_sum_reacher(cost_list)

        return cost

    def convergence_fn(
        self, state: KinematicModelState, out_metrics: Optional[ArmReacherMetrics] = None
    ) -> ArmReacherMetrics:
        if out_metrics is None:
            out_metrics = ArmReacherMetrics()
        if not isinstance(out_metrics, ArmReacherMetrics):
            out_metrics = ArmReacherMetrics(**vars(out_metrics))
        base_metrics = super(ArmReacher, self).convergence_fn(state, out_metrics)
        
        # Copy base metrics to our typed metrics
        if base_metrics != out_metrics:
            out_metrics.cost = base_metrics.cost
            out_metrics.constraint = base_metrics.constraint
            out_metrics.feasible = base_metrics.feasible
            out_metrics.state = base_metrics.state

        # compute error with pose?
        if (
            self._goal_buffer.goal_pose.position is not None
            and self.convergence_cfg.pose_cfg is not None
        ):
            (
                out_metrics.pose_error,
                out_metrics.rotation_error,
                out_metrics.position_error,
            ) = self.pose_convergence.forward_out_distance(
                state.ee_pos_seq, state.ee_quat_seq, self._goal_buffer
            )
            out_metrics.goalset_index = self.pose_convergence.goalset_index_buffer  # .clone()
        if (
            self._goal_buffer.links_goal_pose is not None
            and self.convergence_cfg.pose_cfg is not None
        ):
            pose_error = [out_metrics.pose_error] if out_metrics.pose_error is not None else []
            position_error = [out_metrics.position_error] if out_metrics.position_error is not None else []
            quat_error = [out_metrics.rotation_error] if out_metrics.rotation_error is not None else []
            link_poses = state.link_pose

            for k in self._goal_buffer.links_goal_pose.keys():
                if k != self.kinematics.ee_link:
                    current_fn = self._link_pose_convergence[k]
                    if current_fn.enabled:
                        # get link pose
                        current_pos = link_poses[k].position.contiguous()
                        current_quat = link_poses[k].quaternion.contiguous()

                        pose_err, pos_err, quat_err = current_fn.forward_out_distance(
                            current_pos, current_quat, self._goal_buffer, k
                        )
                        pose_error.append(pose_err)
                        position_error.append(pos_err)
                        quat_error.append(quat_err)
            if pose_error:
                out_metrics.pose_error = cat_max(pose_error)
                out_metrics.rotation_error = cat_max(quat_error)
                out_metrics.position_error = cat_max(position_error)

        if (
            self._goal_buffer.goal_state is not None
            and self.convergence_cfg.cspace_cfg is not None
            and self.cspace_convergence.enabled
        ):
            _, out_metrics.cspace_error = self.cspace_convergence.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state.state_seq.position,
                self._goal_buffer.batch_goal_state_idx,
                True,
            )

        if (
            self.convergence_cfg.null_space_cfg is not None
            and self.null_convergence.enabled
            and self._goal_buffer.batch_retract_state_idx is not None
        ):
            out_metrics.null_space_error = self.null_convergence.forward_target_idx(
                self._goal_buffer.retract_state,
                state.state_seq.position,
                self._goal_buffer.batch_retract_state_idx,
            )

        return out_metrics

    def update_params(
        self,
        goal: Goal,
    ):
        """
        Update params for the cost terms and dynamics model.

        """

        super(ArmReacher, self).update_params(goal)
        if goal.batch_pose_idx is not None:
            self._goal_idx_update = False
        if goal.goal_pose.position is not None:
            self.enable_cspace_cost(False)
        return True

    def enable_pose_cost(self, enable: bool = True):
        if enable:
            self.goal_cost.enable_cost()
        else:
            self.goal_cost.disable_cost()

    def enable_cspace_cost(self, enable: bool = True):
        if enable:
            self.dist_cost.enable_cost()
            self.cspace_convergence.enable_cost()
        else:
            self.dist_cost.disable_cost()
            self.cspace_convergence.disable_cost()

    def _format_multi_arm_ee_data(self, state: KinematicModelState, num_arms: int):
        """Format end-effector data for multi-arm pose cost.
        
        Constructs 4D tensors [batch, horizon, num_arms, 3/4] from available link poses.
        Uses configurable arm link mapping from robot config or sensible defaults.
        
        Args:
            state: Kinematic model state containing link poses
            num_arms: Number of arms expected by the multi-arm pose cost
            
        Returns:
            Tuple of (ee_pos_4d, ee_quat_4d) tensors in 4D format
        """
        # Get batch and horizon dimensions from state
        batch_size = state.ee_pos_seq.shape[0] if state.ee_pos_seq is not None else 1
        horizon = state.ee_pos_seq.shape[1] if state.ee_pos_seq is not None else 1
        
        # Initialize 4D tensors for multi-arm end-effector data
        device = self.tensor_args.device
        dtype = self.tensor_args.dtype
        
        ee_pos_4d = torch.zeros((batch_size, horizon, num_arms, 3), device=device, dtype=dtype)
        ee_quat_4d = torch.zeros((batch_size, horizon, num_arms, 4), device=device, dtype=dtype)
        
        # Default quaternion (identity: w=1, x=0, y=0, z=0)
        ee_quat_4d[:, :, :, 0] = 1.0  # Set w component to 1
        
        # Get arm link mapping from robot configuration or use defaults
        arm_link_mapping = self._get_arm_link_mapping(num_arms)
        
        
        
        # Fill in data for available links
        link_poses = state.link_pose
        if link_poses is not None:
            for arm_idx, link_name in enumerate(arm_link_mapping):
                if link_name in link_poses:
                    # Extract pose data for this arm
                    link_pose = link_poses[link_name]
                    ee_pos_4d[:, :, arm_idx, :] = link_pose.position
                    ee_quat_4d[:, :, arm_idx, :] = link_pose.quaternion
                else:
                    # Link not found - try alternative approaches
                    if arm_idx == 0 and state.ee_pos_seq is not None and state.ee_quat_seq is not None:
                        # Use primary end-effector data for first arm
                        ee_pos_4d[:, :, arm_idx, :] = state.ee_pos_seq
                        ee_quat_4d[:, :, arm_idx, :] = state.ee_quat_seq
                    elif hasattr(self, 'kinematics') and hasattr(self.kinematics, 'get_state'):
                        # Try to compute the missing end-effector pose using kinematics
                        try:
                            # Get current joint positions
                            current_joints = state.state_seq.position
                            if current_joints is not None:
                                # Use kinematics to compute pose for the missing link
                                kin_state = self.kinematics.get_state(current_joints)
                                if hasattr(kin_state, 'link_pose') and link_name in kin_state.link_pose:
                                    computed_pose = kin_state.link_pose[link_name]
                                    ee_pos_4d[:, :, arm_idx, :] = computed_pose.position
                                    ee_quat_4d[:, :, arm_idx, :] = computed_pose.quaternion
                        except Exception as e:
                            pass
                    else:
                        pass
                    # If all fallbacks fail, leave as zeros/identity quaternion (already initialized)
        

        
        return ee_pos_4d, ee_quat_4d

    def _get_arm_link_mapping(self, num_arms: int) -> List[str]:
        """Get arm end-effector link mapping from robot config or generate defaults.
        
        Args:
            num_arms: Number of arms in the system
            
        Returns:
            List of end-effector link names for each arm
        """
        # PRIORITY 1: Try to get link names from robot configuration
        if hasattr(self, 'kinematics') and hasattr(self.kinematics, 'link_names'):
            link_names = self.kinematics.link_names
            if isinstance(link_names, list) and len(link_names) >= num_arms:
                # Use the first num_arms link names from the config
                # Only print once to avoid spam
                if not hasattr(self, '_link_mapping_printed'):
                    print(f"Using robot config link_names: {link_names[:num_arms]}")
                    self._link_mapping_printed = True
                return link_names[:num_arms]
            else:
                if not hasattr(self, '_link_mapping_printed'):
                    print(f"Robot config has {len(link_names) if link_names else 0} link_names, need {num_arms}")
                    self._link_mapping_printed = True
        
        # PRIORITY 2: Backward compatibility fallbacks (only if config unavailable)
        if num_arms == 2:
            # Dual-arm setup (existing behavior)
            return ['left_panda_hand', 'right_panda_hand']
        elif num_arms == 3:
            # Triple-arm setup - updated to match common auto-generated naming
            return ['arm_0_panda_hand', 'arm_1_panda_hand', 'arm_2_panda_hand']
        elif num_arms == 4:
            # Quad-arm setup
            return ['arm_0_panda_hand', 'arm_1_panda_hand', 'arm_2_panda_hand', 'arm_3_panda_hand']
        else:
            # Generic multi-arm setup - generate default names
            return [f'arm_{i}_panda_hand' for i in range(num_arms)]

    def get_pose_costs(
        self,
        include_link_pose: bool = False,
        include_convergence: bool = True,
        only_convergence: bool = False,
    ):
        if only_convergence:
            return [self.pose_convergence]
        pose_costs = [self.goal_cost]
        if include_convergence:
            pose_costs += [self.pose_convergence]
        if include_link_pose:
            log_error("Not implemented yet")
        return pose_costs

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
    ):
        pose_costs = self.get_pose_costs(
            include_link_pose=metric.include_link_pose, include_convergence=False
        )
        for p in pose_costs:
            p.update_metric(metric, update_offset_waypoint=True)

        pose_costs = self.get_pose_costs(only_convergence=True)
        for p in pose_costs:
            p.update_metric(metric, update_offset_waypoint=False)

    def set_dynamic_obs_coll_predictor(self, predictor: DynamicObsCollPredictor):
        self._dynamic_obs_coll_predictor = predictor
    
    def get_dynamic_obs_coll_predictor(self) -> Optional[DynamicObsCollPredictor]:
        return self._dynamic_obs_coll_predictor

    def _update_live_plot(self, cost_dict:dict[str, torch.Tensor]):
        """Update live plot of cost values in real-time"""
        
        
        # Initialize plotting components if not already done
        if not hasattr(self, '_plot_initialized'):
            self._plot_initialized = True
            self._cost_histories = {}  # Dictionary to store history for each cost component
            self._cost_lines = {}  # Dictionary to store plot lines for each cost component
            self._plot_counter = 0  # Counter for plotting frequency
            self._plot_every_k = 5  # Plot every 5 iterations to save resources
            
            # Set up the figure and axis
            plt.ion()  # Turn on interactive mode
            self._fig, self._ax = plt.subplots(1, 1, figsize=(16, 10))
            self._fig.suptitle('Real-time Cost Monitoring - ArmReacher All Components')
            
            self._ax.set_title('All Cost Components Over Time (Base + Reacher + Custom)')
            self._ax.set_xlabel('Iteration')
            self._ax.set_ylabel('Cost Value')
            self._ax.grid(True)
            
            plt.tight_layout()
            # Manually adjust subplot parameters to avoid stretched plot
            plt.subplots_adjust(right=0.84)
            plt.show(block=False)
        
        # Increment counter and check if we should plot this iteration
        self._plot_counter += 1
        if self._plot_counter % self._plot_every_k != 0:
            return  # Skip this iteration
        
    
        # Colors for plotting
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'darkred', 'darkgreen', 'darkblue']
        
        # Process each cost component
        active_costs = []
        for i,label in enumerate(cost_dict.keys()):        
                # Calculate mean of this cost component
                cost_mean = torch.mean(cost_dict[label]).cpu().numpy().item()
                
                # Track all costs, even very small ones, but highlight significant ones
                active_costs.append((label, cost_mean, i))
                    
                # Initialize history for this component if not exists
                if label not in self._cost_histories:
                    self._cost_histories[label] = deque(maxlen=200)  # Keep last 200 plot points
                    color = colors[i % len(colors)]
                    
                    # Special styling for different cost types
                    if 'Goal' in label or 'Pose' in label:
                        linewidth = 3
                        marker = 'o'
                        markersize = 4
                    elif 'Custom' in label:
                        linewidth = 2.5
                        marker = 's'  # Square markers for custom costs
                        markersize = 4
                    else:
                        linewidth = 2
                        marker = 'o'
                        markersize = 3
                    
                    self._cost_lines[label], = self._ax.plot([], [], color=color, label=label, 
                                                           linewidth=linewidth, marker=marker, markersize=markersize)
                
                # Add current value to history
                self._cost_histories[label].append(cost_mean)
        
      
        # Update all plot lines
        for label, history in self._cost_histories.items():
            if len(history) > 0:
                x_data = list(range(len(history)))
                y_data = list(history)
                self._cost_lines[label].set_data(x_data, y_data)
        
        # Update plot limits and legend
        if self._cost_histories:
            # Get all x and y data for proper scaling
            all_x_data = []
            all_y_data = []
            for history in self._cost_histories.values():
                if len(history) > 0:
                    all_x_data.extend(range(len(history)))
                    all_y_data.extend(history)
            
            if all_x_data and all_y_data:
                self._ax.set_xlim(0, max(all_x_data) + 1)
                y_min, y_max = min(all_y_data), max(all_y_data)
                y_range = y_max - y_min
                if y_range > 0:
                    self._ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                else:
                    self._ax.set_ylim(y_min - 0.1, y_max + 0.1)
        
        # Update legend (only when new components are added)
        if not hasattr(self, '_legend_updated') or len(self._cost_lines) != getattr(self, '_last_legend_count', 0):
            self._ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
            self._legend_updated = True
            self._last_legend_count = len(self._cost_lines)
        
        # Refresh the plot
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        
        # Optional: Save periodic snapshots (less frequent)
        if self._save_plots and self._plot_counter % (self._plot_every_k * 20) == 0:  # Every 100 actual iterations
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if not hasattr(self, '_cost_plots_dir'):
                self._cost_plots_dir = os.path.join(os.getcwd(),'tmp_artifacts', 'cost_plots', timestamp)
                os.makedirs(self._cost_plots_dir, exist_ok=False)
            self._fig.savefig(os.path.join(self._cost_plots_dir, f'costs_iter_{self._plot_counter}.png'), dpi=150, bbox_inches='tight')
            print(f"Saved plot snapshot at iteration {self._plot_counter}")

    def set_plot_frequency(self, k: int):
        """Set how often to update the live plot (every k iterations)
        
        Args:
            k (int): Update plot every k iterations
        """
        if hasattr(self, '_plot_every_k'):
            self._plot_every_k = k
            print(f"Plot frequency set to every {k} iterations")
        else:
            print("Live plotting not initialized yet. This will take effect when plotting starts.")


@get_torch_jit_decorator()
def cat_sum_reacher(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=0)
    return cat_tensor


@get_torch_jit_decorator()
def cat_sum_horizon_reacher(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=(0, -1))
    return cat_tensor
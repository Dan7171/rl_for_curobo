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
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import torch

# CuRobo
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import log_error, log_warn

# Local Folder
from .cost_base import CostBase, CostConfig
from .pose_cost import PoseCost, PoseCostConfig, PoseCostMetric, PoseErrorType


@dataclass
class PoseCostMultiArmConfig(PoseCostConfig):
    """Configuration for multi-arm pose cost.
    
    Expects 4D tensor format with separate dimension for each arm:
    - ee_pos_batch: [batch, horizon, num_arms, 3] 
    - ee_quat_batch: [batch, horizon, num_arms, 4]
    """
    num_arms: int = 1  # Number of arms in the multi-arm system
    
    def __post_init__(self):
        if self.num_arms < 1:
            log_error("num_arms must be at least 1")
        return super().__post_init__()


class PoseCostMultiArm(CostBase, PoseCostMultiArmConfig):
    """Multi-arm pose cost using separate PoseCost instances for each arm.
    
    This class creates individual PoseCost objects for each arm and computes
    pose costs separately, then averages them. Each arm uses the original
    pose_cost.py implementation without any modifications.
    
    Input format:
    - ee_pos_batch: [batch, horizon, num_arms, 3] - positions for each arm
    - ee_quat_batch: [batch, horizon, num_arms, 4] - quaternions for each arm
    - goal: Multi-arm goal with targets for each arm
    """
    
    def __init__(self, config: PoseCostMultiArmConfig):
        PoseCostMultiArmConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        
        # Create individual pose cost instances for each arm
        # Each arm gets its own PoseCost object using the original implementation
        self._arm_pose_costs = []
        for i in range(self.num_arms):
            # Handle offset_waypoint properly to avoid tensor boolean ambiguity
            offset_waypoint_val = getattr(self, 'offset_waypoint', None)
            if offset_waypoint_val is None:
                offset_waypoint_val = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            # Create a config for each arm (same parameters as the multi-arm config)
            arm_config = PoseCostConfig(
                cost_type=self.cost_type,
                use_metric=self.use_metric,
                project_distance=self.project_distance,
                run_vec_weight=getattr(self, 'run_vec_weight', None),
                use_projected_distance=self.use_projected_distance,
                offset_waypoint=offset_waypoint_val,
                offset_tstep_fraction=self.offset_tstep_fraction,
                waypoint_horizon=self.waypoint_horizon,
                weight=self.weight,
                vec_weight=self.vec_weight,
                vec_convergence=self.vec_convergence,
                run_weight=self.run_weight,
                return_loss=self.return_loss,
                terminal=self.terminal,
                tensor_args=self.tensor_args
            )
            
            # Create individual pose cost for this arm using original PoseCost
            arm_pose_cost = PoseCost(arm_config)
            self._arm_pose_costs.append(arm_pose_cost)
        
        self._batch_size = 0
        self._horizon = 0
    
    def update_metric(self, metric: PoseCostMetric, update_offset_waypoint: bool = True):
        """Update metrics for all arms."""
        for arm_pose_cost in self._arm_pose_costs:
            arm_pose_cost.update_metric(metric, update_offset_waypoint)
    
    def hold_partial_pose(self, run_vec_weight: torch.Tensor):
        """Hold partial pose for all arms."""
        for arm_pose_cost in self._arm_pose_costs:
            arm_pose_cost.hold_partial_pose(run_vec_weight)
    
    def release_partial_pose(self):
        """Release partial pose for all arms."""
        for arm_pose_cost in self._arm_pose_costs:
            arm_pose_cost.release_partial_pose()
    
    def reach_partial_pose(self, vec_weight: torch.Tensor):
        """Set reach partial pose for all arms."""
        for arm_pose_cost in self._arm_pose_costs:
            arm_pose_cost.reach_partial_pose(vec_weight)
    
    def reach_full_pose(self):
        """Set reach full pose for all arms."""
        for arm_pose_cost in self._arm_pose_costs:
            arm_pose_cost.reach_full_pose()
    
    def update_batch_size(self, batch_size, horizon):
        """Update batch size for all arms."""
        if batch_size != self._batch_size or horizon != self._horizon:
            for arm_pose_cost in self._arm_pose_costs:
                arm_pose_cost.update_batch_size(batch_size, horizon)
            self._batch_size = batch_size
            self._horizon = horizon
    
    def _extract_arm_ee_data(self, ee_pos_batch: torch.Tensor, ee_quat_batch: torch.Tensor, arm_idx: int):
        """Extract end-effector data for a specific arm from 4D tensors.
        
        Args:
            ee_pos_batch: [batch, horizon, num_arms, 3]
            ee_quat_batch: [batch, horizon, num_arms, 4]
            arm_idx: Index of the arm (0-based)
            
        Returns:
            Tuple of (arm_ee_pos, arm_ee_quat) in format expected by original PoseCost:
            - arm_ee_pos: [batch, horizon, 3]
            - arm_ee_quat: [batch, horizon, 4]
        """
        # Extract data for this arm - this is the format original PoseCost expects
        arm_ee_pos = ee_pos_batch[:, :, arm_idx, :]  # [batch, horizon, 3]
        arm_ee_quat = ee_quat_batch[:, :, arm_idx, :]  # [batch, horizon, 4]
        
        return arm_ee_pos, arm_ee_quat
    
    def _create_arm_goal(self, goal: Goal, arm_idx: int) -> Goal:
        """Create a goal for a specific arm from the multi-arm goal.
        
        This extracts the target pose for the specified arm and creates a new Goal
        object that can be used with the original PoseCost implementation.
        
        Args:
            goal: Multi-arm goal containing targets for all arms
            arm_idx: Index of the arm to extract goal for (0-based)
            
        Returns:
            Goal object for the specific arm in format expected by original PoseCost
        """
        # Create a new goal for this specific arm
        arm_goal = Goal()
        
        # Copy state information (same for all arms)
        arm_goal.current_state = goal.current_state
        arm_goal.goal_state = goal.goal_state
        arm_goal.batch_pose_idx = goal.batch_pose_idx
        arm_goal.batch_goal_state_idx = goal.batch_goal_state_idx
        arm_goal.batch_retract_state_idx = goal.batch_retract_state_idx
        arm_goal.retract_state = goal.retract_state
        arm_goal.links_goal_pose = goal.links_goal_pose
        
        # Extract the target pose for this specific arm
        if goal.goal_pose is not None and goal.goal_pose.position is not None:
            goal_pos_shape = goal.goal_pose.position.shape
            
            # Debug: print goal extraction occasionally (reduced output)
            if not hasattr(self, '_goal_debug_counter'):
                self._goal_debug_counter = 0
            self._goal_debug_counter += 1
            
            if len(goal_pos_shape) == 2 and goal_pos_shape[0] >= self.num_arms:
                # Multi-arm format: [num_arms, 3]
                arm_goal_pos = goal.goal_pose.position[arm_idx:arm_idx+1, :]  # [1, 3]
                if goal.goal_pose.quaternion is not None:
                    arm_goal_quat = goal.goal_pose.quaternion[arm_idx:arm_idx+1, :]  # [1, 4]
                else:
                    # Default identity quaternion
                    device = self.tensor_args.device
                    dtype = self.tensor_args.dtype
                    arm_goal_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)
                    
            elif len(goal_pos_shape) == 3 and goal_pos_shape[1] >= self.num_arms:
                # Batch multi-arm format: [batch, num_arms, 3]
                arm_goal_pos = goal.goal_pose.position[:, arm_idx:arm_idx+1, :]  # [batch, 1, 3]
                if goal.goal_pose.quaternion is not None:
                    arm_goal_quat = goal.goal_pose.quaternion[:, arm_idx:arm_idx+1, :]  # [batch, 1, 4]
                else:
                    # Default identity quaternion
                    device = self.tensor_args.device
                    dtype = self.tensor_args.dtype
                    batch_size = goal.goal_pose.position.shape[0]
                    arm_goal_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 4)
                    
            else:
                # Single goal for all arms or fallback
                arm_goal_pos = goal.goal_pose.position
                arm_goal_quat = goal.goal_pose.quaternion if goal.goal_pose.quaternion is not None else torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            
            # Debug: removed excessive output
            
            # Create pose for this arm in format expected by original PoseCost
            arm_goal.goal_pose = Pose(position=arm_goal_pos, quaternion=arm_goal_quat)
        else:
            # No goal pose available
            arm_goal.goal_pose = goal.goal_pose
        
        return arm_goal
    
    def forward_out_distance(
        self, ee_pos_batch, ee_quat_batch, goal: Goal, link_name: Optional[str] = None
    ):
        """Compute multi-arm pose cost with detailed distance outputs.
        
        This method uses the original PoseCost.forward_out_distance() for each arm separately.
        
        Args:
            ee_pos_batch: [batch, horizon, num_arms, 3] - positions for each arm
            ee_quat_batch: [batch, horizon, num_arms, 4] - quaternions for each arm
            goal: Multi-arm goal containing targets for all arms
            link_name: Optional link name (ignored in multi-arm setup)
            
        Returns:
            Tuple of (cost, rotation_error, position_distance) averaged across arms
        """
        if ee_pos_batch is None or ee_quat_batch is None:
            log_error("ee_pos_batch and ee_quat_batch must not be None for multi-arm pose cost")
            return torch.zeros((1, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype), \
                   torch.zeros((1, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype), \
                   torch.zeros((1, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        
        # Validate input shapes
        b, h = ee_pos_batch.shape[:2]
        
        if len(ee_pos_batch.shape) != 4 or ee_pos_batch.shape[2] != self.num_arms or ee_pos_batch.shape[3] != 3:
            log_error(f"Expected ee_pos_batch shape [batch, horizon, {self.num_arms}, 3], got {ee_pos_batch.shape}")
            return torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype), \
                   torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype), \
                   torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
                   
        if len(ee_quat_batch.shape) != 4 or ee_quat_batch.shape[2] != self.num_arms or ee_quat_batch.shape[3] != 4:
            log_error(f"Expected ee_quat_batch shape [batch, horizon, {self.num_arms}, 4], got {ee_quat_batch.shape}")
            return torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype), \
                   torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype), \
                   torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        
        self.update_batch_size(b, h)
        
        # Compute cost for each arm using original PoseCost and sum them
        total_cost = None
        total_rot_err = None 
        total_pos_dist = None
        
        for arm_idx in range(self.num_arms):
            # Extract this arm's end-effector pose in format expected by original PoseCost
            arm_ee_pos, arm_ee_quat = self._extract_arm_ee_data(ee_pos_batch, ee_quat_batch, arm_idx)
            
            # Create this arm's goal in format expected by original PoseCost
            arm_goal = self._create_arm_goal(goal, arm_idx)
            
            # Use original PoseCost.forward_out_distance() for this arm
            arm_cost, arm_rot_err, arm_pos_dist = self._arm_pose_costs[arm_idx].forward_out_distance(
                arm_ee_pos, arm_ee_quat, arm_goal, link_name=None
            )
            
            # Sum costs (we'll average at the end)
            if total_cost is None:
                total_cost = arm_cost
                total_rot_err = arm_rot_err
                total_pos_dist = arm_pos_dist
            else:
                total_cost = total_cost + arm_cost
                total_rot_err = total_rot_err + arm_rot_err
                total_pos_dist = total_pos_dist + arm_pos_dist
        
        # Average by number of arms
        if total_cost is not None:
            avg_cost = total_cost / self.num_arms
            avg_rot_err = total_rot_err / self.num_arms if total_rot_err is not None else total_rot_err
            avg_pos_dist = total_pos_dist / self.num_arms if total_pos_dist is not None else total_pos_dist
        else:
            # Fallback (should not happen)
            avg_cost = torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            avg_rot_err = torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
            avg_pos_dist = torch.zeros((b, h), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        
        return avg_cost, avg_rot_err, avg_pos_dist
    
    def forward(self, ee_pos_batch, ee_quat_batch, goal_buffer):
        """Forward pass for multi-arm pose cost using original PoseCost for each arm.
        
        This method uses the original PoseCost.forward() for each arm separately.
        
        Args:
            ee_pos_batch: [batch, horizon, num_arms, 3] - positions for each arm
            ee_quat_batch: [batch, horizon, num_arms, 4] - quaternions for each arm
            goal_buffer: Multi-arm goal containing targets for all arms
            
        Returns:
            Total cost averaged across all arms
        """
        # Validate tensor shapes
        if ee_pos_batch.dim() != 4 or ee_quat_batch.dim() != 4:
            raise ValueError(f"Expected 4D tensors [batch, horizon, num_arms, 3/4], got pos: {ee_pos_batch.shape}, quat: {ee_quat_batch.shape}")
        
        batch_size, horizon, num_arms_data, pos_dim = ee_pos_batch.shape
        if num_arms_data != self.num_arms or pos_dim != 3:
            raise ValueError(f"Position tensor shape mismatch. Expected [..., {self.num_arms}, 3], got {ee_pos_batch.shape}")
        
        if ee_quat_batch.shape != (batch_size, horizon, self.num_arms, 4):
            raise ValueError(f"Quaternion tensor shape mismatch. Expected [..., {self.num_arms}, 4], got {ee_quat_batch.shape}")
        
        total_cost = torch.zeros((batch_size, horizon), device=ee_pos_batch.device, dtype=ee_pos_batch.dtype)
        
        # Compute cost for each arm using original PoseCost
        arm_costs = []
        for arm_idx in range(self.num_arms):
            # Extract pose data for this arm in format expected by original PoseCost
            arm_ee_pos, arm_ee_quat = self._extract_arm_ee_data(ee_pos_batch, ee_quat_batch, arm_idx)
            
            # Create goal for this arm in format expected by original PoseCost
            arm_goal = self._create_arm_goal(goal_buffer, arm_idx)
            
            # Use original PoseCost.forward() for this arm
            arm_cost = self._arm_pose_costs[arm_idx].forward(arm_ee_pos, arm_ee_quat, arm_goal)
            arm_costs.append(arm_cost)
            
            # Add to total cost
            total_cost += arm_cost
        
        # Average by number of arms
        averaged_cost = total_cost / self.num_arms
        
        # Debug cost values occasionally (reduced frequency)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        
        if self._debug_counter % 1000 == 0:  # Reduced from every 100 to every 1000 calls
            print(f"Multi-arm Cost Debug (call #{self._debug_counter}): Average cost = {averaged_cost.mean().item():.2f}")
        
        return averaged_cost
    
    @property
    def goalset_index_buffer(self):
        """Return goalset index buffer from the first arm (all arms should have the same)."""
        return self._arm_pose_costs[0].goalset_index_buffer 
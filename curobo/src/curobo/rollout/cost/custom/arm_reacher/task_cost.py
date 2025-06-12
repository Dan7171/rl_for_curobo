"""
Example task-specific cost for arm_reacher.
Users can copy this file and modify it for their own task needs.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState


@dataclass
class TaskCostConfig(CostConfig):
    """Configuration for task-specific cost."""
    
    target_position: Optional[torch.Tensor] = None
    position_weight: float = 1.0
    orientation_weight: float = 0.5
    
    def __post_init__(self):
        if self.target_position is not None:
            self.target_position = self.tensor_args.to_device(self.target_position)
        return super().__post_init__()


class TaskCost(CostBase, TaskCostConfig):
    """
    Task-specific cost that penalizes deviation from a target pose.
    This is an arm_reacher cost - specific to the manipulation task.
    """
    
    def __init__(self, config: TaskCostConfig):
        TaskCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
    
    def forward(self, state: KinematicModelState, **kwargs) -> torch.Tensor:
        """
        Compute task cost based on end-effector pose deviation from target.
        
        Args:
            state: Robot kinematic state containing pose information
            **kwargs: Additional arguments (for compatibility with arm_reacher)
            
        Returns:
            Cost tensor of shape [batch, horizon]
        """
        # Get end-effector positions and orientations
        ee_pos = state.ee_pos_seq  # [batch, horizon, 3]
        ee_quat = state.ee_quat_seq  # [batch, horizon, 4]
        
        # Check if data is available
        if ee_pos is None or not isinstance(ee_pos, torch.Tensor):
            # No end-effector data available, return zero cost
            state_seq = state.state_seq
            position = state_seq.position
            if not isinstance(position, torch.Tensor):
                raise ValueError("Expected position to be a torch.Tensor")
            batch_size, horizon = position.shape[:2]
            return torch.zeros(
                (batch_size, horizon),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            )
        
        batch_size, horizon, _ = ee_pos.shape
        
        # Initialize cost
        cost = torch.zeros(
            (batch_size, horizon),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        )
        
        if self.target_position is not None:
            # Position cost - distance to target
            target_pos = self.target_position.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
            position_error = torch.norm(ee_pos - target_pos, dim=-1)  # [batch, horizon]
            cost += self.position_weight * position_error
        
        # Optional: Add orientation cost here if needed
        # This is a simplified example - you might want to add quaternion error
        
        return self._weight * cost 
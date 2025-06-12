"""
Example energy minimization cost for arm_base.
Users can copy this file and modify it for their own needs.
"""

from dataclasses import dataclass
from typing import Optional
import torch
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState


@dataclass
class EnergyCostConfig(CostConfig):
    """Configuration for energy minimization cost."""
    
    energy_scale: float = 1.0
    velocity_weight: float = 1.0
    acceleration_weight: float = 0.1
    
    def __post_init__(self):
        return super().__post_init__()


class EnergyCost(CostBase, EnergyCostConfig):
    """
    Energy minimization cost that penalizes high velocities and accelerations.
    This is an arm_base cost - general robot behavior, not task-specific.
    """
    
    def __init__(self, config: EnergyCostConfig):
        EnergyCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
    
    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """
        Compute energy cost based on velocities and accelerations.
        
        Args:
            state: Robot kinematic state containing joint information
            
        Returns:
            Cost tensor of shape [batch, horizon]
        """
        # Get velocities and accelerations
        velocity = state.state_seq.velocity  # [batch, horizon, dof]
        acceleration = state.state_seq.acceleration  # [batch, horizon, dof]
        
        # Check if tensors are valid
        if not isinstance(velocity, torch.Tensor):
            raise ValueError("Expected velocity to be a torch.Tensor")
        if not isinstance(acceleration, torch.Tensor):
            raise ValueError("Expected acceleration to be a torch.Tensor")
            
        batch_size, horizon, n_dofs = velocity.shape
        
        # Compute velocity energy
        vel_energy = torch.sum(velocity ** 2, dim=-1)  # [batch, horizon]
        
        # Compute acceleration energy  
        acc_energy = torch.sum(acceleration ** 2, dim=-1)  # [batch, horizon]
        
        # Combine with weights
        total_energy = (self.velocity_weight * vel_energy + 
                       self.acceleration_weight * acc_energy)
        
        # Apply cost scaling and weight
        cost = self.energy_scale * total_energy
        
        return self._weight * cost 
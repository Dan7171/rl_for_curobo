"""
Example Custom Cost Terms

This file provides working examples of custom cost terms that can be used
as reference implementations.
"""

# Standard Library
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState


@dataclass
class RandomCostConfig(CostConfig):
    """Configuration for a random cost term (for testing/demonstration)."""
    
    random_scale: float = 1.0
    use_terminal_only: bool = False
    
    def __post_init__(self):
        return super().__post_init__()


class RandomCost(CostBase, RandomCostConfig):
    """Example random cost term.
    
    This cost term returns random values between 0 and random_scale.
    Useful for testing the integration of custom cost terms.
    """
    
    def __init__(self, config: RandomCostConfig):
        RandomCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        
    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """Generate random costs.
        
        Args:
            state: KinematicModelState containing robot state information
            
        Returns:
            torch.Tensor: Random cost values [batch_size, horizon]
        """
        # Extract batch and horizon dimensions from state_seq
        state_seq = state.state_seq
        position = state_seq.position
        if not isinstance(position, torch.Tensor):
            raise ValueError("Expected position to be a torch.Tensor")
        batch_size, horizon = position.shape[:2]
        
        # Generate random costs
        cost = torch.rand(
            (batch_size, horizon),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        ) * self.random_scale
        
        # Apply terminal-only if configured
        if self.use_terminal_only:
            terminal_mask = torch.zeros_like(cost)
            terminal_mask[:, -1] = 1.0
            cost = cost * terminal_mask
        return self._weight * cost   


@dataclass 
class VelocitySmoothnessCostConfig(CostConfig):
    """Configuration for velocity smoothness cost."""
    
    smoothness_weight: float = 1.0
    
    def __post_init__(self):
        return super().__post_init__()


class VelocitySmoothnessCost(CostBase, VelocitySmoothnessCostConfig):
    """Example velocity smoothness cost for arm_base.
    
    Penalizes large changes in velocity between timesteps.
    """
    
    def __init__(self, config: VelocitySmoothnessCostConfig):
        VelocitySmoothnessCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
    
    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """Compute velocity smoothness cost.
        
        Args:
            state: KinematicModelState containing robot state information
            
        Returns:
            torch.Tensor: Smoothness costs [batch_size, horizon]
        """
        # Extract velocity from state
        velocity = state.state_seq.velocity  # [batch, horizon, n_dofs]
        if not isinstance(velocity, torch.Tensor):
            raise ValueError("Expected velocity to be a torch.Tensor")
        batch_size, horizon, n_dofs = velocity.shape
        
        if horizon < 2:
            # Need at least 2 timesteps for smoothness
            return torch.zeros(
                (batch_size, horizon),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            )
        
        # Compute velocity differences between consecutive timesteps
        vel_diff = velocity[:, 1:, :] - velocity[:, :-1, :]  # [batch, horizon-1, n_dofs]
        
        # Sum squared differences across DOFs
        smoothness_cost = torch.sum(vel_diff ** 2, dim=-1)  # [batch, horizon-1]
        
        # Pad to match horizon length (no cost for first timestep)
        cost = torch.zeros(
            (batch_size, horizon),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        )
        cost[:, 1:] = smoothness_cost * self.smoothness_weight
        
        return self._weight * cost


@dataclass
class EndEffectorRegionCostConfig(CostConfig):
    """Configuration for end-effector region cost."""
    
    region_center: Optional[torch.Tensor] = None
    region_radius: float = 0.5
    cost_outside_region: bool = True
    
    def __post_init__(self):
        if self.region_center is not None:
            self.region_center = self.tensor_args.to_device(self.region_center)
        return super().__post_init__()


class EndEffectorRegionCost(CostBase, EndEffectorRegionCostConfig):
    """Example end-effector region cost for arm_reacher.
    
    Penalizes end-effector positions outside (or inside) a spherical region.
    This cost is designed for arm_reacher but can also work in arm_base.
    """
    
    def __init__(self, config: EndEffectorRegionCostConfig):
        EndEffectorRegionCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
    
    def forward(self, state: KinematicModelState, **kwargs) -> torch.Tensor:
        """Compute region cost.
        
        Args:
            state: KinematicModelState containing robot state information
            **kwargs: Additional arguments (for compatibility with arm_reacher)
            
        Returns:
            torch.Tensor: Region costs [batch_size, horizon]
        """
        # Extract end-effector positions from state
        ee_pos_batch = state.ee_pos_seq  # [batch, horizon, 3]
        
        if ee_pos_batch is None or not isinstance(ee_pos_batch, torch.Tensor):
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
        
        batch_size, horizon, _ = ee_pos_batch.shape
        
        if self.region_center is None:
            # No region defined, return zero cost
            return torch.zeros(
                (batch_size, horizon),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype
            )
        
        # Compute distance to region center
        center = self.region_center.unsqueeze(0).unsqueeze(0)  # [1, 1, 3]
        distance = torch.norm(ee_pos_batch - center, dim=-1)  # [batch, horizon]
        
        if self.cost_outside_region:
            # Cost for being outside the region
            cost = torch.nn.functional.relu(distance - self.region_radius)
        else:
            # Cost for being inside the region  
            cost = torch.nn.functional.relu(self.region_radius - distance)
        
        return self._weight * cost 
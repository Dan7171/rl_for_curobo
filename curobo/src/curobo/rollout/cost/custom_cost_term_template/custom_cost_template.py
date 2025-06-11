"""
Custom Cost Template

This template shows how to create custom cost terms for CuRobo.
Copy this file and modify it to create your own custom cost terms.
"""

# Standard Library
from dataclasses import dataclass

# Third Party
import torch

# CuRobo
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState


@dataclass
class MyCustomCostConfig(CostConfig):
    """Configuration for your custom cost term.
    
    Add any custom parameters here. All CostConfig parameters are available:
    - weight: Cost weight (float)
    - vec_weight: Per-timestep weights (Optional[torch.Tensor])
    - terminal: Apply only to terminal state (bool)
    - run_weight: Weight for non-terminal timesteps (float)
    - return_loss: Return individual losses instead of sum (bool)
    """
    
    # Add your custom parameters here
    my_param1: float = 1.0
    my_param2: bool = False
    
    def __post_init__(self):
        # Call parent post_init to handle tensor_args setup
        return super().__post_init__()


class MyCustomCost(CostBase, MyCustomCostConfig):
    """Template for a custom cost term.
    
    This template shows the basic structure for implementing custom costs.
    Replace 'MyCustomCost' with your actual cost name and implement the forward method.
    """
    
    def __init__(self, config: MyCustomCostConfig):
        # Initialize configuration
        MyCustomCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        
    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """Compute the cost.
        
        This is the main function where you implement your cost logic.
        
        Args:
            state: KinematicModelState containing robot state information.
                  Key attributes:
                  - state.state_seq.position: Joint positions [batch, horizon, n_dofs]
                  - state.state_seq.velocity: Joint velocities [batch, horizon, n_dofs]
                  - state.state_seq.acceleration: Joint accelerations [batch, horizon, n_dofs]
                  - state.ee_pos_seq: End-effector positions [batch, horizon, 3]
                  - state.ee_quat_seq: End-effector quaternions [batch, horizon, 4]
                  - state.robot_spheres: Robot collision spheres [batch, horizon, n_spheres, 4]
                  - state.link_pos_seq: Link positions [batch, horizon, n_links, 3]
                  - state.link_quat_seq: Link quaternions [batch, horizon, n_links, 4]
                      
        Returns:
            torch.Tensor: Cost values of shape [batch_size, horizon]
                        Values should be non-negative.
        """
        # Extract dimensions from state
        state_seq = state.state_seq
        position = state_seq.position
        if not isinstance(position, torch.Tensor):
            raise ValueError("Expected position to be a torch.Tensor")
        batch_size, horizon = position.shape[:2]
        
        # TODO: Implement your cost computation here
        # This is just a placeholder - replace with your actual cost logic
        
        # Example: Create a simple cost based on joint positions
        cost = torch.zeros(
            (batch_size, horizon),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        )
        
        # Example implementation: penalize large joint positions
        if self.my_param2:
            joint_cost = torch.sum(position ** 2, dim=-1)  # Sum over DOFs
            cost = joint_cost * self.my_param1
        
        # Apply weight (inherited from CostConfig) - ensure it returns a tensor
        if isinstance(self.weight, torch.Tensor):
            cost = self.weight * cost
        else:
            # Convert weight to float if it's not already
            weight_val = float(self.weight) if isinstance(self.weight, (int, float)) else 1.0
            cost = cost * weight_val
        
        return cost


# You can define multiple cost classes in the same file
@dataclass
class AnotherCustomCostConfig(CostConfig):
    """Another example cost configuration."""
    
    scaling_factor: float = 2.0
    
    def __post_init__(self):
        return super().__post_init__()


class AnotherCustomCost(CostBase, AnotherCustomCostConfig):
    """Another example custom cost."""
    
    def __init__(self, config: AnotherCustomCostConfig):
        AnotherCustomCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        
    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """Compute another type of cost."""
        # Extract dimensions from state
        state_seq = state.state_seq
        velocity = state_seq.velocity
        if not isinstance(velocity, torch.Tensor):
            raise ValueError("Expected velocity to be a torch.Tensor")
        batch_size, horizon = velocity.shape[:2]
        
        # Example: Penalize high velocities
        velocity_magnitude = torch.norm(velocity, dim=-1)  # [batch, horizon]
        cost = velocity_magnitude * self.scaling_factor
        
        # Apply weight properly to ensure tensor return type
        if isinstance(self.weight, torch.Tensor):
            cost = self.weight * cost
        else:
            # Convert weight to float if it's not already
            weight_val = float(self.weight) if isinstance(self.weight, (int, float)) else 1.0
            cost = cost * weight_val
            
        return cost 
"""
Example energy minimization cost for arm_base.
Users can copy this file and modify it for their own needs.
"""

from dataclasses import dataclass, field
from typing import Optional
from curobo.rollout import cost
import torch
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState

from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor


@dataclass
class DynamicObsCostConfig(CostConfig):
    """Configuration for energy minimization cost."""
    
    num_particles: int = -1
    horizon: int = -1
    X: tuple[float, float, float, float, float, float, float] = (0, 0, 0, 1, 0, 0, 0) # robot base pose expressed in the world frame: x,y,z qx,qy,qz,qw (default is identity quaternion)
    n_coll_spheres: int = -1 # total number of spheres of "self" (the robot in which the cost is running) 
    n_own_spheres: int = -1 # total number of spheres of "others" (the dynamic obstacles) 
    sparse_steps: dict = field(default_factory=lambda: {'use': False, 'ratio': 0.5})
    sparse_spheres: dict = field(default_factory=lambda: {'exclude_self': [], 'exclude_others': []})
    col_pred: Optional[DynamicObsCollPredictor] = None
    def __post_init__(self):
        return super().__post_init__()

    
class DynamicObsCost(CostBase, DynamicObsCostConfig):
    """
    Energy minimization cost that penalizes high velocities and accelerations.
    This is an arm_base cost - general robot behavior, not task-specific.
    """
    
    def __init__(self, config: DynamicObsCostConfig):
        DynamicObsCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        
        # Extract weight value properly
        weight_value = self.weight
        if isinstance(weight_value, torch.Tensor):
            weight_value = weight_value.cpu().item()
        elif isinstance(weight_value, (list, tuple)):
            weight_value = weight_value[0] if len(weight_value) > 0 else 1.0
        elif not isinstance(weight_value, (int, float)):
            weight_value = 1.0
            
        self.col_pred = DynamicObsCollPredictor(self.tensor_args,
                                                            None,
                                                            self.horizon,
                                                            self.num_particles ,
                                                            self.n_own_spheres,
                                                            self.n_coll_spheres,
                                                            weight_value,
                                                            self.X,
                                                            self.sparse_steps,
                                                            self.sparse_spheres)

    def __post_init__(self):
        return super().__post_init__()

    
    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """
        
        Args:
            state: Robot kinematic state containing joint information
            
        Returns:
            Cost tensor of shape [batch, horizon]
        """
        # Ensure col_pred is not None and robot_spheres is not None
        if self.col_pred is None:
            raise RuntimeError("Collision predictor not initialized")
        if state.robot_spheres is None:
            raise RuntimeError("Robot spheres not available in state")
            
        result = self.col_pred.cost_fn(state.robot_spheres)
        return result

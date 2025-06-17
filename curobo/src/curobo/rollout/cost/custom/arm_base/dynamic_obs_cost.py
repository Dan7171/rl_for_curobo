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
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import get_topics

@dataclass
class DynamicObsCostConfig(CostConfig):
    """Configuration for energy minimization cost."""
    weight: float = 100
    X: tuple[float, float, float, float, float, float, float] = (0, 0, 0, 1, 0, 0, 0) # robot base pose expressed in the world frame: x,y,z qx,qy,qz,qw (default is identity quaternion)
    n_coll_spheres: int = -1 # total number of spheres of "self" (the robot in which the cost is running) 
    sparse_steps: dict = field(default_factory=lambda: {'use': False, 'ratio': 0.5})
    sparse_spheres: dict = field(default_factory=lambda: {'exclude_self': [], 'exclude_others': []})
    col_pred: Optional[DynamicObsCollPredictor] = None
    env_id: int = 0
    robot_id: int = 0
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
        weight_value = weight_value.cpu().item() # cost weight
        
        runtime_topics = get_topics()
        if runtime_topics is None:
            raise RuntimeError("Runtime topics not initialized")
        n_own_spheres = runtime_topics.topics[self.env_id][self.robot_id]['mpc_cfg']['cost']['custom']['n_own_spheres']
        horizon = runtime_topics.topics[self.env_id][self.robot_id]['mpc_cfg']['model']['horizon']
        n_rollouts = runtime_topics.topics[self.env_id][self.robot_id]['mpc_cfg']['mppi']['num_particles']
        
        self.col_pred = DynamicObsCollPredictor(self.tensor_args,
                                                            horizon,
                                                            n_rollouts,
                                                            n_own_spheres,
                                                            self.n_coll_spheres,
                                                            weight_value,
                                                            self.X,
                                                            self.sparse_steps,
                                                            self.sparse_spheres,
                                                            self.env_id,
                                                            self.robot_id
                                                            )

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

"""
Dynamic obstacle collision cost for arm_base.
This cost dynamically computes collision avoidance with other robots.
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import get_topics

@dataclass
class DynamicObsCostConfig(CostConfig):
    """Configuration for dynamic obstacle collision cost."""
    weight: float = 100.0
    # Remove duplicated fields - these will be computed dynamically:
    # X: robot pose (computed from robot instance)
    # n_coll_spheres: total obstacle spheres (computed from col_pred_with)
    # env_id, robot_id: passed from robot context
    
    # Optional sparsity configuration
    sparse_steps: dict = field(default_factory=lambda: {'use': False, 'ratio': 0.5})
    sparse_spheres: dict = field(default_factory=lambda: {'exclude_self': [], 'exclude_others': []})
    
    def __post_init__(self):
        return super().__post_init__()

    
class DynamicObsCost(CostBase, DynamicObsCostConfig):
    """
    Dynamic obstacle collision cost that avoids collisions with other robots.
    This cost automatically computes robot poses and obstacle counts from runtime context.
    """
    
    def __init__(self, config: DynamicObsCostConfig):
        print("DynamicObsCost.__init__() called - initializing dynamic obstacle cost")
        DynamicObsCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        
        # Extract weight value properly (handle torch tensor case)
        weight_value = self.weight
        if isinstance(weight_value, torch.Tensor):
            weight_value = weight_value.cpu().item()
        print(f"DynamicObsCost weight: {weight_value}")
        
        runtime_topics = get_topics()
        if runtime_topics is None:
            print("ERROR: Runtime topics not initialized - disabling dynamic obstacle cost")
            self.col_pred = None
            self.disable_cost()
            return
        
        # Find robot context using simple assignment counter
        available_contexts = []
        for env_id in range(len(runtime_topics.topics)):
            for robot_id in range(len(runtime_topics.topics[env_id])):
                try:
                    topic = runtime_topics.topics[env_id][robot_id]
                    if 'robot_id' in topic:
                        available_contexts.append(topic)
                except (IndexError, KeyError):
                    continue
        
        # Sort by robot_id for deterministic assignment
        available_contexts.sort(key=lambda x: x['robot_id'])
        
        # Assign context based on creation order
        if not hasattr(DynamicObsCost, '_assignment_index'):
            DynamicObsCost._assignment_index = 0
        
        if DynamicObsCost._assignment_index < len(available_contexts):
            robot_context = available_contexts[DynamicObsCost._assignment_index]
            self.robot_id = robot_context['robot_id']
            DynamicObsCost._assignment_index += 1
            print(f"DynamicObsCost assigned to robot {self.robot_id}")
        else:
            print(f"ERROR: No robot context available - found {len(available_contexts)} contexts")
            self.col_pred = None
            self.disable_cost()
            return
        
        # Get required values from robot context - fail if any are missing
        required_keys = ['robot_pose', 'n_obstacle_spheres', 'n_own_spheres', 'horizon', 'n_rollouts']
        for key in required_keys:
            if key not in robot_context:
                print(f"ERROR: Required key '{key}' missing from robot context for robot {self.robot_id}")
                self.col_pred = None
                self.disable_cost()
                return
        
        X = robot_context['robot_pose']
        n_coll_spheres = robot_context['n_obstacle_spheres']
        n_own_spheres = robot_context['n_own_spheres']
        horizon = robot_context['horizon']
        n_rollouts = robot_context['n_rollouts']
        
        # Log key initialization parameters
        print(f"DynamicObsCost initialized for robot {self.robot_id}: {n_coll_spheres} obstacle spheres, {n_own_spheres} own spheres")
        
        if n_coll_spheres == 0:
            # If no obstacle spheres, disable this cost and set col_pred to None
            self.col_pred = None
            self.disable_cost()
            return
        
        self.col_pred = DynamicObsCollPredictor(
            self.tensor_args,
            horizon,
            n_rollouts,
            n_own_spheres,
            n_coll_spheres,
            weight_value,
            X,
            self.sparse_steps,
            self.sparse_spheres
        )
        print(f"DynamicObsCost successfully initialized for robot {self.robot_id} with {n_coll_spheres} obstacle spheres")



    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """
        Compute dynamic obstacle collision cost.
        
        Args:
            state: Robot kinematic state containing joint information
            
        Returns:
            Cost tensor of shape [batch, horizon]
        """
        if not self.enabled or self.col_pred is None:
            # Return zero cost if disabled or no collision predictor
            if state.robot_spheres is not None:
                b, h = state.robot_spheres.shape[:2]
                return torch.zeros(b, h, device=state.robot_spheres.device, dtype=state.robot_spheres.dtype)
            else:
                # Fallback if robot_spheres is None - return minimal tensor
                return torch.zeros(1, 1, device=self.tensor_args.device)
            
        if state.robot_spheres is None:
            raise RuntimeError("Robot spheres not available in state")
            
        return self.col_pred.cost_fn(state.robot_spheres)

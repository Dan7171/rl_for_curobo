"""
Example energy minimization cost for arm_base.
Users can copy this file and modify it for their own needs.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import torch
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import runtime_topics
from projects_root.utils.transforms import transform_poses_batched


@dataclass
class EgocentricmCostConfig(CostConfig):
    """Configuration for energy minimization cost."""
    
    p_otherRobotsTargets: list[torch.Tensor] = field(default_factory=lambda: [])
    robot_id: int = 0
    env_id: int = 0
    sparse_spheres_exclude_self:list[int] = field(default_factory=lambda: [61,62,63,64]) # graspped object: 61,62,63,64
            
    def __post_init__(self):
        return super().__post_init__()

    
class EgocentricmCost(CostBase, EgocentricmCostConfig):
    """
    Energy minimization cost that penalizes high velocities and accelerations.
    This is an arm_base cost - general robot behavior, not task-specific.
    """
    
    def __init__(self, config: EgocentricmCostConfig):
        EgocentricmCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self.n_own_spheres = runtime_topics.topics[self.env_id][self.robot_id]['mpc_cfg']['cost']['custom']['n_own_spheres'] # n_own_spheres # number of valid spheres of the robot (ignoring 4 spheres which are not valid due to negative radius)
        self.valid_spheres = np.array(list(set(list(range(self.n_own_spheres))) - set(self.sparse_spheres_exclude_self)))
    def __post_init__(self):
        return super().__post_init__()

    
    def forward(self, state: KinematicModelState) -> torch.Tensor:
        """
        
        Args:
            state: Robot kinematic state containing joint information
            
        Returns:
            Cost tensor of shape [batch, horizon]
        """
        eps = 1e-6

        env_topics = runtime_topics.topics[self.env_id]
        pose_cost_matrix = runtime_topics.topics[self.env_id ][self.robot_id]["cost"]["pose"]
        pose_cost_matrix += eps
        # calc the cost of achieving the goal, inversely proportional to the pose cost
        goal_achievement_costs = 1 / pose_cost_matrix # the inverse of the pose cost
        ans = torch.zeros(pose_cost_matrix.shape, device=pose_cost_matrix.device)
        
        if len(ans) == 1: # initiation
            n_cols = runtime_topics.topics[self.env_id][self.robot_id]['mpc_cfg']['model']['horizon']
            n_rows = runtime_topics.topics[self.env_id][self.robot_id]['mpc_cfg']['mppi']['num_particles']
            return torch.zeros(n_rows, n_cols, device=pose_cost_matrix.device)

        p_spheres_rollouts = env_topics[self.robot_id]["p_spheres_rollouts"]
        p_spheres_rollouts = p_spheres_rollouts[:,:,self.valid_spheres]
        if len(p_spheres_rollouts):
            for id in range(len(env_topics)):    
                if id != self.robot_id:
                    X_targetOther = env_topics[id]["target"]
                    # for each rollout and step, calc the min distance from some sphere of self, to the target of other robot    
                    min_sphere_dist_to_other_target = torch.linalg.norm(p_spheres_rollouts - torch.from_numpy(X_targetOther[:3]).to(p_spheres_rollouts.device),dim=3).min(dim=2)[0] 
                    # make sure no division by zero
                    min_sphere_dist_to_other_target = min_sphere_dist_to_other_target + eps

                    # calc the cost of disturbing the other robot's target, 
                    # inversely proportional to the distance to other robot's target
                    proximity_to_other_target_cost =  1 / min_sphere_dist_to_other_target
                    
                    # finally, multiply the cost of disturbing the other robot's target by the goal achievement cost
                    # means that the more self achieves the goal, the more it's penalized for being around other robot's target
                    cost_of_disturbing_other_target = goal_achievement_costs * proximity_to_other_target_cost # element-wise multiplication 

                    # add the cost of disturbing the other robot's target to the total cost 
                    ans += cost_of_disturbing_other_target
        
        # finally, multiply the total cost by the (fixed, hyperparameter) weight of the cost
        return self.weight * ans
        

"""
Dynamic obstacle collision cost for arm_base.
This cost dynamically computes collision avoidance with other robots.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import torch
from curobo.rollout.cost.cost_base import CostBase, CostConfig
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.dynamic_obs_coll_checker import DynamicObsCollPredictor
from projects_root.projects.dynamic_obs.dynamic_obs_predictor.runtime_topics import get_topics
from curobo.util_file import load_yaml

@dataclass
class DynamicObsCostConfig(CostConfig):
    """Configuration for dynamic obstacle collision cost."""
    weight: float = 100.0    
    sparse_steps: dict = field(default_factory=lambda: {'use': False, 'ratio': 0.5})
    sparse_spheres: dict = field(default_factory=lambda: {'use': False})
    
    def __post_init__(self):
        return super().__post_init__()

    
class DynamicObsCost(CostBase, DynamicObsCostConfig):
    """
    Dynamic obstacle collision cost that avoids collisions with other robots.
    This cost automatically computes robot poses and obstacle counts from runtime context.
    """
    
    def __init__(self, config: DynamicObsCostConfig):
        DynamicObsCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        
        # Extract weight value properly (handle torch tensor case)
        weight_value = self.weight
        if isinstance(weight_value, torch.Tensor):
            weight_value = weight_value.cpu().item()
        
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
        else:
            print(f"ERROR: No robot context available - found {len(available_contexts)} contexts")
            self.col_pred = None
            self.disable_cost()
            return
        
        # Get required values from robot context - fail if any are missing
        required_keys = ['robot_pose', 'n_obstacle_spheres', 'n_own_spheres', 'horizon', 'n_rollouts', 'col_pred_with']
        for key in required_keys:
            if key not in robot_context:
                print(f"ERROR: Required key '{key}' missing from robot context for robot {self.robot_id}")
                self.col_pred = None
                self.disable_cost()
                return
        
        X = robot_context['robot_pose']
        n_obstacle_spheres = robot_context['n_obstacle_spheres']
        n_own_spheres = robot_context['n_own_spheres']
        horizon = robot_context['horizon']
        n_rollouts = robot_context['n_rollouts']
        col_pred_with = robot_context['col_pred_with']
        
        if n_obstacle_spheres == 0:
            # If no obstacle spheres, disable this cost and set col_pred to None
            self.col_pred = None
            self.disable_cost()
            return
        
        # Extract sparse sphere filtering configuration
        use_sparse_spheres = self.sparse_spheres.get('use', False)
        
        # Get published sphere exclusion info for this robot and other robots
        spheres_to_exclude_self = []
        col_with_idx_map = {}
        
        if use_sparse_spheres:
            # Get sphere exclusion info from robot context (passed from main script)
            required_sparse_keys = ['mpc_config_paths', 'robot_config_paths', 'robot_sphere_counts']
            for key in required_sparse_keys:
                if key not in robot_context:
                    print(f"ERROR: Required sparse key '{key}' missing from robot context for robot {self.robot_id}")
                    self.col_pred = None
                    self.disable_cost()
                    return
            
            mpc_config_paths = robot_context['mpc_config_paths']
            robot_config_paths = robot_context['robot_config_paths']
            robot_sphere_counts = robot_context['robot_sphere_counts']  # [(base_count, extra_count), ...]
            
            # Get this robot's exclusion list from MPC config
            if len(mpc_config_paths) > self.robot_id:
                mpc_config = load_yaml(mpc_config_paths[self.robot_id])
                if 'cost' in mpc_config and 'custom' in mpc_config['cost'] and 'published_info' in mpc_config['cost']['custom']:
                    spheres_to_exclude_self = mpc_config['cost']['custom']['published_info'].get('spheres_to_exclude_in_sparse_mode', [])
            
            # Build col_with_idx_map for other robots
            current_idx = 0
            for other_robot_id in col_pred_with:
                # Load MPC config for other robot to get their exclusion list
                spheres_to_exclude_other = []
                if len(mpc_config_paths) > other_robot_id:
                    other_mpc_config = load_yaml(mpc_config_paths[other_robot_id])
                    if 'cost' in other_mpc_config and 'custom' in other_mpc_config['cost'] and 'published_info' in other_mpc_config['cost']['custom']:
                        raw_exclusion_list = other_mpc_config['cost']['custom']['published_info'].get('spheres_to_exclude_in_sparse_mode', [])
                        
                        # Get total sphere count for this other robot from pre-calculated values
                        other_base_spheres, other_extra_spheres = robot_sphere_counts[other_robot_id]
                        other_total_spheres = other_base_spheres + other_extra_spheres
                        
                        # CRITICAL FIX: Validate and filter exclusion indices
                        spheres_to_exclude_other = [idx for idx in raw_exclusion_list if 0 <= idx < other_total_spheres]
                        invalid_indices = [idx for idx in raw_exclusion_list if idx >= other_total_spheres]
                        
                        if invalid_indices:
                            print(f"WARNING: Robot {other_robot_id} has invalid exclusion indices {invalid_indices} (total spheres: {other_total_spheres})")
                
                # Get total sphere count for this other robot from pre-calculated values (moved up for validation)
                other_base_spheres, other_extra_spheres = robot_sphere_counts[other_robot_id]
                other_total_spheres = other_base_spheres + other_extra_spheres
                
                # Calculate valid sphere indices after filtering
                valid_indices_other = list(set(range(other_total_spheres)) - set(spheres_to_exclude_other))
                n_valid_other = len(valid_indices_other)
                
                # Map this robot to its index range in the concatenated tensor
                col_with_idx_map[other_robot_id] = {
                    'start_idx': current_idx,
                    'end_idx': current_idx + n_valid_other,
                    'valid_indices': valid_indices_other,
                    'exclude_list': spheres_to_exclude_other
                }
                current_idx += n_valid_other
        
        self.col_pred = DynamicObsCollPredictor(
            self.tensor_args,
            horizon,
            n_rollouts,
            n_own_spheres,
            n_obstacle_spheres,
            weight_value,
            X,
            self.sparse_steps,
            {
                'use': use_sparse_spheres,
                'exclude_self': spheres_to_exclude_self,
                'exclude_others': []  # Not used anymore - handled by col_with_idx_map
            },
            col_with_idx_map
        )
        print(f"DynamicObsCost successfully initialized for robot {self.robot_id} with {n_obstacle_spheres} obstacle spheres")

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

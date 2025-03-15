import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import torch.nn.functional as F

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.geom.types import WorldConfig, Cuboid
from curobo.util.logger import setup_curobo_logger
from curobo.wrap.reacher.mpc import MpcSolver

from .....rl_algs_api.ActorCritic import ActorCritic, ActorCriticConfig

@dataclass
class RLMPCConfig:
    # State representation config
    use_debug_mode: bool = True  # If True, use precise object info, else use depth camera
    robot_state_dim: int = 28  # 7 joints x 4 (pos, vel, acc, jerk)
    goal_pose_dim: int = 7  # position (3) + quaternion (4)
    world_state_dim: int = 50  # Configurable based on max objects/features
    
    # RL config
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reward config
    collision_penalty: float = -100.0
    goal_reward: float = 100.0
    time_penalty: float = -0.1
    distance_scale: float = 1.0
    
    # MPC config
    mpc_horizon: int = 10
    mpc_dt: float = 0.1

class RLMPCAgent:
    def __init__(self, config: RLMPCConfig, mpc_solver: MpcSolver):
        self.config = config
        self.mpc_solver = mpc_solver
        self.tensor_args = TensorDeviceType()
        
        # Calculate total state dimension
        state_dim = (config.robot_state_dim + 
                    config.goal_pose_dim * 2 +  # Original goal + current goal
                    config.world_state_dim)
        
        # Action dimension is goal pose (position + quaternion)
        action_dim = config.goal_pose_dim
        
        # Initialize RL algorithm
        ac_config = ActorCriticConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_grad_norm=config.clip_grad_norm,
            device=config.device
        )
        self.rl_algorithm = ActorCritic(ac_config)
        
    def get_robot_state(self, joint_state: JointState) -> torch.Tensor:
        """Extract robot state features from joint state."""
        state_list = []
        
        # Position
        state_list.append(torch.tensor(joint_state.position, device=self.config.device))
        
        # Velocity
        state_list.append(torch.tensor(joint_state.velocity, device=self.config.device))
        
        # Acceleration
        state_list.append(torch.tensor(joint_state.acceleration, device=self.config.device))
        
        # Jerk
        state_list.append(torch.tensor(joint_state.jerk, device=self.config.device))
        
        return torch.cat(state_list)

    def get_world_state_debug(self, world: WorldConfig) -> torch.Tensor:
        """Get world state representation using precise object information."""
        state_list = []
        max_objects = self.config.world_state_dim // 10  # 10 features per object
        
        # Get obstacles through the obstacles list
        for i, obstacle in enumerate(world.obstacles):
            if i >= max_objects:
                break
                
            # Position (3)
            state_list.append(torch.tensor(obstacle.pose[:3], device=self.config.device))
            
            # Orientation (4)
            state_list.append(torch.tensor(obstacle.pose[3:7], device=self.config.device))
            
            # Dimensions for cuboids or radius for spheres (1-3)
            if isinstance(obstacle, Cuboid):
                state_list.append(torch.tensor(obstacle.dims, device=self.config.device))
            else:  # Sphere
                state_list.append(torch.tensor([obstacle.radius, 0, 0], device=self.config.device))
                
        # Pad if needed
        if len(state_list) < max_objects:
            padding = torch.zeros(max_objects - len(state_list), 10, device=self.config.device)
            state_list.append(padding)
            
        return torch.cat(state_list)

    def get_world_state_camera(self, depth_image: torch.Tensor) -> torch.Tensor:
        """Get world state representation using depth camera information."""
        # Process depth image to extract features
        # This is a placeholder - implement actual depth image processing
        processed_features = F.adaptive_avg_pool2d(depth_image, (5, 10))  # Example size
        return processed_features.flatten()[:self.config.world_state_dim]

    def get_state_representation(self, 
                               robot_state: JointState,
                               original_goal: Pose,
                               current_goal: Pose,
                               world_config: WorldConfig,
                               depth_image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine all state components into a single representation."""
        state_components = []
        
        # 1. Robot state
        robot_features = self.get_robot_state(robot_state)
        state_components.append(robot_features)
        
        # 2. Original goal pose
        state_components.append(torch.cat([
            torch.tensor(original_goal.position, device=self.config.device),
            torch.tensor(original_goal.quaternion, device=self.config.device)
        ]))
        
        # 3. Current goal pose
        state_components.append(torch.cat([
            torch.tensor(current_goal.position, device=self.config.device),
            torch.tensor(current_goal.quaternion, device=self.config.device)
        ]))
        
        # 4. World state
        if self.config.use_debug_mode:
            world_features = self.get_world_state_debug(world_config)
        else:
            assert depth_image is not None, "Depth image required in camera mode"
            world_features = self.get_world_state_camera(depth_image)
        state_components.append(world_features)
        
        return torch.cat(state_components)

    def compute_reward(self,
                      robot_state: JointState,
                      original_goal: Pose,
                      current_goal: Pose,
                      collision_checker: Any,
                      done: bool) -> float:
        """Compute reward based on goal progress and collisions."""
        reward = self.config.time_penalty  # Base time penalty
        
        # Check for collisions
        collision_cost = collision_checker.get_collision_cost()
        if collision_cost > 0:
            reward += self.config.collision_penalty * collision_cost
            
        # Distance to goal reward
        # Get end-effector pose through forward kinematics
        current_pose = self.mpc_solver.get_ee_pose(robot_state)
        distance_to_goal = torch.norm(current_pose.position - original_goal.position).item()
        reward -= self.config.distance_scale * distance_to_goal
        
        # Goal achievement reward
        if done and distance_to_goal < 0.05:  # 5cm threshold
            reward += self.config.goal_reward
            
        return float(reward)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Pose:
        """Select a new goal pose based on current state."""
        action = self.rl_algorithm.select_action(state, deterministic)
        
        # Convert action to Pose
        position = action[:3]
        quaternion = action[3:]
        
        # Normalize quaternion
        quaternion = quaternion / torch.norm(quaternion)
        
        return Pose(position=position, quaternion=quaternion)

    def update(self, 
               states: List[torch.Tensor],
               actions: List[torch.Tensor],
               rewards: List[float],
               dones: List[bool],
               next_states: List[torch.Tensor]) -> Dict[str, float]:
        """Update the RL algorithm with collected experience."""
        # Convert lists to tensors
        states_tensor = torch.stack(states)
        actions_tensor = torch.stack(actions)
        rewards_tensor = torch.tensor(rewards, device=self.config.device)
        dones_tensor = torch.tensor(dones, device=self.config.device)
        next_states_tensor = torch.stack(next_states)
        
        return self.rl_algorithm.update(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            dones_tensor,
            next_states_tensor
        )

    def save(self, path: str):
        """Save RL agent state."""
        self.rl_algorithm.save(path)

    def load(self, path: str):
        """Load RL agent state."""
        self.rl_algorithm.load(path)






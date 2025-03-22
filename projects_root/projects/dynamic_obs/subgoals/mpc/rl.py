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
from curobo.geom.sdf.world import CollisionQueryBuffer
from projects_root.utils.rl_algs_api.ActorCritic import ActorCritic, ActorCriticConfig

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
        """Extract robot state features from joint state.
        
        Returns a tensor of shape (robot_state_dim,) containing:
        - Position (7)
        - Velocity (7)
        - Acceleration (7)
        - Jerk (7)
        Total: 28 features
        """
        # Initialize full state with zeros
        full_state = torch.zeros(self.config.robot_state_dim, device=self.config.device)
        
        # Get number of joints
        n_joints = 7  # Assuming 7-DOF robot
        
        # Position (first 7 elements)
        if joint_state.position is not None:
            pos = torch.as_tensor(joint_state.position, device=self.config.device)
            full_state[:n_joints] = pos[:n_joints]
            
        # Velocity (next 7 elements)
        if joint_state.velocity is not None:
            vel = torch.as_tensor(joint_state.velocity, device=self.config.device)
            full_state[n_joints:2*n_joints] = vel[:n_joints]
            
        # Acceleration (next 7 elements)
        if joint_state.acceleration is not None:
            acc = torch.as_tensor(joint_state.acceleration, device=self.config.device)
            full_state[2*n_joints:3*n_joints] = acc[:n_joints]
            
        # Jerk (final 7 elements)
        if joint_state.jerk is not None:
            jerk = torch.as_tensor(joint_state.jerk, device=self.config.device)
            full_state[3*n_joints:4*n_joints] = jerk[:n_joints]
            
        return full_state

    def get_world_state_debug(self, world: WorldConfig) -> torch.Tensor:
        """Get world state representation using precise object information."""
        state_list = []
        max_objects = self.config.world_state_dim // 10  # 10 features per object
        
        # Get obstacles through the world config
        obstacles = []
        if world.cuboid is not None:
            obstacles.extend(world.cuboid)
        if world.mesh is not None:
            obstacles.extend(world.mesh)
        if world.sphere is not None:
            obstacles.extend(world.sphere)
        
        # Initialize full state with zeros
        full_state = torch.zeros(self.config.world_state_dim, device=self.config.device)
        current_idx = 0
        
        for i, obstacle in enumerate(obstacles):
            if i >= max_objects:
                break
                
            if obstacle.pose is None:
                continue
                
            # Each obstacle takes 10 features:
            # Position (3) + Orientation (4) + Dimensions (3) = 10 total
            features = []
            
            # Position (3)
            pos = torch.tensor(obstacle.pose[:3], device=self.config.device)
            features.extend(pos.tolist())
            
            # Orientation (4)
            quat = torch.tensor(obstacle.pose[3:7], device=self.config.device)
            features.extend(quat.tolist())
            
            # Dimensions (3)
            if isinstance(obstacle, Cuboid):
                dims = torch.tensor(obstacle.dims, device=self.config.device)
            elif hasattr(obstacle, 'radius'):
                dims = torch.tensor([obstacle.radius, 0, 0], device=self.config.device)
            else:
                dims = torch.tensor([0.1, 0.1, 0.1], device=self.config.device)
            features.extend(dims.tolist())
            
            # Add to full state
            if current_idx + 10 <= self.config.world_state_dim:
                full_state[current_idx:current_idx + 10] = torch.tensor(features, device=self.config.device)
                current_idx += 10
        
        return full_state

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
        state_components.append(robot_features.flatten())  # Ensure 1D
        
        # 2. Original goal pose - handle position and quaternion separately
        orig_pos = original_goal.position.clone().detach().to(device=self.config.device)
        orig_quat = original_goal.quaternion.clone().detach().to(device=self.config.device)
        state_components.append(orig_pos.flatten())  # Ensure 1D
        state_components.append(orig_quat.flatten())  # Ensure 1D
        
        # 3. Current goal pose - handle position and quaternion separately
        curr_pos = current_goal.position.clone().detach().to(device=self.config.device)
        curr_quat = current_goal.quaternion.clone().detach().to(device=self.config.device)
        state_components.append(curr_pos.flatten())  # Ensure 1D
        state_components.append(curr_quat.flatten())  # Ensure 1D
        
        # 4. World state
        if self.config.use_debug_mode:
            world_features = self.get_world_state_debug(world_config)
        else:
            assert depth_image is not None, "Depth image required in camera mode"
            world_features = self.get_world_state_camera(depth_image)
        state_components.append(world_features.flatten())  # Ensure 1D
        
        # Concatenate all components and ensure result is 1D
        state = torch.cat(state_components).flatten()
        
        # Verify state dimension matches expected
        expected_dim = (self.config.robot_state_dim + 
                       self.config.goal_pose_dim * 2 +  # Original goal + current goal
                       self.config.world_state_dim)
        assert state.shape[0] == expected_dim, f"State dimension mismatch. Expected {expected_dim}, got {state.shape[0]}"
        
        return state.unsqueeze(0)  # Add batch dimension

    def compute_reward(self,
                      robot_state: JointState,
                      original_goal: Pose,
                      current_goal: Pose,
                      collision_checker: Any,  # Not used, kept for compatibility
                      done: bool) -> float:
        """Compute reward based on goal progress and collisions."""
        reward = self.config.time_penalty  # Base time penalty
        distance_to_goal = float('inf')
        
        # Check for collisions using world collision checker
        if (self.mpc_solver.world_collision is not None and 
            hasattr(robot_state, 'position') and 
            robot_state.position is not None):
            # Create query spheres for collision checking
            pos_tensor = torch.tensor(robot_state.position, device=self.config.device)
            radius_tensor = torch.tensor([0.1], device=self.config.device)
            query_spheres = torch.cat([pos_tensor, radius_tensor]).unsqueeze(0)  # Shape: [1, 4]
            
            collision_buffer = CollisionQueryBuffer()
            collision_buffer.update_buffer_shape(query_spheres.shape, self.tensor_args, self.mpc_solver.world_collision.collision_types)
            
            collision_cost = self.mpc_solver.world_collision.get_sphere_distance(
                query_spheres,
                collision_buffer,
                weight=torch.tensor([1.0], device=self.config.device),
                activation_distance=torch.tensor([0.0], device=self.config.device)
            )
            if collision_cost > 0:
                reward += self.config.collision_penalty * collision_cost.item()
        
        # Distance to goal reward
        # Get end-effector pose through forward kinematics
        if (hasattr(robot_state, 'position') and 
            robot_state.position is not None and 
            hasattr(self.mpc_solver, 'rollout_fn')):
            kinematics_state = self.mpc_solver.rollout_fn.compute_kinematics(robot_state)
            if (kinematics_state is not None and 
                hasattr(kinematics_state, 'ee_pos_seq') and 
                kinematics_state.ee_pos_seq is not None and
                hasattr(kinematics_state, 'ee_quat_seq') and
                kinematics_state.ee_quat_seq is not None):
                # Get the last position and quaternion if sequence, or use as is if single state
                ee_pos = kinematics_state.ee_pos_seq[-1] if isinstance(kinematics_state.ee_pos_seq, torch.Tensor) and kinematics_state.ee_pos_seq.dim() > 1 else kinematics_state.ee_pos_seq
                ee_quat = kinematics_state.ee_quat_seq[-1] if isinstance(kinematics_state.ee_quat_seq, torch.Tensor) and kinematics_state.ee_quat_seq.dim() > 1 else kinematics_state.ee_quat_seq
                
                current_pose = Pose(position=ee_pos, quaternion=ee_quat)
                if (original_goal.position is not None and 
                    current_pose.position is not None):
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






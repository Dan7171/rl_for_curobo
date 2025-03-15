import torch
import numpy as np
import os
import argparse
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

# CuRobo imports
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig, Cuboid
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

# Local imports
from .rl import RLMPCAgent, RLMPCConfig

@dataclass
class TrainingConfig:
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    eval_interval: int = 10
    save_interval: int = 100
    
    # Environment parameters
    num_obstacles: int = 3
    obstacle_speed_range: Tuple[float, float] = (-0.2, 0.2)
    workspace_bounds: Dict[str, Tuple[float, float]] = {
        'x': (0.0, 1.0),
        'y': (-0.5, 0.5),
        'z': (0.0, 1.0)
    }
    
    # Paths
    model_save_path: str = "models"
    robot_config: str = "franka.yml"

def setup_environment(world_config: WorldConfig, config: TrainingConfig) -> Tuple[WorldConfig, List[Cuboid]]:
    """Set up the simulation environment with obstacles."""
    obstacles = []
    
    # Create random obstacles
    for i in range(config.num_obstacles):
        # Random position within workspace
        pos = [
            np.random.uniform(*config.workspace_bounds['x']),
            np.random.uniform(*config.workspace_bounds['y']),
            np.random.uniform(*config.workspace_bounds['z'])
        ]
        
        # Random size
        size = np.random.uniform(0.05, 0.15)
        
        # Create obstacle
        cuboid = Cuboid(
            name=f"obstacle_{i}",
            pose=[*pos, 1.0, 0.0, 0.0, 0.0],  # position + quaternion
            dims=[size, size, size]
        )
        obstacles.append(cuboid)
        world_config.add_obstacle(cuboid)
    
    return world_config, obstacles

def update_obstacles(world_config: WorldConfig, 
                    obstacles: List[Cuboid], 
                    config: TrainingConfig,
                    dt: float):
    """Update obstacle positions based on their velocities."""
    for i, obstacle in enumerate(obstacles):
        # Get current position
        pos = obstacle.pose[:3]
        
        # Update position with velocity
        vel = [
            np.random.uniform(*config.obstacle_speed_range),
            np.random.uniform(*config.obstacle_speed_range),
            np.random.uniform(*config.obstacle_speed_range)
        ]
        new_pos = pos + np.array(vel) * dt
        
        # Keep within workspace bounds
        for j, (axis, bounds) in enumerate(config.workspace_bounds.items()):
            if new_pos[j] < bounds[0] or new_pos[j] > bounds[1]:
                vel[j] *= -1  # Reverse direction if hitting boundary
                new_pos[j] = np.clip(new_pos[j], bounds[0], bounds[1])
        
        # Update obstacle pose
        obstacle.pose = [*new_pos, 1.0, 0.0, 0.0, 0.0]  # position + quaternion

def train(config: TrainingConfig):
    """Main training loop."""
    # Configure CuRobo logging
    setup_curobo_logger("warn")
    tensor_args = TensorDeviceType()
    
    # Load robot configuration
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), config.robot_config))["robot_cfg"]
    
    # Create initial world config
    world_config = WorldConfig()
    
    # Set up MPC solver
    mpc_config = MpcSolverConfig.load_from_robot_config(
        robot_cfg=robot_cfg,
        world_model=world_config,  # Use initial world config
        use_cuda_graph=True,
        collision_checker_type=CollisionCheckerType.MESH,
        store_rollouts=True,
        step_dt=0.02
    )
    mpc_solver = MpcSolver(mpc_config)
    
    # Initialize RL agent
    rl_config = RLMPCConfig()
    rl_agent = RLMPCAgent(rl_config, mpc_solver)
    
    # Training loop
    episode_rewards = []
    
    for episode in range(config.num_episodes):
        print(f"Starting episode {episode}")
        
        # Reset environment
        world_config = WorldConfig()
        world_config, obstacles = setup_environment(world_config, config)
        mpc_solver.update_world(world_config)
        
        # Initialize episode variables
        episode_reward = 0
        states = []
        actions = []
        rewards = []
        dones = []
        next_states = []
        
        # Set random initial robot configuration
        initial_joint_state = JointState.from_position(
            torch.rand(7, device=tensor_args.device) * 2 - 1  # Random between -1 and 1
        )
        
        # Set random goal
        goal_position = torch.tensor([
            np.random.uniform(*config.workspace_bounds['x']),
            np.random.uniform(*config.workspace_bounds['y']),
            np.random.uniform(*config.workspace_bounds['z'])
        ], device=tensor_args.device)
        goal_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=tensor_args.device)
        original_goal = Pose(position=goal_position, quaternion=goal_quaternion)
        
        # Episode loop
        for step in range(config.max_steps_per_episode):
            # Get current state
            current_joint_state = initial_joint_state  # In real implementation, get from robot
            current_state = rl_agent.get_state_representation(
                robot_state=current_joint_state,
                original_goal=original_goal,
                current_goal=original_goal,  # Initially same as original
                world_config=world_config
            )
            
            # Select action (new goal pose)
            current_goal = rl_agent.select_action(current_state)
            
            # Update MPC goal
            mpc_goal = Goal(
                current_state=current_joint_state,
                goal_pose=current_goal
            )
            mpc_solver.update_goal(mpc_goal)
            
            # Run MPC step
            mpc_result = mpc_solver.step(current_joint_state)
            
            # Update obstacles
            update_obstacles(world_config, obstacles, config, dt=0.02)
            mpc_solver.update_world(world_config)
            
            # Get next state
            next_joint_state = current_joint_state  # In real implementation, get from robot
            next_state = rl_agent.get_state_representation(
                robot_state=next_joint_state,
                original_goal=original_goal,
                current_goal=current_goal,
                world_config=world_config
            )
            
            # Compute reward and done flag
            reward = rl_agent.compute_reward(
                robot_state=next_joint_state,
                original_goal=original_goal,
                current_goal=current_goal,
                collision_checker=mpc_solver.world_collision_checker,
                done=(step == config.max_steps_per_episode - 1)
            )
            done = (step == config.max_steps_per_episode - 1)
            
            # Store transition
            states.append(current_state)
            actions.append(current_goal)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            episode_reward += reward
            
            if done:
                break
        
        # Update RL agent
        update_info = rl_agent.update(states, actions, rewards, dones, next_states)
        episode_rewards.append(episode_reward)
        
        # Logging
        print(f"Episode {episode} - Reward: {episode_reward:.2f}")
        print(f"Actor Loss: {update_info['actor_loss']:.4f}")
        print(f"Critic Loss: {update_info['critic_loss']:.4f}")
        
        # Save model periodically
        if episode % config.save_interval == 0:
            os.makedirs(config.model_save_path, exist_ok=True)
            rl_agent.save(os.path.join(config.model_save_path, f"model_ep_{episode}.pt"))
        
        # Evaluation
        if episode % config.eval_interval == 0:
            eval_reward = evaluate(rl_agent, mpc_solver, config)
            print(f"Evaluation Reward: {eval_reward:.2f}")

def evaluate(rl_agent: RLMPCAgent, 
            mpc_solver: MpcSolver, 
            config: TrainingConfig,
            num_eval_episodes: int = 5) -> float:
    """Evaluate the current policy."""
    eval_rewards = []
    
    for _ in range(num_eval_episodes):
        # Reset environment
        world_config = WorldConfig()
        world_config, obstacles = setup_environment(world_config, config)
        mpc_solver.update_world(world_config)
        
        # Set random goal and initial state
        initial_joint_state = JointState.from_position(
            torch.rand(7, device=rl_agent.config.device) * 2 - 1
        )
        
        goal_position = torch.tensor([
            np.random.uniform(*config.workspace_bounds['x']),
            np.random.uniform(*config.workspace_bounds['y']),
            np.random.uniform(*config.workspace_bounds['z'])
        ], device=rl_agent.config.device)
        goal_quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0], device=rl_agent.config.device)
        original_goal = Pose(position=goal_position, quaternion=goal_quaternion)
        
        episode_reward = 0
        for step in range(config.max_steps_per_episode):
            # Run episode with deterministic policy
            current_state = rl_agent.get_state_representation(
                robot_state=initial_joint_state,
                original_goal=original_goal,
                current_goal=original_goal,
                world_config=world_config
            )
            
            current_goal = rl_agent.select_action(current_state, deterministic=True)
            
            mpc_goal = Goal(
                current_state=initial_joint_state,
                goal_pose=current_goal
            )
            mpc_solver.update_goal(mpc_goal)
            
            mpc_result = mpc_solver.step(initial_joint_state)
            
            update_obstacles(world_config, obstacles, config, dt=0.02)
            mpc_solver.update_world(world_config)
            
            next_state = rl_agent.get_state_representation(
                robot_state=initial_joint_state,
                original_goal=original_goal,
                current_goal=current_goal,
                world_config=world_config
            )
            
            reward = rl_agent.compute_reward(
                robot_state=initial_joint_state,
                original_goal=original_goal,
                current_goal=current_goal,
                collision_checker=mpc_solver.world_collision_checker,
                done=(step == config.max_steps_per_episode - 1)
            )
            
            episode_reward += reward
            
            if step == config.max_steps_per_episode - 1:
                break
            
        eval_rewards.append(episode_reward)
    
    return float(np.mean(eval_rewards))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL-enhanced MPC for dynamic obstacle avoidance")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--num_obstacles", type=int, default=3)
    parser.add_argument("--model_path", type=str, default="models")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        num_obstacles=args.num_obstacles,
        model_save_path=args.model_path
    )
    
    train(config)






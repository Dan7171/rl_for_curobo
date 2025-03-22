import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

@dataclass
class ActorCriticConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
        # For action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Normal, torch.Tensor]:
        mean = self.network(state)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std), mean

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class ActorCritic:
    def __init__(self, config: ActorCriticConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize networks
        self.actor = Actor(config.state_dim, config.action_dim, 
                          config.hidden_dim, config.num_hidden_layers).to(self.device)
        self.critic = Critic(config.state_dim, config.hidden_dim, 
                           config.num_hidden_layers).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)

    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select an action given current state."""
        with torch.no_grad():
            dist, mean = self.actor(state)
            if deterministic:
                action = mean
            else:
                action = dist.sample()
        return action

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                   next_value: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        running_returns = next_value
        running_advs = 0

        for t in reversed(range(len(rewards))):
            running_returns = rewards[t] + self.config.gamma * running_returns * (1 - dones[t])
            returns[t] = running_returns

            td_error = rewards[t] + self.config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            running_advs = td_error + self.config.gamma * self.config.gae_lambda * running_advs * (1 - dones[t])
            advantages[t] = running_advs

        return returns, advantages

    def update(self, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
              dones: torch.Tensor, next_states: torch.Tensor) -> Dict[str, float]:
        """Update actor and critic networks."""
        # Get values for all states including next state
        with torch.no_grad():
            next_value = self.critic(next_states[-1:])
            values = self.critic(states)
            values = torch.cat([values, next_value], dim=0)

        # Compute returns and advantages
        returns, advantages = self.compute_gae(rewards, values[:-1], values[-1], dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get current action distribution and values
        dist, _ = self.actor(states)
        curr_values = self.critic(states)
        log_probs = dist.log_prob(actions).sum(-1)

        # Compute actor loss
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Compute critic loss
        critic_loss = F.mse_loss(curr_values, returns)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.clip_grad_norm)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.clip_grad_norm)
        self.critic_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "mean_advantage": advantages.mean().item()
        }

    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict']) 
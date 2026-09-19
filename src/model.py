import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DuelingQNetwork(nn.Module):
    """Dueling Deep Q-Network for Snake action-value approximation."""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 256, output_dim: int = 3):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


class DQNTrainer:
    """Trainer encapsulating Bellman Equation optimization and target network updates."""

    def __init__(self, model: nn.Module, target_model: nn.Module, lr: float = 1e-3, gamma: float = 0.9):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss replaces MSELoss

    def train_step(self, states, actions, rewards, next_states, dones) -> float:
        """Executes one step of gradient descent using Bellman Q-targets."""
        # Convert inputs to tensors
        state_t = torch.tensor(np.array(states), dtype=torch.float32)
        next_state_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        action_t = torch.tensor(actions, dtype=torch.long)
        reward_t = torch.tensor(rewards, dtype=torch.float32)
        done_t = torch.tensor(dones, dtype=torch.bool)

        # Handle 1D single-transition inputs by expanding dimension to batch
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
            next_state_t = next_state_t.unsqueeze(0)
            action_t = action_t.unsqueeze(0)
            reward_t = reward_t.unsqueeze(0)
            done_t = done_t.unsqueeze(0)

        # 1. Current predicted Q values: Q(s, a)
        pred_q = self.model(state_t)
        
        # Select the Q-value corresponding to the taken action
        state_action_values = pred_q.gather(1, action_t.unsqueeze(1)).squeeze(1)

        # 2. Compute Target Q values using Double DQN
        with torch.no_grad():
            # Online network picks the action
            next_actions = self.model(next_state_t).argmax(dim=1, keepdim=True)
            # Target network evaluates the action
            next_q_values = self.target_model(next_state_t).gather(1, next_actions).squeeze(1)
            next_q_values[done_t] = 0.0  # Terminal state Q-value is 0
            expected_state_action_values = reward_t + (self.gamma * next_q_values)

        # 3. Loss computation and gradient update
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping prevents explosive gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        
        self.optimizer.step()

        return float(loss.item())

    def soft_update_target_network(self, tau: float = 0.005) -> None:
        """Applies Polyak averaging to softly update target network weights."""
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    """Deep Q-Network for Snake action-value approximation."""

    def __init__(self, input_dim: int = 11, hidden_dim: int = 256, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNTrainer:
    """Trainer encapsulating Bellman Equation optimization and target network updates."""

    def __init__(self, model: nn.Module, target_model: nn.Module, lr: float = 1e-3, gamma: float = 0.9):
        self.model = model
        self.target_model = target_model
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

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
        # action_t shape: [batch_size], gather requires [batch_size, 1]
        state_action_values = pred_q.gather(1, action_t.unsqueeze(1)).squeeze(1)

        # 2. Compute Target Q values using the Target Network: r + gamma * max_a Q_target(s', a)
        with torch.no_grad():
            next_q_values = self.target_model(next_state_t).max(dim=1)[0]
            next_q_values[done_t] = 0.0  # Terminal state Q-value is 0
            expected_state_action_values = reward_t + (self.gamma * next_q_values)

        # 3. Loss computation and gradient update
        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self) -> None:
        """Copies weights from online policy model to target network."""
        self.target_model.load_state_dict(self.model.state_dict())
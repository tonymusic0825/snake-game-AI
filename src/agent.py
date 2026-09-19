import random
from collections import deque
import numpy as np
import torch
from src.model import DuelingQNetwork, DQNTrainer

class DQNAgent:
    """Agent orchestrating action selection, experience replay, and learning."""

    def __init__(
        self,
        input_dim: int = 16,
        action_dim: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.9,
        max_memory: int = 100_000,
        batch_size: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995
    ):
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)

        # Epsilon decay configuration
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Online Network (Policy) and Target Network
        self.model = DuelingQNetwork(input_dim=input_dim, output_dim=action_dim)
        self.target_model = DuelingQNetwork(input_dim=input_dim, output_dim=action_dim)
        
        # Initialize target network with online network weights
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # Set target network to evaluation mode

        self.trainer = DQNTrainer(
            model=self.model,
            target_model=self.target_model,
            lr=lr,
            gamma=gamma
        )

    def remember(self, state, action, reward, next_state, done) -> None:
        """Stores transition step in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(state_t)
            return int(torch.argmax(prediction).item())

    def train_step(self, state, action, reward, next_state, done) -> float:
        loss = self.trainer.train_step(state, action, reward, next_state, done)
        self.trainer.soft_update_target_network()
        return loss

    def train_replay_batch(self) -> float:
        """Trains on random mini-batch sampled from memory buffer (long-term memory)."""
        if len(self.memory) < self.batch_size:
            sample_batch = self.memory
        else:
            sample_batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*sample_batch)
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)

        # ADD THIS LINE: Soft update target network every batch
        self.trainer.soft_update_target_network()

        # Decay epsilon after training on replay buffer
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

        return loss

    def save_checkpoint(self, filepath: str = "checkpoints/best_ddqn.pth") -> None:
        """Saves current online network state dictionary."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load_checkpoint(self, filepath: str = "checkpoints/best_dqn.pth") -> None:
        """Loads saved weights into policy model and updates target network."""
        self.model.load_state_dict(torch.load(filepath))
        self.target_model.load_state_dict(self.model.state_dict())
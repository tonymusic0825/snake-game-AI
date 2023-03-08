import torch
import torch.nn as nn
import os

class Model(nn.Module):
    """Neural network that will predict snake movement"""

    def __init__(self):
        """Intializes model
           This model contains 3 layers total (input, 1 hidden, output)
           Input: 12 Nodes
           Hidden: 300 Nodes
           Ouput: 4 Nodes
        """
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(13, 500),
            nn.ReLU(),
            nn.Linear(500, 4)
        )

    def forward(self, x):
        """Forward propgation"""
        x = self.nn(x)
        return x

class Trainer():
    """The trainer that will interact with the model to train"""
    def __init__(self, lr, gamma, model):
        """Initializes trainer
        
        Parameters
            lr: Learning rate for the optimizer
            gamma: gamma value for bellman equation
            model: Model to train on
            """

        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), self.lr)
        self.loss = nn.MSELoss()
    
    def train_step(self, old_state, action, reward, next_state, game_over):
        """Trains on each snake step
        
        Parameters
            old_state: The state of game before an action is taken
            action: Action to take in the current game state
            reward: Reward of the game
            next_state: The state after the 'action' is performed
            game_over: Whether game is over
        """

        # We must make our data tensors to used in pytorch model
        state = torch.tensor(old_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        next_state = torch.tensor(next_state, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # If it's training on the 'short' memory we must turn our input into a batch
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        # Apply bellman's equation for Q
        pred = self.model(state)
        target = pred.clone()

        for i in range(len(game_over)):
            Q_new = reward[i]
            if not game_over[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(next_state[i]))
            target[i][torch.argmax(action[i]).item()] = Q_new
        
        # Train
        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()
    
    def get_optimizer(self):
        """Returns the optimizer for the current trainer"""
        return self.optimizer





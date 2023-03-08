import torch
import random
import numpy as np 
from collections import deque
from simple_snake_AI import *
from model import *

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent(object):
    """The Agent that will play the game for us"""
    def __init__(self):
        """Initializes agent instance"""
        self.n_games = 0 
        self.epsilon = 0 # Check get_action() function
        self.memory = deque(maxlen=MAX_MEMORY)

        self.gamma = 0.85
        self.model = Model()
        self.trainer = Trainer(LR, self.gamma, self.model)

    def get_state(self, game: GameAI):
        """Returns the current state of the game in a list made up of boolean values
           [1. danger up, danger down, danger left, danger right, impossible direction
            2. current directions: direction up, direction down, direction left, direction right 
            3. current food position relative to the snake: food up, food down, food left, food right]

           [0, 0, 0, 0, 0
            0, 0, 0, 0
            0, 0, 0, 0]

           List will store 12 total values
        
        Parameters
            game: Game instance
        
        Returns
            list: List of 12 values 
        """

        state = []
        snake = game.get_snake()
        snake_head = snake.get_position()[0]
        sh_x = snake_head.x
        sh_y = snake_head.y
        
        # Handle 1. Danger
        danger = [0] * 5
        
        # Check up, down, left, right positions respectively
        if snake.snake_died(Position(sh_x, sh_y - BLOCK_SIZE)):
            danger[0] = 1

        if snake.snake_died(Position(sh_x, sh_y + BLOCK_SIZE)):
            danger[1] = 1

        if snake.snake_died(Position(sh_x - BLOCK_SIZE, sh_y)):
            danger[2] = 1
        
        if snake.snake_died(Position(sh_x + BLOCK_SIZE, sh_y)):
            danger[3] = 1
        
        imp_dir = snake.impossible_move_for_ai(game.direction)
        danger[4] = imp_dir
        
        state.extend(danger)

        # Handle 2. Direction
        return_direction = AGENT_DIRECTION[game.get_direction()]
        state.extend(return_direction)

        # Handle 3. Apple direction
        food = [game.apple.y < sh_y,
                game.apple.y > sh_y,
                game.apple.x < sh_x, 
                game.apple.x > sh_x]
        
        state.extend(food)
        return np.array(state, dtype=int)

    def save_memory(self, old_state, action, reward, next_state, game_over):
        """Stores the results of one action

        Parameters
            old_state: State of the game before action
            action: Action taken from the old_state
            reward: Reward given for action
            next_state: The state after action is completed
            game_over: True if snake died
        """
        # Pop left if max memory is reached
        self.memory.append((old_state, action, reward, next_state, game_over)) 

    def train_long_memory(self):
        """Trains every 1000 actions (batch)
        Parameters
            old_state: State of the game before action
            action: Action taken from the old_state
            reward: Reward given for action
            next_state: The state after action is completed
            game_over: True if snake died
        """

        # Check that there are more than 1000 samples within the memory
        # If there aren't 1000 just take the whole memory
        if len(self.memory) > BATCH_SIZE:
            r_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            r_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*r_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(self, old_state, action, reward, next_state, game_over):
        """Trains each and every action step

        Parameters
            old_state: State of the game before action
            action: Action taken from the old_state
            reward: Reward given for action
            next_state: The state after action is completed
            game_over: True if snake died
        """

        self.trainer.train_step(old_state, action, reward, next_state, game_over)

    def get_action(self, state):
        """Creates an action either randomly or through forward propagation of model and returns
           It is only randomly created during the very early stages of training
           Referred to as tradeoff exploration / exploitation
        """

        # Random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        f_move = [0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3) # Random value 0, 1, 2, 3
            f_move[move] = 1
        else:
            X = torch.tensor(state, dtype=torch.float)
            prediction = torch.argmax(self.model(X)).item()
            f_move[prediction] = 1
        
        return f_move
    
    def save(self):
        """Saves the model and optimizer's state dicts'
           Format: {model state dict
                    optimizer state dict}
        """

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.get_optimizer().state_dict()
        }, "best.pth")
    
    def load(self):
        """Loads the model and optimizer state dicts"""

        load = torch.load("best.pth")
        self.model.load_state_dict(load['model_state_dict'])
        self.trainer.get_optimizer().load_state_dict(load['optimizer_state_dict'])
            
def train():
    """Start training"""

    record = 0 
    agent = Agent()
    game = GameAI()

    while True:

        # Get current state
        current_state = agent.get_state(game)

        # Get an action from the model
        move = agent.get_action(current_state)

        # Move using the action given and get new state
        game_over, reward, score = game.step(move)
        new_state = agent.get_state(game)

        # Train short memory
        agent.train_short_memory(current_state, move, reward, new_state, game_over)

        # Save all data
        agent.save_memory(current_state, move, reward, new_state, game_over)

        # If game is over then reset the game and train the long memory
        if game_over:
            game.reset() 
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.save()
                print("MODEL AND OPTIMIZER HAS BEEN SAVED")
            
            print("Game", agent.n_games, "Score", score, "Record:", record)

if __name__ == "__main__":
    train()

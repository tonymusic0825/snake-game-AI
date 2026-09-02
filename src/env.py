from enum import Enum
import random
from typing import Tuple, List, Dict, Any
import numpy as np

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class SnakeEnv:
    """Headless Snake environment conforming to standard RL interfaces."""

    # Clockwise directional vectors: [x, y]
    CLOCKWISE = [
        (1, 0),   # RIGHT
        (0, 1),   # DOWN
        (-1, 0),  # LEFT
        (0, -1)   # UP
    ]

    def __init__(self, grid_w: int = 20, grid_h: int = 20):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.reset()

    def reset(self) -> np.ndarray:
        """Resets the environment to initial state."""
        self.direction = Direction.RIGHT
        
        # Initialize head at middle of grid
        head = (self.grid_w // 2, self.grid_h // 2)
        # Initialize initial 3-segment body extending left
        self.snake: List[Tuple[int, int]] = [
            head,
            (head[0] - 1, head[1]),
            (head[0] - 2, head[1])
        ]
        
        self.score = 0
        self.food: Tuple[int, int] = (0, 0)
        self._spawn_food()
        self.frame_iteration = 0
        
        return self.get_obs()

    def _spawn_food(self) -> None:
        """Spawns food at a random position not occupied by the snake."""
        while True:
            x = random.randint(0, self.grid_w - 1)
            y = random.randint(0, self.grid_h - 1)
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one step in the environment.
        
        Parameters:
            action: 0 -> Continue Straight
                    1 -> Turn Right 90°
                    2 -> Turn Left 90°
        
        Returns:
            observation, reward, terminated, info
        """
        self.frame_iteration += 1

        # 1. Update Direction based on relative action
        current_idx = self.direction.value
        if action == 1:    # Turn Right
            new_idx = (current_idx + 1) % 4
        elif action == 2:  # Turn Left
            new_idx = (current_idx - 1) % 4
        else:              # Action 0: Straight
            new_idx = current_idx
            
        self.direction = Direction(new_idx)

        # 2. Move Snake Head
        dir_x, dir_y = self.CLOCKWISE[self.direction.value]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dir_x, head_y + dir_y)

        # 3. Check Collision or Infinite Loop Timeout
        terminated = False
        reward = 0.0

        # Maximum steps allowed without eating food to prevent infinite looping
        max_frames = 100 * len(self.snake)

        if self.is_collision(new_head) or self.frame_iteration > max_frames:
            terminated = True
            reward = -10.0
            return self.get_obs(), reward, terminated, {"score": self.score}

        # 4. Insert New Head
        self.snake.insert(0, new_head)

        # 5. Check Food Collision & Reward Shaping
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self._spawn_food()
        else:
            self.snake.pop()
            # Slight step penalty or small reward based on distance can be added here

        return self.get_obs(), reward, terminated, {"score": self.score}

    def is_collision(self, pt: Tuple[int, int] = None) -> bool:
        """Checks if point collides with grid boundary or snake body."""
        if pt is None:
            pt = self.snake[0]

        x, y = pt
        # Boundary check
        if x < 0 or x >= self.grid_w or y < 0 or y >= self.grid_h:
            return True
        # Self collision check
        if pt in self.snake[1:]:
            return True

        return False

    def get_obs(self) -> np.ndarray:
        """
        Generates 11-feature relative state representation:
        [Danger Straight, Danger Right, Danger Left,
         Dir Left, Dir Right, Dir Up, Dir Down,
         Food Left, Food Right, Food Up, Food Down]
        """
        head_x, head_y = self.snake[0]

        # Points around head relative to current direction
        dir_idx = self.direction.value
        dir_straight = self.CLOCKWISE[dir_idx]
        dir_right = self.CLOCKWISE[(dir_idx + 1) % 4]
        dir_left = self.CLOCKWISE[(dir_idx - 1) % 4]

        pt_straight = (head_x + dir_straight[0], head_y + dir_straight[1])
        pt_right = (head_x + dir_right[0], head_y + dir_right[1])
        pt_left = (head_x + dir_left[0], head_y + dir_left[1])

        obs = [
            # Danger relative features (3)
            self.is_collision(pt_straight),
            self.is_collision(pt_right),
            self.is_collision(pt_left),

            # Current absolute direction features (4)
            self.direction == Direction.LEFT,
            self.direction == Direction.RIGHT,
            self.direction == Direction.UP,
            self.direction == Direction.DOWN,

            # Food location relative to head (4)
            self.food[0] < head_x,  # Food is Left
            self.food[0] > head_x,  # Food is Right
            self.food[1] < head_y,  # Food is Up
            self.food[1] > head_y   # Food is Down
        ]

        return np.array(obs, dtype=np.float32)
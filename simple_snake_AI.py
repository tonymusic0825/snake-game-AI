import pygame, sys
import random
from collections import namedtuple
import numpy as np

GREEN = [(0, 100, 0), (89, 204, 0)]
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (35, 35, 35)
RED = (255, 0, 0)

WINDOW_SIZE = (660, 500)
BLOCK_SIZE = 20

AGENT_DIRECTION = {"up":[1,0,0,0], "down":[0,1,0,0], "left":[0,0,1,0], "right":[0,0,0,1]}

Position = namedtuple('Position', 'x, y')

class GameAI():
    """Game field class"""

    def __init__(self):
        """Initializes the game board"""
        # Pygame
        self.screen = pygame.display.set_mode((WINDOW_SIZE[0], WINDOW_SIZE[1]))
        self.clock = pygame.time.Clock()

        self.direction = "right" # Initial snake direction
        self.score = 0
        self.frame = 0
        self.game_over = 0

        # Grid 
        self.n_blocks_col = WINDOW_SIZE[0] // BLOCK_SIZE
        self.n_blocks_row = WINDOW_SIZE[1] // BLOCK_SIZE
        self.draw_grid()

        # Entities
        self.snake = Snake(self.screen)
        self.apple = None
        self.spawn_apple()

        pygame.display.update()
    
    def get_direction(self):
        return self.direction
    
    def get_snake(self):
        """Returns the snake of the game"""
        return self.snake
    
    def get_opposite(self):
        """Returns the opposite direction value of the current snake's direction
           0 = "down"
           1 = "up"
           2 = "right"
           3 = "left" 
        """
        if self.direction == "up":
            return 1
        elif self.direction == "down":
            return 0
        elif self.direction == "left":
            return 3
        else:
            return 2

    def reset(self):
        """Resets the status of the game"""
        self.direction = "right" # Initial snake direction
        self.score = 0
        self.frame = 0
        self.game_over = 0

        # Grid 
        self.n_blocks_col = WINDOW_SIZE[0] // BLOCK_SIZE
        self.n_blocks_row = WINDOW_SIZE[1] // BLOCK_SIZE
        self.draw_grid()

        # Entities
        self.snake = Snake(self.screen)
        self.apple = None
        self.spawn_apple()

        pygame.display.update()

    def spawn_apple(self):
        """Spawns apple in the game in a position that isn't a barrier or the snake"""
        while True:
            # Make sure apple cannot spawn in barrier
            x = random.randrange(BLOCK_SIZE, (WINDOW_SIZE[0]-BLOCK_SIZE), BLOCK_SIZE)
            y = random.randrange(BLOCK_SIZE,(WINDOW_SIZE[1]-BLOCK_SIZE), BLOCK_SIZE)

            # Check apple position is not snake
            if self.not_snake(Position(x, y)):
                self.apple = Position(x, y)
                self.draw_apple(self.apple)
                break

    def draw_apple(self, position):
        """Draws the apple in game
        
        Parameters
            position: (x, y) position to spawn the apple"""
        x = position.x
        y = position.y 
        pygame.draw.rect(self.screen, RED, ((x, y, BLOCK_SIZE, BLOCK_SIZE)), 2)
    
    def not_snake(self, position):
        """Returns whether a certain position is a snake or not
        
        Parameters
            position: (x, y) position to check
        
        Returns:
            bool: True of the given position is not a snake"""
        if position in self.snake.get_position():
            return False
        
        return True

    def draw_grid(self):
        """Draws the playing field into grid format
        
        Parameters
            screen: Game window
        """

        self.screen.fill(BLACK)

        for x in range(self.n_blocks_col):
            for y in range(self.n_blocks_row):
                rect = pygame.Rect(x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)

                # Create padding/barriers on the outer edge
                if x == 0 or y == 0 or x == self.n_blocks_col-1 or y == self.n_blocks_row-1:
                    pygame.draw.rect(self.screen, WHITE, ((x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)), 2)
                else:
                    pygame.draw.rect(self.screen, GREY, rect, 1)
    
    def move_snake(self, ate_apple):
        """Moves the snake towards the current direction
        
        Parameters
            ate_apple: Boolean value if an apple was eaten"""
    
        snake_head = self.snake.get_position()[0]
        x = snake_head.x
        y = snake_head.y

        if self.direction == "up":
            y -= BLOCK_SIZE
        elif self.direction == "down":
            y += BLOCK_SIZE
        elif self.direction == "left":
            x -= BLOCK_SIZE
        elif self.direction == "right":
            x += BLOCK_SIZE
        
        self.snake.move(Position(x, y), ate_apple)
    
    def step(self, action=None):
        """Steps the game by 1 tick
        
        Parameter:
            action: Action for which to play the game
        """

        reward = 0
        self.frame += 1 

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        if np.array_equal(action, AGENT_DIRECTION["up"]) and self.snake.possible_move("up"):
            self.direction = "up"
        elif np.array_equal(action, AGENT_DIRECTION["down"]) and self.snake.possible_move("down"):
            self.direction = "down"
        elif np.array_equal(action, AGENT_DIRECTION["left"]) and self.snake.possible_move("left"):
            self.direction = "left"
        elif np.array_equal(action, AGENT_DIRECTION["right"]) and self.snake.possible_move("right"):
            self.direction = "right"

        if self.snake.snake_died() or self.frame > 200 * len(self.snake.get_position()):
            self.game_over = 1
            reward = -10

            return self.game_over, reward, self.score
        
        if self.snake.get_position()[0] == self.apple:
            self.score += 1
            self.spawn_apple()
            reward = 10
            self.move_snake(True)
        else:
            self.move_snake(False)
            
        
        pygame.display.update()
        self.clock.tick(30)

        return self.game_over, reward, self.score
        
class Snake():
    """Represents the snake that the player will move in the game"""
    
    def __init__(self, screen):
        """Initializes snake at a given position

        Parameters
            screen: Game screen
        """
        self.screen = screen
        self.head = Position(6*BLOCK_SIZE, 10*BLOCK_SIZE)
        self.body = [self.head]
        self.colour = (0, 100, 0) 

        for i in range(4):
            self.body.append(Position(self.head.x-((i+1)*BLOCK_SIZE), self.head.y))
        
        self.draw()
        
    def draw(self):
        """Draws the snake on the game board within its current position"""
        for body in self.body:
            pygame.draw.rect(self.screen, self.colour, pygame.Rect(body.x, body.y, BLOCK_SIZE, BLOCK_SIZE))
    
    def delete_tail(self):
        """Deletes the tail of the snake when it's moved out of it's past position"""
        x = self.body[-1].x
        y = self.body[-1].y

        rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
        pygame.Surface.fill(self.screen, BLACK, rect)
        pygame.draw.rect(self.screen, GREY, rect, 1)

    def get_position(self):
        """Returns the position of the snake in list format"""
        return self.body
    
    def move(self, position, ate_apple):
        """Updates the position of the snake by 1 block
        
        Parameters
            position: new (x, y) head position of snake after moving 1 block
            ate_apple: Whether the snake ate an apple
        """

        # Update head
        self.head = position
        self.body.insert(0, self.head)

        # If an apple isn't eaten delete the tail
        if not ate_apple:
            self.delete_tail()
            self.body.pop(-1)
        
        self.draw()

    def snake_died(self, position: Position = None):
        """Returns True if the snake has died or if a given position is safe for the snake
           1. Snake will die if it collides with itself
           2. Snake will die if it collides with a barrier

           Parameters
            position: The position to check if not given then the snake head will be used
        """

        if position is not None:
            check = position
        else:
            check = self.head

        # Check if snake has eaten itself
        if check in self.body[1:]:
            return True

        # Check if it has hit the barriers
        if check.x < BLOCK_SIZE or check.y < BLOCK_SIZE or check.x >= WINDOW_SIZE[0] - BLOCK_SIZE or check.y >= WINDOW_SIZE[1] - BLOCK_SIZE:
            return True

        return False
    
    def possible_move(self, direction):
        """Checks if a certain movement is possible for this snake. 
           1. If the snake is in an upwards position it cannot move down
           2. If the snake is in a downwards position it cannot move up
           3. If the snake is moving left it cannot move right
           4. If the snake is moving right it cannot move left
           
        Parameters
            direction: Direction to move the snake
            
        Returns
            bool: Whether moving in the direction in the current position is possible
        """

        can_move = True
        x = self.body[0].x
        y = self.body[0].y
        
        if direction == "up":
            y -= BLOCK_SIZE
        elif direction == "down":
            y += BLOCK_SIZE
        elif direction == "left":
            x -= BLOCK_SIZE 
        elif direction == "right":
            x += BLOCK_SIZE
        
        if Position(x, y) == self.body[1]:
            can_move = False
        
        return can_move
    
    def impossible_move_for_ai(self, direction):
        """Returns the impossible move of a direction
           0 = UP
           1 = DOWN
           2 = LEFT
           3 = RIGHT
        """

        if direction == "up":
            return 1
        elif direction == "down":
            return 0
        elif direction == "left":
            return 3
        else:
            return 2



        
import sys
import pygame
from src.env import SnakeEnv

class SnakeRenderer:
    """PyGame Visualizer for SnakeEnv."""

    # Color definitions
    BLACK = (15, 15, 15)
    WHITE = (255, 255, 255)
    GREEN = (46, 204, 113)
    HEAD_BLUE = (52, 152, 219) 
    RED = (231, 76, 60)
    GRAY = (40, 40, 40)

    def __init__(self, env: SnakeEnv, block_size: int = 25):
        pygame.init()
        pygame.font.init()

        self.env = env
        self.block_size = block_size
        self.width = env.grid_w * block_size
        self.height = env.grid_h * block_size

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18, bold=True)

    def render(self, fps: int = 30) -> None:
        """Renders one frame of the current environment state."""
        # Process PyGame UI events to avoid window freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.BLACK)

       # 1. Draw Snake Body
        for idx, (x, y) in enumerate(self.env.snake):
            rect = pygame.Rect(
                x * self.block_size, y * self.block_size,
                self.block_size, self.block_size
            )
            color = self.HEAD_BLUE if idx == 0 else self.GREEN
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 1)
        # 2. Draw Food
        fx, fy = self.env.food
        food_rect = pygame.Rect(
            fx * self.block_size, fy * self.block_size,
            self.block_size, self.block_size
        )
        pygame.draw.rect(self.screen, self.RED, food_rect)

        # 3. Draw Overlay Score Text
        score_surface = self.font.render(f"Score: {self.env.score}", True, self.WHITE)
        self.screen.blit(score_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self) -> None:
        """Cleanly releases PyGame resources."""
        pygame.quit()
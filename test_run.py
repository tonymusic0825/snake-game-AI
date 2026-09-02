# test_run.py
from src.env import SnakeEnv
from src.renderer import SnakeRenderer
import random

env = SnakeEnv(grid_w=20, grid_h=20)
renderer = SnakeRenderer(env)

obs = env.reset()
terminated = False

while not terminated:
    action = random.choice([0, 1, 2])  # Pick random relative move
    obs, reward, terminated, info = env.step(action)
    renderer.render(fps=15)

print(f"Game Over! Final Score: {info['score']}")
renderer.close()
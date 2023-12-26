# snake-game-AI
Video Link (영상 링크): https://www.youtube.com/watch?v=pu7yxE4YT8I&t=4s&ab_channel=%EC%BD%94%EB%93%9C%EC%A0%80%EC%9E%A5%EC%86%8C-CodeRepository

This is an AI that plays the snake-game

# Game:
This game was created using the Python programming language in addition with the Pygame library.

Develop Process:
1. Create the game's playing field which is just a simple grid.
2. Initialize Snake
3. Initialize Apple
4. Use pygame.time.Clock() and create a clock
5. In addition to number 4 use pygame.display.update() & self.clock.tick() to add the animations
6. Make sure when using number 4 & 5 the snake and apple are updated correctly
7. Add key config (Arrow keys) to the game

Since the AI isn't going to be able to use the keyboard we must change the game so that the AI is able to control the snake in a different way.
8. Used the following "up":[1,0,0,0], "down":[0,1,0,0], "left":[0,0,1,0], "right":[0,0,0,1] in the format of a dictionary where each index of the
   list is able to represent a certain direction
9. Also added a reward variable to pass back to the AI since we will using reinforcement learning

# AI: 
This AI uses reinforcement learning with the addition of bellman's Q equation (Max Q).
Additionally, it has two different types of training methods; short and long memory training.
The short memory training is done every single step/tick.
Meanwhile the long memory training is done after every 1000 steps.
The AI trains using the "Game state" data which includes:
 - The danger locations of the snake's next possible moves
 - The current direction in which the snake is moving
 - The current position of the apple in relative to the snake

Develop Process:
1. Create AI Model, in my case I used a simple 3 layer Neural Network | Input -> Hidden -> ReLu -> Output
2. Create a trainer which trains the Model.
3. Create an Agent which will play the game and manage the whole program as a whole. 

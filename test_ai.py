from model import *
from simple_snake_AI import *
from agent import *

# Create model and load trained model
model = Model()
load = torch.load("best.pth")
model.load_state_dict(load["model_state_dict"])

game = GameAI()
agent = Agent()


def main():
    n_games = 0
    while True:
        state = agent.get_state(game)

        state = torch.tensor(state, dtype=torch.float)
        move = model(state)
        final_move = [0, 0, 0, 0]
        move = torch.argmax(move).item()
        final_move[move] = 1 

        game_over, reward, score = game.step(final_move)

        if game_over:
                n_games += 1
                print("Game", agent.n_games, "Score", score)
                game.reset()


if __name__ == "__main__":
    main()
        


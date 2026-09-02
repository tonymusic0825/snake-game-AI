import argparse
import time
from src.env import SnakeEnv
from src.agent import DQNAgent
from src.renderer import SnakeRenderer


def train(args):
    """Executes the training loop with optional target network updates and visual rendering."""
    env = SnakeEnv(grid_w=args.grid_size, grid_h=args.grid_size)
    agent = DQNAgent(
        input_dim=11,
        action_dim=3,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        epsilon_decay=args.epsilon_decay
    )

    renderer = None
    if args.render:
        renderer = SnakeRenderer(env, block_size=args.block_size)

    record_score = 0
    total_score = 0

    print(f"--- Starting DQN Training ({'Visual Mode' if args.render else 'Headless Mode'}) ---")

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            # 1. Select Action
            action = agent.get_action(state)

            # 2. Step Environment
            next_state, reward, terminated, info = env.step(action)

            # 3. Render if visual flag is set
            if renderer:
                renderer.render(fps=args.fps)

            # 4. Train Short Memory & Remember Transition
            agent.train_step(state, action, reward, next_state, terminated)
            agent.remember(state, action, reward, next_state, terminated)

            state = next_state
            episode_reward += reward

        # 5. End of Episode: Train Long Memory (Experience Replay Buffer)
        loss = agent.train_replay_batch()

        # 6. Synchronize Target Network Periodically
        if episode % args.target_update_freq == 0:
            agent.update_target_network()

        score = info["score"]
        total_score += score

        if score > record_score:
            record_score = score
            agent.save_checkpoint(args.checkpoint)
            print(f" New Record! Score: {record_score} | Saved Checkpoint to '{args.checkpoint}'")

        # Log Progress Every N Episodes
        if episode % args.log_freq == 0:
            avg_score = total_score / episode
            print(
                f"Episode: {episode}/{args.episodes} | "
                f"Score: {score} | "
                f"Record: {record_score} | "
                f"Avg Score: {avg_score:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {loss:.4f}"
            )

    if renderer:
        renderer.close()

    print(f"--- Training Finished! Highest Record: {record_score} ---")


def evaluate(args):
    """Evaluates a trained checkpoint visually using PyGame."""
    env = SnakeEnv(grid_w=args.grid_size, grid_h=args.grid_size)
    agent = DQNAgent(input_dim=11, action_dim=3)
    
    try:
        agent.load_checkpoint(args.checkpoint)
        print(f"Successfully loaded checkpoint from '{args.checkpoint}'")
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{args.checkpoint}' not found.")
        return

    # Force epsilon to minimum during evaluation (no random exploration)
    agent.epsilon = 0.0
    renderer = SnakeRenderer(env, block_size=args.block_size)

    for ep in range(1, args.eval_episodes + 1):
        state = env.reset()
        terminated = False

        while not terminated:
            action = agent.get_action(state)
            state, reward, terminated, info = env.step(action)
            renderer.render(fps=args.fps)

        print(f"Evaluation Game {ep} - Score: {info['score']}")
        time.sleep(1)

    renderer.close()


def main():
    parser = argparse.ArgumentParser(description="Portfolio-Grade Snake Deep Q-Learning (DQN)")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    # 1. Train Parser Options
    train_parser = subparsers.add_parser("train", help="Train the DQN agent")
    train_parser.add_argument("--episodes", type=int, default=1000, help="Total episodes to train")
    train_parser.add_argument("--grid-size", type=int, default=20, help="Grid dimension (N x N)")
    train_parser.add_argument("--block-size", type=int, default=25, help="UI block pixel size")
    train_parser.add_argument("--batch-size", type=int, default=1000, help="Replay buffer batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    train_parser.add_argument("--epsilon-decay", type=float, default=0.995, help="Epsilon decay rate")
    train_parser.add_argument("--target-update-freq", type=int, default=10, help="Target network update episode frequency")
    train_parser.add_argument("--log-freq", type=int, default=10, help="Terminal logging episode frequency")
    train_parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dqn.pth", help="Checkpoint file path")
    train_parser.add_argument("--render", action="store_true", help="Enable PyGame visual rendering during training")
    train_parser.add_argument("--fps", type=int, default=60, help="Rendering frame rate")

    # 2. Evaluate Parser Options
    eval_parser = subparsers.add_parser("eval", help="Visually evaluate a trained checkpoint")
    eval_parser.add_argument("--eval-episodes", type=int, default=5, help="Number of games to evaluate")
    eval_parser.add_argument("--grid-size", type=int, default=20, help="Grid dimension (N x N)")
    eval_parser.add_argument("--block-size", type=int, default=25, help="UI block pixel size")
    eval_parser.add_argument("--checkpoint", type=str, default="checkpoints/best_dqn.pth", help="Checkpoint file path")
    eval_parser.add_argument("--fps", type=int, default=20, help="Evaluation rendering frame rate")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)


if __name__ == "__main__":
    main()
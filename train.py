"""
train.py
 
Trains the DQN agent to play Snake, logging progress and saving:
  - results/training_log.csv       (per-episode score, reward, epsilon, loss)
  - results/training_curves.png    (score / reward / epsilon over training)
  - results/best_model.npz         (weights of the best-performing snapshot)
  - results/final_model.npz        (weights at the end of training)
 
Run with:  python train.py
"""
 
import csv
import os
import time
import numpy as np
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from snake_env import SnakeEnv
from dqn_agent import DQNAgent
 
NUM_EPISODES = 2000
GRID_SIZE = 12
LOG_EVERY = 50
 
 
def moving_average(values, window=50):
    if len(values) < window:
        return np.array(values)
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window
 
 
def train():
    os.makedirs("results", exist_ok=True)
    env = SnakeEnv(grid_size=GRID_SIZE)
    agent = DQNAgent(state_size=env.observation_space_n, action_size=env.action_space_n)
 
    scores, rewards_history, epsilons, losses = [], [], [], []
    best_score = -1
    start_time = time.time()
 
    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_losses = []
 
        while not done:
            action = agent.act(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            state = next_state
            episode_reward += reward
 
        agent.decay_epsilon()
 
        score = info["score"]
        scores.append(score)
        rewards_history.append(episode_reward)
        epsilons.append(agent.epsilon)
        losses.append(np.mean(episode_losses) if episode_losses else 0.0)
 
        if score > best_score:
            best_score = score
            agent.save("results/best_model.npz")
 
        if episode % LOG_EVERY == 0:
            avg_score = np.mean(scores[-LOG_EVERY:])
            elapsed = time.time() - start_time
            print(
                f"Episode {episode:5d}/{NUM_EPISODES} | "
                f"avg score (last {LOG_EVERY}): {avg_score:5.2f} | "
                f"best: {best_score:3d} | epsilon: {agent.epsilon:.3f} | "
                f"elapsed: {elapsed:6.1f}s"
            )
 
    agent.save("results/final_model.npz")
 
    # --- Save raw log ---
    with open("results/training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "reward", "epsilon", "avg_td_loss"])
        for i in range(NUM_EPISODES):
            writer.writerow([i + 1, scores[i], rewards_history[i], epsilons[i], losses[i]])
 
    # --- Plot training curves ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
 
    ax = axes[0, 0]
    ax.plot(scores, alpha=0.3, color="tab:blue", label="Score per episode")
    if len(scores) >= 50:
        ma = moving_average(scores, 50)
        ax.plot(range(50, 50 + len(ma)), ma, color="tab:blue", linewidth=2, label="50-episode moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (food eaten)")
    ax.set_title("Score per Episode")
    ax.legend()
 
    ax = axes[0, 1]
    ax.plot(rewards_history, alpha=0.3, color="tab:green")
    if len(rewards_history) >= 50:
        ma = moving_average(rewards_history, 50)
        ax.plot(range(50, 50 + len(ma)), ma, color="tab:green", linewidth=2, label="50-episode moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Episode Reward")
    ax.legend()
 
    ax = axes[1, 0]
    ax.plot(epsilons, color="tab:orange")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (epsilon) Decay")
 
    ax = axes[1, 1]
    ax.plot(losses, color="tab:red", alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg TD loss")
    ax.set_title("Average TD (Bellman) Loss per Episode")
 
    plt.tight_layout()
    plt.savefig("results/training_curves.png", dpi=150)
    print("\nSaved training curves to results/training_curves.png")
    print(f"Best score achieved during training: {best_score}")
    print(f"Average score, last 100 episodes: {np.mean(scores[-100:]):.2f}")
    print(f"Total training time: {time.time() - start_time:.1f}s")
 
 
if __name__ == "__main__":
    train()
 
"""
play.py

Loads a trained DQN agent and lets it play a few fresh games of Snake with exploration turned off,
so you're seeing exactly what it learned rather than any random moves. It saves:

results/gameplay_frames.png -> a grid of frames from one game
results/gameplay.gif —> an animated playthrough
and then prints the score stats across several evaluation games!

Run the game with: python play.py
"""
 
import os
import numpy as np
import matplotlib
 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
 
from snake_env import SnakeEnv
from dqn_agent import DQNAgent
 
MODEL_PATH = "results/best_model.npz"
GRID_SIZE = 12
NUM_EVAL_GAMES = 20
 
CMAP = ListedColormap(["#1e1e2e", "#8be9fd", "#50fa7b", "#ff5555"])  # empty, body, head, food
 
 
def frame_to_image(grid, scale=24):
    """Render a grid array to an RGB image using the color map, upscaled for clarity."""
    rgba = CMAP(grid / 3.0)
    img = (rgba[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(img).resize(
        (grid.shape[1] * scale, grid.shape[0] * scale), Image.NEAREST
    )
    return im
 
 
def run_episode(env, agent, record=False):
    state = env.reset()
    done = False
    frames = []
    if record:
        frames.append(env.render_grid().copy())
    while not done:
        action = agent.act(state, explore=False)
        state, reward, done, info = env.step(action)
        if record:
            frames.append(env.render_grid().copy())
    return info["score"], frames
 
 
def main():
    os.makedirs("results", exist_ok=True)
    env = SnakeEnv(grid_size=GRID_SIZE)
    agent = DQNAgent(state_size=env.observation_space_n, action_size=env.action_space_n)
    agent.load(MODEL_PATH)
 
    # --- Evaluate over multiple games 
    scores = []
    for _ in range(NUM_EVAL_GAMES):
        score, _ = run_episode(env, agent, record=False)
        scores.append(score)
 
    print(f"Evaluation over {NUM_EVAL_GAMES} games (epsilon=0, greedy policy):")
    print(f"  scores: {scores}")
    print(f"  mean:   {np.mean(scores):.2f}")
    print(f"  max:    {np.max(scores)}")
    print(f"  min:    {np.min(scores)}")
 
    best_score, best_frames = -1, None
    for _ in range(5):
        score, frames = run_episode(env, agent, record=True)
        if score > best_score:
            best_score, best_frames = score, frames
 
    print(f"\nRecorded a demo game with score {best_score} ({len(best_frames)} frames)")
 
    # Save a grid of frames from the best game
    n_show = min(8, len(best_frames))
    idxs = np.linspace(0, len(best_frames) - 1, n_show).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, i in zip(axes.flat, idxs):
        ax.imshow(best_frames[i], cmap=CMAP, vmin=0, vmax=3)
        ax.set_title(f"Step {i}")
        ax.axis("off")
    fig.suptitle(f"Trained DQN Agent Playing Snake (final score: {best_score})", fontsize=14)
    plt.tight_layout()
    plt.savefig("results/gameplay_frames.png", dpi=150)
    print("Saved results/gameplay_frames.png")
 
    # Save an animated GIF of the full game
    images = [frame_to_image(f) for f in best_frames]
    images[0].save(
        "results/gameplay.gif",
        save_all=True,
        append_images=images[1:],
        duration=120,
        loop=0,
    )
    print("Saved results/gameplay.gif")
 
    # Save a simple bar chart of evaluation scores
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, len(scores) + 1), scores, color="tab:blue")
    ax.axhline(np.mean(scores), color="tab:red", linestyle="--", label=f"Mean = {np.mean(scores):.1f}")
    ax.set_xlabel("Evaluation game #")
    ax.set_ylabel("Score (food eaten)")
    ax.set_title(f"Greedy-Policy Evaluation over {NUM_EVAL_GAMES} Games")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/eval_scores.png", dpi=150)
    print("Saved results/eval_scores.png")
 
 
if __name__ == "__main__":
    main()
'''
q_vs_actual.py

Checking for overestimation.
'''

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.mspacman.frame_stack import FrameStack
from src.mspacman.mspacman_dqn import preprocess_observation

### Helpers
def discounted(rewards, gamma, dones=None):
    G, out = 0.0, []
    for i in range(len(rewards) - 1, -1, -1):
        if dones is not None and dones[i]:
            G = 0.0
        G = rewards[i] + gamma * G
        out.append(G)
    return out[::-1]

def _finish_episode(qs, rs, gamma, truncated, tail, pred, actual, lens, ds=None):
    G = discounted(rs, gamma, ds)
    keep = len(qs) - tail if truncated else len(qs)
    keep = max(keep, 0)
    pred.extend(qs[:keep])
    actual.extend(G[:keep])
    lens.append(keep)

### CartPole (gamma = 0.95 using raw rewards)
def collect_cartpole(env, net, gamma=0.95, n_episodes=100, tail=100):
    net.eval()
    pred, actual, lens = [], [], []
    for _ in range(n_episodes):
        s, _ = env.reset()
        qs, rs, over = [], [], False
        while not over:
            with torch.no_grad():
                q = net(torch.as_tensor(s, dtype=torch.float32))
            a = q.argmax().item()
            qs.append(q.max().item())
            s, r, term, trunc, _ = env.step(a)
            rs.append(r)
            over = term or trunc
        _finish_episode(qs, rs, gamma, trunc and not term, tail, pred, actual, lens)
    return np.array(pred), np.array(actual), lens

### MsPacman (gamma = 0.99 with clipped rewards)
def collect_pacman(env, net, device, gamma=0.99, n_episodes=20, tail=300, eps=0.0):
    net.eval()
    stack = FrameStack(4)
    pred, actual, lens = [], [], []
    for _ in range(n_episodes):
        s = stack.reset(preprocess_observation(env.reset()[0])[:, :, 0])
        lives = 3
        qs, rs, ds, over = [], [], [], False
        while not over:
            with torch.no_grad():
                q = net(torch.as_tensor(s, device=device).unsqueeze(0))[0]
            a = env.action_space.sample() if np.random.rand() < eps else q.argmax().item()
            qs.append(q.max().item())
            obs, r, term, trunc, info = env.step(a)
            s = stack.step(preprocess_observation(obs)[:, :, 0])
            rs.append(float(np.sign(r))) # the net was trained on clipped rewards
            ds.append(term or info["lives"] < lives)
            lives = info["lives"]
            over = term or trunc
        _finish_episode(qs, rs, gamma, trunc and not term, tail, pred, actual, lens, ds)
    return np.array(pred), np.array(actual), lens

### Plot
def plot_q_vs_return(pred, actual, title, fname, max_points=20000):
    bias = np.mean(pred - actual)
    corr = np.corrcoef(pred, actual)[0, 1]
    idx = np.random.choice(len(pred), min(max_points, len(pred)), replace=False)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(actual[idx], pred[idx], s=4, alpha=0.25)
    plt.plot([lo, hi], [lo, hi], "r--", label="y = x (perfect)")
    plt.xlabel("Actual discounted return from state")
    plt.ylabel("Predicted max Q(s, a)")
    plt.title(f"{title}\nmean bias = {bias:+.2f}, corr = {corr:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"{title}: bias (Q - G) = {bias:+.3f}, corr = {corr:.3f}, n = {len(pred)}")

### Animation
def animate_q_vs_return(pred, actual, lens, fname, max_episodes=30, fps=2, title=""):
    lens = [l for l in lens if l > 0][:max_episodes]
    starts = np.cumsum([0] + lens)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    def draw(k):
        ax1.clear()
        ax2.clear()
        a, b = starts[k], starts[k + 1]
        # cumulative scatter (past episodes faint, current episode highlighted)
        ax1.scatter(actual[:a], pred[:a], s=4, alpha=0.2, color="gray")
        ax1.scatter(actual[a:b], pred[a:b], s=6, alpha=0.8, color="tab:blue")
        ax1.plot([lo, hi], [lo, hi], "r--")
        bias = np.mean(pred[:b] - actual[:b])
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_xlabel("Actual discounted return")
        ax1.set_ylabel("Predicted max Q")
        ax1.set_title(f"{title} episodes 1-{k+1}, running bias = {bias:+.2f}")
        # timeline of the current episode
        ax2.plot(pred[a:b], label="Predicted max Q")
        ax2.plot(actual[a:b], label="Actual return")
        ax2.set_xlabel("Timestep")
        ax2.set_title(f"Episode {k+1} ({b-a} steps)")
        ax2.legend()

    anim = FuncAnimation(fig, draw, frames=len(lens), interval=1000 / fps)
    anim.save(fname, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def plot_prediction_error(pred, actual, lens, episode=-1, fname="prediction_error.png"):
    # Locate the selected episode
    starts = np.cumsum([0] + lens)
    episode = episode if episode >= 0 else len(lens) + episode

    if episode < 0 or episode >= len(lens):
        raise ValueError("Invalid episode index")

    a, b = starts[episode], starts[episode + 1]

    p = pred[a:b]
    g = actual[a:b]

    error = p - g
    timesteps = np.arange(len(error))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(timesteps, error, label="Prediction error")
    ax.axhline(0, color="red", linestyle="--", label="Perfect prediction")

    # Mean error
    mean_error = np.mean(error)
    ax.axhline(mean_error, color="green", linestyle=":",
               label=f"Mean error = {mean_error:+.2f}")

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Q(s, a*) - G")
    ax.set_title(f"Prediction Error - Episode {episode + 1}")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.show()

    print(f"Episode {episode + 1}")
    print(f"Mean error: {np.mean(error):+.3f}")
    print(f"MAE: {np.mean(np.abs(error)):.3f}")
    print(f"RMSE: {np.sqrt(np.mean(error**2)):.3f}")

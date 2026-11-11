'''
mspacman_dqn.py

DQN solution to MsPacman-v5.
'''

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py

from pathlib import Path
import logging
from datetime import datetime

from .model import PacmanDQN
from .replay_buffer import ReplayBuffer
from .frame_stack import FrameStack

# ------ LOGGING ------ #

# Directory containing this script
BASE_DIR = Path(__file__).resolve().parent

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = OUTPUT_DIR / "logs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Timestamped log file
log_file = LOG_DIR / f"dqn_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
# ----------------- #

# GPU and env set up
gym.register_envs(ale_py)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Preprocessing
mspacman_color = 210 + 164 + 74
def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)

### Helpers
def to_t(state): # (4, 88,80) int8 --> (1,4,88,80) tensor
    return torch.as_tensor(state, device=device).unsqueeze(0)

def select_action(env, net, s, eps):
    with torch.no_grad():
        q = net(to_t(s))[0] # (9,)
    a = env.action_space.sample() if random.random() < eps else q.argmax().item()
    return a, q.max().item()

def update(online, target, optimizer, buffer, gamma, batch_size):
    s, a, s2, r, d = buffer.sample(batch_size)
    s  = torch.as_tensor(s,device=device)  # (B,4,88,80)
    s2 = torch.as_tensor(s2, device=device)
    a  = torch.as_tensor(a, device=device)
    r  = torch.as_tensor(r, device=device)
    d  = torch.as_tensor(d, device=device)

    with torch.no_grad():
        next_a = online(s2).argmax(1, keepdim=True)
        next_q = target(s2).gather(1, next_a).squeeze(1)
        y = r + gamma * (1 - d) * next_q

    q = online(s).gather(1, a.unsqueeze(1)).squeeze(1)
    loss = F.smooth_l1_loss(q, y) # huber
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), 10.0) # Clipping our gradient to max norm of 10
    optimizer.step()
    return loss.item()

### Training
def train(env, online, target, optimizer, buffer, n_episodes=7_000, gamma=0.99, batch_size=32, warmup=10_000, train_every=4, target_every=10_000, 
          eps_start=1.0, eps_end=0.1, eps_decay=250_000, save_path="pacman_dqn_run_1.pt"):
    episodic_rewards, episodic_max_q = [], []
    losses = []
    step = 0

    stack = FrameStack(4)
    
    for ep in range(n_episodes):
        obs0, info0 = env.reset()
        s = stack.reset(preprocess_observation(obs0)[:, :, 0])   # (4, 88,80)
        lives = info0["lives"]
        ep_reward, over = 0.0, False
        ep_qs = []
        while not over:
            eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay)
            a, max_q = select_action(env, online, s, eps)
            # if start_q is None:
            #     start_q = max_q
            ep_qs.append(max_q)

            obs2, r, term, trunc, info = env.step(a)
            s2 = stack.step(preprocess_observation(obs2)[:, :, 0])
            over = term or trunc
            ep_reward += r

            # Track when you lose a life
            life_lost = info["lives"] < lives
            lives = info["lives"]
            done_flag = float(term or life_lost)
            
            buffer.push(s, a, s2, np.sign(r), float(done_flag)) # store the clipped reward
            s = s2
            step += 1

            if len(buffer) >= warmup and step % train_every == 0: # Fill up the buffer with at least 10,000 tuples
                loss = update(online, target, optimizer, buffer, gamma, batch_size)
                losses.append(loss)
            
            if step % target_every == 0: # Every multiple of 10,000 update the target network
                target.load_state_dict(online.state_dict())
                target.eval()
            
            # Periodic step logging
            if step % 1000 == 0:
                avg_loss = np.mean(losses) if losses else 0
                avg_reward = np.mean(episodic_rewards[-10:]) if episodic_rewards else 0
                logger.info(f"Step: {step} | Epsilon: {eps:.4f} | Avg Loss: {avg_loss:.5f} | Avg Reward - last 10 episodes: {avg_reward:.2f}")
                losses = []

        ep_max_q = max(ep_qs)
        episodic_rewards.append(ep_reward)
        episodic_max_q.append(ep_max_q)
        # Episode logging
        logger.info(f"Episode: {ep + 1} | Reward: {ep_reward:.2f} | Max Q: {ep_max_q:.4f} | Epsilon: {eps:.4f} | Steps: {step}")

        if (ep + 1) % 50 == 0:
            print(f"ep {ep+1}  avg100 reward {np.mean(episodic_rewards[-100:]):.1f}  eps {eps:.3f}  steps {step}  maxQ {ep_max_q:.2f}")
            torch.save(online.state_dict(), OUTPUT_DIR / save_path)
            np.save(OUTPUT_DIR / "pacman_rewards_run_1.npy", episodic_rewards)
            np.save(OUTPUT_DIR / "pacman_maxq_run_1.npy", episodic_max_q)
            logger.info(f"Checkpoint saved after episode {ep + 1}")

    return episodic_rewards, episodic_max_q

### Main
def main():
    logger.info("Starting MsPacman DQN training")
    logger.info("Environment: ALE/MsPacman-v5")
    logger.info("Episodes: 7,000")
    logger.info("Batch size: 32")
    logger.info("Discount factor: 0.99")
    logger.info("Learning rate: 0.0001")
    logger.info("Replay buffer capacity: 100000")
    logger.info(f"Device: {device}")

    env = gym.make("ALE/MsPacman-v5")

    online = PacmanDQN(env.action_space.n).to(device)
    target = PacmanDQN(env.action_space.n).to(device)

    target.load_state_dict(online.state_dict())
    target.eval()

    for p in target.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
    buffer = ReplayBuffer(100_000)

    try:
        rewards, max_qs = train(env, online, target, optimizer, buffer, n_episodes=7_000)

        # Save final training metrics
        np.savez(OUTPUT_DIR / "pacman_outputs_run_1.npz", rewards=rewards, max_qs=max_qs)

        # Save final model
        torch.save(online.state_dict(), OUTPUT_DIR / "pacman_dqn_final_run_1.pt")

        logger.info("Training completed successfully")
        logger.info(f"Total episodes: {len(rewards)}")

        if rewards:
            logger.info(f"Final 10 episode average reward: {np.mean(rewards[-10:]):.2f}")

    except Exception:
        logger.exception("MsPacman DQN training failed")
        raise

    finally:
        env.close()
        logger.info("Environment closed")

if __name__ == "__main__":
    main()



    
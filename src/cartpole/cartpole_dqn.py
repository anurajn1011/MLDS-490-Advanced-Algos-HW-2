'''
cartpole-dqn.py

Solving Cartpole-v1, as provided by gymnasium, utilizing a DQN
'''

import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.distributions import Categorical
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from .replay_buffer import ReplayBuffer, initialize_buffer
from .model import CartpoleDQN

# ------ LOGGING ------ #
import logging
from datetime import datetime

# Directory containing this script
BASE_DIR = Path(__file__).resolve().parent

# Create output directories
LOG_DIR = BASE_DIR / "output" / "logs"
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
# ---------------------# 

### Epsilon greedy actions
def select_action(env, net, s, eps):
    with torch.no_grad():
        q = net(torch.as_tensor(s, dtype=torch.float32))
    a = env.action_space.sample() if random.random() < eps else q.argmax().item()
    return a, q.max().item()

def dqn_target(online, target, next_states, rewards, dones, gamma, double=True):
    with torch.no_grad():
        if double:
            na = online(next_states).argmax(1, keepdim=True)
            nq = target(next_states).gather(1, na).squeeze(1)
        else:
            nq = target(next_states).max(1).values
        return rewards + gamma * (1 - dones) * nq

### Training 
def training(env, online, target, optimizer, buffer, total_steps, discount_rate=0.95, batch_size=64, eps_start=1, eps_end=0.05, eps_decay=10_000):
    '''
        The training loop.
    '''
    episodic_rewards, episodic_max_q = [], []
    reward, ep_qs = 0, []
    s, _ = env.reset()

    losses = []

    pbar = tqdm(range(total_steps), desc="Training DQN")

    for step in pbar:
        eps = max(eps_end, eps_start - ((eps_start - eps_end) * step /eps_decay))
        a, max_q = select_action(env, online, s, eps)
        ep_qs.append(max_q)
        s2, r, done, truncated, _ = env.step(a)

        # Update the buffer
        buffer.push(s, a, s2, r, done)
        reward += r

        # Restart if game complete
        if done or truncated:
            episodic_rewards.append(reward)
            episodic_max_q.append(max(ep_qs))

            # Logging
            logger.info(
                f"Episode: {len(episodic_rewards)} | "
                f"Reward: {reward:.2f} | "
                f"Max Q: {max(ep_qs):.4f}"
            )
            
            reward, ep_qs = 0, []
            s, _ = env.reset()
        else: 
            s = s2

        # Pull some batched 4-tuples from buffer and ensure dtypes
        states, actions, next_states, rewards, dones = buffer.sample(batch_size)
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        next_states = torch.as_tensor(next_states, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        dones = torch.as_tensor(dones, dtype=torch.float32)

        # Predict Q-Value for the next state given next state
        y = dqn_target(online, target, next_states, rewards, dones, discount_rate)

        # # Predict next actions
        # with torch.no_grad():
        #     next_actions = online(next_states).argmax(1, keepdim=True)
        #     next_q_vals = target(next_states).gather(1, next_actions).squeeze(1)
        #     y = rewards + discount_rate * (1 - dones) * next_q_vals

        # Compute loss of current value estimate vs better estimate
        q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 1000 == 0:
            # update the target
            target.load_state_dict(online.state_dict())
            target.eval()

            # Logging
            avg_loss = np.mean(losses) if losses else 0
            avg_reward = np.mean(episodic_rewards[-10:]) if episodic_rewards else 0
        
            logger.info(
                f"Step: {step}/{total_steps} | "
                f"Epsilon: {eps:.4f} | "
                f"Avg Loss: {avg_loss:.5f} | "
                f"Avg Reward (last 10 episodes): {avg_reward:.2f}"
            )
            losses = []
        
    return episodic_rewards, episodic_max_q

### Exectuion routine
def main():
    '''
    The training loop for the Cartpole-v1 DQN. 
    '''
    # Logging background
    logger.info("Starting CartPole DQN training")
    logger.info(f"Total steps: {100_000}")
    logger.info(f"Batch size: {64}")
    logger.info(f"Discount rate: {0.95}")
    
    # Environment 
    env = gym.make("CartPole-v1")

    # Network initialization
    online_network = CartpoleDQN()
    optimizer = torch.optim.Adam(online_network.parameters(), lr=0.001)
    
    target_network = CartpoleDQN()
    target_network.load_state_dict(online_network.state_dict()) # We copy the online network exactly, fixing it
    target_network.eval() # Eval mode for batchnorm

    # Fill the Replay Buffer
    buffer = ReplayBuffer(1000)
    initialize_buffer(env, buffer, 1000)

    # Train
    try:
        episodic_rewards, episodic_qs = training(env, online_network, target_network, optimizer, buffer, total_steps=100_000, discount_rate=0.95, batch_size=64, eps_start=1, eps_end=0.05, eps_decay=10_000)
    except Exception:
        logger.exception("Training failed")

    # Saving the rewards
    p = BASE_DIR / "output" / "outputs.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, rewards=episodic_rewards, qs=episodic_qs)

    # Saving the model
    torch.save(online_network.state_dict(), BASE_DIR / "output" / "cartpole_dqn.pt")

    # Closing logs
    logger.info("Training completed successfully")
    logger.info(f"Total episodes: {len(episodic_rewards)}")
    logger.info(f"Final 10 episode average reward: {np.mean(episodic_rewards[-10:]):.2f}")

### EXECUTE MAIN
if __name__ == "__main__":
    main()






    
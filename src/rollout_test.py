'''
rollout_test.py

Rollout 500 episodes of the specified game for model to test on. Collect rewards.
'''

import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
import gymnasium as gym
from tqdm import tqdm

from src.mspacman.frame_stack import FrameStack
from src.mspacman.mspacman_dqn import preprocess_observation

def rollout(env, net, n_episodes=500):
    net.eval()
    scores = []
    for _ in range(n_episodes):
        s, _ = env.reset()
        total, over = 0, False
        while not over:
            with torch.no_grad():
                a = net(torch.as_tensor(s, dtype=torch.float32)).argmax().item()
            s, r, term, trunc, _ = env.step(a)
            total += r
            over = term or trunc
        scores.append(total)
    return scores

def rollout_pacman(env, net, device, n_episodes=500, eps=0.0):
    net.eval()
    stack = FrameStack(4)
    scores = []
    for _ in tqdm(range(n_episodes)):
        s = stack.reset(preprocess_observation(env.reset()[0])[:, :, 0])  # (4,88,80)
        total, over = 0.0, False
        while not over:
            if np.random.rand() < eps:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = net(torch.as_tensor(s, device=device).unsqueeze(0)).argmax().item()  # (1,4,88,80)
            obs, r, term, trunc, _ = env.step(a)
            s = stack.step(preprocess_observation(obs)[:, :, 0])
            total += r # raw reward, not clipped
            over = term or trunc
        scores.append(total)
    return scores

# scores = rollout_pacman(env, model)
# print(f"mean = {np.mean(scores):.1f}, std = {np.std(scores):.1f}")

# scores = rollout(env, online_network)
# print(f"mean = {np.mean(scores):.2f}, std = {np.std(scores):.2f}")
'''
replay_buffer.py

Replay buffer for MsPacman-v5
'''

import random
import numpy as np
import gymnasium as gym
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.cap, self.i, self.n = capacity, 0, 0
        self.s  = np.zeros((capacity, 4, 88, 80), np.int8)
        self.a  = np.zeros(capacity, np.int64)
        self.s2 = np.zeros((capacity, 4, 88, 80), np.int8)
        self.r  = np.zeros(capacity, np.float32)
        self.d  = np.zeros(capacity, np.float32)

    def push(self, s, a, s2, r, done):
        self.s[self.i], self.a[self.i] = s, a
        self.s2[self.i], self.r[self.i], self.d[self.i] = s2, r, done
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, B):
        idx = np.random.randint(0, self.n, B)
        return self.s[idx], self.a[idx], self.s2[idx], self.r[idx], self.d[idx]

    def __len__(self):
        return self.n

# def initialize_buffer(env, buffer, length):
#     '''
#     Given an environment and a buffer object, add in preliminary objects to train on.
#     '''
#     s1, _ = env.reset()
#     for _ in range(length):
#         a = env.action_space.sample()
#         s2, r, done, truncated, _ = env.step(a)
#         buffer.push(s1, a, s2, r, done)
#         s1 = s2 if not (done or truncated) else env.reset()[0]
#     return buffer

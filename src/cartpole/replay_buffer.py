'''
replay_buffer_cartpole.py

Constructing a replay buffer for Cartpole-v1. 
'''

import random
import numpy as np
import gymnasium as gym
from collections import deque

class ReplayBuffer:
    def __init__(self, length):
        self.cap = length
        self.n = 0
        self.buffer = deque(maxlen=length)

    def push(self, s1, a, s2, r, done):
        self.buffer.append((s1, a, s2, r, done))
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s1, a, s2, r, done = map(np.array, zip(*batch))
        return s1, a, s2, r, done

    def __len__(self):
        return self.n

def initialize_buffer(env, buffer, length):
    '''
    Given an environment and a buffer object, add in preliminary objects to train on.
    '''
    s1, _ = env.reset()
    for _ in range(length):
        a = env.action_space.sample()
        s2, r, done, truncated, _ = env.step(a)
        buffer.push(s1, a, s2, r, done)
        s1 = s2 if not (done or truncated) else env.reset()[0]
    return buffer
    
        
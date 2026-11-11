'''
model.py

The model for MsPacman-v5 from gymnasium.
'''

import numpy as np
import torch
import torch.nn as nn

### Neural Network
class PacmanDQN(nn.Module):
    def __init__(self, n_actions=9):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), 
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 6, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x): # x: (B, 4, 88, 80), int8 range -128..127
        x = x.float() / 128.0 # scale to about [-1, 1]
        return self.fc(self.conv(x).flatten(1))




        
'''
model.py

Contains the pytorch model and associated modules necessary for training the DQN.
'''

import numpy as np
import torch
import torch.nn as nn

### Neural Network of CartPole
class CartpoleDQN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)
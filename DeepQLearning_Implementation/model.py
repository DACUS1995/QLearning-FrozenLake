import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ACTION_SPACE_SIZE = 4
STATE_SPACE_SIZE = 16

class FrozenLakeModel(nn.Module):
    def __init__(self, state_space_size=STATE_SPACE_SIZE, action_space_size=ACTION_SPACE_SIZE):
        super().__init__()
        # The state space and the actions space are rather small so only a hidden layer should be enough
        self.fc1 = nn.Linear(1, 512)
        self.fc2 = nn.Linear(512, action_space_size)

    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out)
        return self.fc2(out)
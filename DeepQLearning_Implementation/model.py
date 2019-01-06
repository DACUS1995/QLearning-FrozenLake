import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ACTION_SPACE_SIZE = 4
STATE_SPACE_SIZE = 16

class FrozenLakeModel(nn.Module):
    def __init__(self, state_space_size, action_space_size):
        super().__init__()
        self.fc1 = nn.Linear(STATE_SPACE_SIZE, 512)
        self.fc2 = nn.Linear(512, ACTION_SPACE_SIZE)

    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out)
        return self.fc2(out)
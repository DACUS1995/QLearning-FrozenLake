import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config

class FrozenLakeModel(nn.Module):
    def __init__(self, state_space_size=Config.STATE_SPACE_SIZE, action_space_size=Config.ACTION_SPACE_SIZE):
        super().__init__()

        # The state space and the actions space are rather small so only a hidden layer should be enough
        self.fc1 = nn.Linear(1, 512)
        self.fc2 = nn.Linear(512, action_space_size)

    def forward(self, x):
        out = self.fc1(x)
        out = F.leaky_relu(out)
        return F.softmax(self.fc2(out))
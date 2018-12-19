import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FrozenLakeModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass
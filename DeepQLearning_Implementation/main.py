import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import gym

from model import FrozenLakeModel
from memory import MemoryReplay

ENVIRONMENT = "FrozenLake-v0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'[Device selected {device}]')

def train(
		learning_rate,
		num_episodes,
		memory_size
	):
	env = gym.make(ENVIRONMENT)
	replay_memory = MemoryReplay(memory_size)


def main(args):
	train(
		learning_rate=args.learning_rate,
		num_episodes=args.episodes,
		memory_size=args.replay_memory_size
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser("")
	parser.add_argument("-l", "--learning-rate", type=float, default=0.05, help="learning rate")
	parser.add_argument("-ep", "--episodes", type=int, default=1000, help="number of episodes")
	parser.add_argument("-mem", "--replay-memory-size", type=int, default=1000, help="replay memory size")
	args = parser.parse_args()
	main(args)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

from model import FrozenLakeModel




def main(args):
	print("Learning rate ", args.learning_rate)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("")
	parser.add_argument("-l", "--learning-rate", type=int, default=0.05, help="learning rate")
	args = parser.parse_args()
	main(args)
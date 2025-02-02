import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import argparse
import gym
import random
from matplotlib import pyplot as plt

from model import FrozenLakeModel
from memory import MemoryReplay
from config import Config


max_exploration_rate = 1
min_exploration_rate = 0.001
exploration_decay_rate = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'[Device selected {device}]')

def train(
		learning_rate,
		num_episodes,
		memory_size,
		max_steps_per_ep,
		batch_size
	):
	env = gym.make(Config.ENVIRONMENT)
	exploration_rate = 0.1
	replay_memory = MemoryReplay(memory_size)

	target_network = FrozenLakeModel()
	policy_network = FrozenLakeModel()
	target_network.to(device)
	policy_network.to(device)

	policy_network.train()

	optimizer = torch.optim.Adam(policy_network.parameters())
	criterion = nn.MSELoss()
	count = 1000

	loss_all = []
	reward_all = []

	running_loss = 0

	print("---> Started training")

	#Training loop
	for ep in range(num_episodes):
		rewards_ep = 0

		if ep % 50 == 0:
			print(f'---> Running episode [{ep}/{num_episodes}]')

		state = torch.tensor([env.reset()], device=device, dtype=torch.float)

		for step in range(max_steps_per_ep):
			random_threshold = random.uniform(0, 1)

			action = None
			if random_threshold > exploration_rate:
				with torch.no_grad():
					action = np.argmax(policy_network(state).cpu().numpy())
			else:
				action = env.action_space.sample()


			new_state, reward, done, _ = env.step(action)
			rewards_ep += reward

			reward = torch.tensor(reward, device=device)
			new_state = torch.tensor([new_state], device=device, dtype=torch.float)
			action = torch.tensor(action, device=device, dtype=torch.float)
			replay_memory.add_sample((state, action, reward, new_state))

			state = new_state

			# Optimize the model
			if len(replay_memory) >= batch_size:
				sample_batch = replay_memory.get_sample(batch_size)
				(states, actions, rewards, new_states) = zip(*sample_batch)

				states = torch.tensor(states, device=device, dtype=torch.float)
				actions = torch.tensor(actions, device=device, dtype=torch.long)
				rewards = torch.tensor(rewards, device=device, dtype=torch.float)
				new_states = torch.tensor(new_states, device=device, dtype=torch.float)

				# actions = actions.repeat([batch_size, 1])
				actions = actions.unsqueeze(1)
				states = states.reshape((-1, 1))
				new_states = new_states.reshape((-1, 1))

				policy_next_action = policy_network(states).gather(1, actions).squeeze()
				target_next_next_action = target_network(new_states).detach()

				target_expected_next_action = rewards + Config.DISCOUNT * target_next_next_action.max(1)[0]

				loss = criterion(policy_next_action, target_expected_next_action)
				running_loss += loss.item()


				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			if done:
				break

		# print(running_loss / max_steps_per_ep)

		loss_all.append(running_loss)
		reward_all.append(rewards_ep)

		running_loss = 0
		rewards_ep = 0

		# Update the target network
		if ep % 5 == 0:
			target_network.load_state_dict(policy_network.state_dict())

		# Training metrics
		# if (ep + 1) % 1000 == 0:
		# 	print(f'{count} : {rewards_ep / 1000}')
		# 	rewards_ep = 0

		# Lower the exploration rate
		# exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * ep)

	plt.subplot(2, 1, 1)
	plt.ylabel("Loss")
	plt.plot(loss_all)

	plt.subplot(2, 1, 2)
	plt.plot(reward_all)
	plt.xlabel("Episodes")
	plt.ylabel("Reward")
	plt.show()

def main(args):
	train(
		learning_rate=args.learning_rate,
		num_episodes=args.episodes,
		memory_size=args.replay_memory_size,
		max_steps_per_ep=args.max_steps_per_ep,
		batch_size=args.batch_size
	)

if __name__ == "__main__":
	parser = argparse.ArgumentParser("")
	parser.add_argument("-l", "--learning-rate", type=float, default=0.05, help="learning rate")
	parser.add_argument("-ep", "--episodes", type=int, default=5000, help="number of episodes")
	parser.add_argument("-mem", "--replay-memory-size", type=int, default=5, help="replay memory size")
	parser.add_argument("--max-steps-per-ep", type=int, default=1000, help="max steps for episode")
	parser.add_argument("--batch-size", type=int, default=5, help="model optimization batch sizes")
	args = parser.parse_args()
	main(args)
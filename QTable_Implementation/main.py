import numpy as np
import gym
import sys
import random
import json


NUMBER_OF_EPISODES = len(sys.argv) > 1 and int(sys.argv[1]) or 10000
ENVIRONMENT = "FrozenLake-v0"
SAVE_Q_TABLE = True

max_steps_for_episode = 100
exploration_rate = 1
discount_rate = 0.99
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001
learning_rate = 0.1

q_table = np.zeros([16, 4])
all_rewards = []

def main():
	global exploration_rate
	print("---> Creating the environment ", ENVIRONMENT)
	env = gym.make(ENVIRONMENT)

	for episode in range(NUMBER_OF_EPISODES):
		print("----> Running episode ", episode)
		
		state = env.reset()
		done = False
		rewards = 0

		for step in range(max_steps_for_episode):
			random_threshold = random.uniform(0, 1)

			if random_threshold > exploration_rate:
				action = np.argmax(q_table[state, :])
			else:
				action = env.action_space.sample()

			assert action >= 0 and action < 5, "action is not in available action space"

			new_state, reward, done, info = env.step(action)

			# Update the Q Table
			q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.amax(q_table[new_state, :]))

			state = new_state
			rewards += reward 

			if done == True:
				break

		# Lower the exploration rate
		exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
		all_rewards.append(rewards)

	splited_rewards = np.split(np.array(all_rewards), NUMBER_OF_EPISODES / 1000)
	count = 1000

	print("\n:: Results ::\n")

	for rewards in splited_rewards:
		print(count, ": ", str(sum(rewards / 1000)))
		count += 1000

	if SAVE_Q_TABLE:
		save_table(q_table)

def save_table(q_table):
	with open("QTableSave.txt", encoding="utf-8", mode="w") as file:
		file.write(json.dumps(q_table.tolist()))

def load_table(file_name):
	raise Exception("Must be implemented")

if __name__ == "__main__":
	main()

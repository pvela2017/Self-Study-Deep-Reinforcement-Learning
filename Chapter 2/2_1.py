# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
import random
import matplotlib.pyplot as plt

n = 10
probs = np.random.rand(n) #A
eps = 0.1

"""
def get_best_action(actions):
	best_action = 0
	max_action_value = 0
	for i in range(len(actions)): 
		cur_action_value = get_action_value(actions[i]) 
		if cur_action_value > max_action_value:
			best_action = i
			max_action_value = cur_action_value
	return best_action
"""

def get_reward(prob, n=10):
    reward = 0;
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward

reward_test = [get_reward(0.7) for _ in range(2000)]
print(np.mean(reward_test))

plt.figure(figsize=(9,5))
plt.xlabel("Reward",fontsize=22)
plt.ylabel("# Observations",fontsize=22)
plt.hist(reward_test,bins=9)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List
import numpy as np


def str_to_averages(string: str) -> List:
    rewards = list(map(float, string.split(',')))

    total_rewards = [sum(rewards[:i + 1]) for i in range(len(rewards))]
    average_rewards = [total_rewards[i] / (i + 1) for i in range(len(rewards))]
    return average_rewards


def plot_results(path: str, label: str) -> None:
    with open(path) as f:
        average_rewards = str_to_averages(next(f))
        n = len(average_rewards)
        plt.plot(range(n), average_rewards, label=label)


def get_average_reward(path: str) -> int:
    with open(path) as f:
        rewards = list(map(float, next(f).split(',')))
        return np.mean(rewards)

def get_best_mean_rewards(path: str) -> int:
    with open(path) as f:
        rewards = list(map(float, next(f).split(',')))
        means = [np.mean(rewards[i:i+100]) for i in range(len(rewards) - 101)]
        return max(means)

# random_reward = get_average_reward('./data/raw-play/random/random-1.000000.txt')

# # Plot average rewards as training progresses
# plt.figure()
# plt.title('Average Rewards by Featurization Technique')
# plt.xlabel('Number of episodes played')
# plt.ylabel('Average Reward')
# plot_results('./data/raw-play/random/random-1.000000.txt', 'random baseline')
# plot_results('./data/raw-play/downsample/downsample-0.400000.txt', 'human, downsample, ols')
# plot_results('./data/raw-play/pca/pca-8.000000.txt', 'human, pca, ols')
# plot_results('./data/atari-play/pca-ols/pca-8-1e1.txt', 'atari, pca, ols')
# plt.legend()
# plt.savefig('./data/raw-play/averages.png')
#
#
# # Plot average rewards per downsample rate
# plt.figure()
# plt.title('Average Rewards with Downsample Rate')
# plt.xlabel('Downsample Rate (scale of original image)')
# plt.ylabel('Average Reward (500 episodes)')
# ds = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
# average_rewards = [get_average_reward('./data/raw-play/downsample/downsample-%f.txt' % d) for d in ds]
# plt.plot(ds, average_rewards, label='ols')
# plt.legend()
# plt.savefig('./data/raw-play/downsample.png')
#
#
# # Plot average rewards per PCA dimensions
# plt.figure()
# plt.title('Average Rewards with PCA Dimensions')
# plt.xlabel('Dimensionality of subspace')
# plt.ylabel('Average Reward (500 episodes)')
# ds = list(range(5, 15)) + [20, 25, 35]
# average_rewards = [get_average_reward('./data/raw-play/pca/pca-%d.000000.txt' % d) for d in ds]
# plt.plot(ds, average_rewards, label='ols')
# plt.plot(ds, [random_reward] * len(average_rewards), label='random')
# plt.legend()
# plt.savefig('./data/raw-play/pca.png')


# # Plot average rewards per number training episodes used
# plt.figure()
# plt.title('Average Rewards per Number Training Episodes')
# plt.xlabel('Number Training Episodes')
# plt.ylabel('Average Reward (500 episodes)')
# ts = [1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# average_rewards = [get_average_reward('./data/%d-episodes-play/pca/pca-8.000000.txt' % t) for t in ts]
# plt.plot(ts, average_rewards, label='ols')
# plt.plot(ts, [random_reward] * len(average_rewards), label='random')
# plt.legend()
# plt.savefig('./data/raw-play/training.png')

# Plot average rewards per number training episodes used
# plt.figure()
# plt.title('Average Rewards per Number Training Episodes')
# plt.xlabel('Number Training Episodes')
# plt.ylabel('Average Reward (1000 episodes)')
# ts = [200, 1000, 2000, 4000, 6000, 8000, 10000, 20000]
# average_rewards = [get_average_reward('./data/fc5-play/fc5-play-%d.txt' % t) for t in ts]
# plt.plot(ts, average_rewards, label='ols fc5 avg')
# best_mean_rewards = [get_best_mean_rewards('./data/fc5-play/fc5-play-%d.txt' % t) for t in ts]
# plt.plot(ts, best_mean_rewards, label='ols fc5 bmr')
#
# atari_average_reward = get_average_reward('./data/fc5-play/fc5-play--1.txt')
# atari_best_mean_reward = get_best_mean_rewards('./data/fc5-play/fc5-play--1.txt')
# plt.plot(ts, [atari_average_reward] * len(average_rewards), label='atari, avg')
# plt.plot(ts, [atari_best_mean_reward] * len(best_mean_rewards), label='atari, bmr')
# plt.legend()
# plt.savefig('./data/graphs/training.png')


plt.figure()
plt.title('Average Rewards per Number Training Episodes, SpaceInvaders')
plt.xlabel('Number Training Episodes')
plt.ylabel('Average Reward (1000 episodes)')
ts = [200, 2000, 3000, 4000, 5000]
average_rewards = [get_average_reward('./data/results/results-%d.txt' % t) for t in ts]
plt.plot(ts, average_rewards, label='ols fc5 avg')
best_mean_rewards = [get_best_mean_rewards('./data/results/results-%d.txt' % t) for t in ts]
plt.plot(ts, best_mean_rewards, label='ols fc5 bmr')

# atari_average_reward = get_average_reward('./data/fc5-play/fc5-play--1.txt')
# atari_best_mean_reward = get_best_mean_rewards('./data/fc5-play/fc5-play--1.txt')
# plt.plot(ts, [atari_average_reward] * len(average_rewards), label='atari, avg')
# plt.plot(ts, [atari_best_mean_reward] * len(best_mean_rewards), label='atari, bmr')
plt.legend()
plt.savefig('./data/graphs/training-spaceinvaders.png')

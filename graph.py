import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List

plt.figure()
plt.title('Average Rewards')

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

plot_results('./data/raw-play/downsample-0.100000.txt', 'human, downsample')
plot_results('./data/raw-play/random-0.100000.txt', 'human, random')

plt.legend()
plt.savefig('./data/raw-play/averages.png')

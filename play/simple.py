
from solve.interface import SolveInterface
from featurize.interface import FeaturizeInterface
from gym import wrappers

import csv
import glob
import os.path
import numpy as np
import time


class SimplePlay:

    def __init__(
            self,
            env,
            solver: SolveInterface,
            featurizer: FeaturizeInterface,
            record: bool=False,
            params=None):
        self.time_id = str(time.time())[-5:]
        self.env = wrappers.Monitor(
            env, os.path.join(solver.play_dir, self.time_id),
            video_callable=lambda _: record)
        self.solver = solver
        self.parameters = None
        self.featurizer = featurizer
        self.set_parameters(params)


    def set_parameters(self, params):
        """Fetch all parameters.

        If parameters are not set, extract parameters from filenames. If so,
        use those.
        """
        if params is None:
            source_path = os.path.join(self.solver.solve_dir, '*.npz')
            paths = glob.iglob(source_path)
            params = ['.'.join(os.path.basename(path).split('.')[:-1])
                          for path in paths]
            if not params:
                raise UserWarning('No solved models found. Did you forget to run `solve`?')
        self.parameters = params

    def run(self, n_episodes: int=1):
        """Run game for the provided number of episodes."""
        average_rewards, total_rewards = [], []
        for param in self.parameters:
            feature_model = self.featurizer.load_model(param)
            solve_model = self.solver.load_model(param)
            episode_rewards = []
            total_rewards.append((param, episode_rewards))

            best_mean_reward = 0
            for i in range(int(n_episodes)):
                observation = self.env.reset()
                while True:
                    obs = observation.reshape((1, *observation.shape))
                    featurized = self.featurizer.phi(obs, feature_model)
                    action = self.solver.predict(featurized, solve_model)
                    observation, reward, done, info = self.env.step(action)
                    if done:
                        self.env.reset()
                        episode_rewards.append(self.env.get_episode_rewards()[-1])
                        if (i % 100 == 0 or i + 1 == int(n_episodes)) and i > 0:
                            best_mean_reward = max(
                                best_mean_reward,
                                np.mean(self.env.get_episode_rewards()[-100:]))
                        break
            print(' * (%s) Best Mean Reward: %f' % (param, best_mean_reward))
            average_rewards.append(best_mean_reward)
        return total_rewards, average_rewards

    def save_results(self, average_rewards):
        with open(os.path.join(self.solver.play_dir, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for param, reward in zip(self.parameters, average_rewards):
                writer.writerow([param, reward])

    def save_rewards(self, total_rewards):
        for k, rewards in total_rewards:
            path = os.path.join(self.solver.play_dir, '%s-%f.txt' % (
                self.featurizer.technique, float(k)))
            with open(path, 'w') as f:
                f.write(','.join(map(str, rewards)))
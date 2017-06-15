
from solve.interface import SolveInterface
from featurize.interface import FeaturizeInterface
from gym import wrappers
from path import Path

import csv
import glob
import os.path
import numpy as np
import time


class PlayInterface:

    def __init__(
            self,
            path: Path,
            env,
            solver: SolveInterface,
            featurizer: FeaturizeInterface,
            record: bool=False,
            params=None):
        self.time_id = str(time.time())[-5:]
        self.env = wrappers.Monitor(
            env, os.path.join(path.play_dir, self.time_id),
            video_callable=lambda _: record)
        self.solver = solver
        self.parameters = None
        self.featurizer = featurizer
        self.path = path
        self.set_parameters(params)


    def set_parameters(self, params):
        """Fetch all parameters.

        If parameters are not set, extract parameters from filenames. If so,
        use those.
        """
        if params is None:
            source_path = os.path.join(self.path.solve_dir, '*.npz')
            paths = glob.iglob(source_path)
            params = ['.'.join(os.path.basename(path).split('.')[:-1])
                          for path in paths]
            if not params:
                raise UserWarning('No solved models found. Did you forget to run `solve`?')
        self.parameters = params

    def init_record(self, n_episodes: int) -> dict:
        """Record object can be any structure."""
        return {
            'average_rewards': [],
            'total_rewards': [],
            'n_episodes': n_episodes
        }

    def init_record_for_param(self, record: dict, param: str) -> dict:
        """Initialize record for given parameter."""
        record['episode_rewards'] = episode_rewards = []
        record['total_rewards'].append((param, episode_rewards))
        record['best_mean_reward'] = 0

    def update_record_on_action(
            self,
            action: int,
            observation: np.array,
            reward: int,
            record: dict,
            done: bool):
        pass

    def update_record_on_episode_finish(self, i: int, record: dict):
        """Update record whenever an episode finishes.

        :param i: Index of episode being played
        :param record: State, logs, and general information
        """
        record['episode_reward'] = self.env.get_episode_rewards()[-1]
        record['episode_rewards'].append(record['episode_reward'])
        if (i % 100 == 0 or i + 1 == record['n_episodes']) and i > 0:
            record['best_mean_reward'] = max(
                record['best_mean_reward'],
                np.mean(self.env.get_episode_rewards()[-100:]))

    def print_record_on_param_finish(self, param: str, record: dict):
        """Optionally, print results."""
        print(' * (%s) Best Mean Reward: %f' % (
            param, record['best_mean_reward']))
        record['average_rewards'].append(record['best_mean_reward'])

    def run(self, n_episodes: int=1):
        """Run game for the provided number of episodes."""
        record = self.init_record(n_episodes)
        for param in self.parameters:
            feature_model = self.featurizer.load_model(param)
            solve_model = self.solver.load_model(param)
            self.init_record_for_param(record, param)

            for i in range(n_episodes):
                observation = self.env.reset()
                while True:
                    obs = observation.reshape((1, *observation.shape))
                    featurized = self.featurizer.phi(obs, feature_model)
                    action = self.solver.predict(featurized, solve_model)
                    observation, reward, done, info = self.env.step(action)
                    self.update_record_on_action(
                        action, observation, reward, record, done)
                    if done:
                        self.env.reset()
                        self.update_record_on_episode_finish(i, record)
                        break
            self.print_record_on_param_finish(param, record)
        return record

    def log(self, record: dict):
        """Log information held in record."""
        self.save_results(record)
        self.save_rewards(record)

    def save_results(self, record: dict):
        """Save average rewards for each parameter."""
        average_rewards = record['average_rewards']
        with open(os.path.join(self.path.play, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for param, reward in zip(self.parameters, average_rewards):
                writer.writerow([param, reward])

    def save_rewards(self, record):
        """Save all rewards, per episode."""
        total_rewards = record['total_rewards']
        for k, rewards in total_rewards:
            path = os.path.join(self.path.play, '%s-%f.txt' % (
                self.path.featurization, float(k)))
            with open(path, 'w') as f:
                f.write(','.join(map(str, rewards)))
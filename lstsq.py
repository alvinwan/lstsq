"""Experiment using least squares to mimick 'optimal' policies.

Usage:
    lstsq.py encode <param> [options]
    lstsq.py encode <param> <param>... [options]
    lstsq.py solve [options]
    lstsq.py play [--n_episodes=<n>] [options]
    lstsq.py random [--n_episodes=<n>] [options]

Options:
    --env_id=<id>           ID of environment to play in [default: SpaceInvaders-v4]
    --name=<name>           Name of trial - looks for <root>/<name>/*.npz [default: raw]
    --root=<root>           Root of data path [default: ./data/]
    --featurize=<f>         Name of featurization technique [default: downsample]
    --params=<s>            Comma-separated string of params to use
    --solver=<s>            Name of solver to use [default: ols]
    --record                Whether or not to record
    --n_train_episodes=<n>  Number of episodes to train on [default: -1]
"""

import docopt
import numpy as np
import os
import os.path
import glob
import csv
import gym
import random
import time

from gym import wrappers

from featurize.downsample import Downsample
from featurize.pca import PCA
from featurize.random import Random
from featurize.convolveNormal import ConvolveNormal
from featurize.convolveNormalSum import ConvolveNormalSum
from featurize.convolveSlice import ConvolveSlice
from solve.ols import OLS
from solve.rols import RegularizedOLS
from solve.random import Random as RandomSolve

from utils import get_data


def main():

    random.seed(0)
    np.random.seed(0)

    arguments = docopt.docopt(__doc__)

    root = arguments['--root']
    name = arguments['--name']
    featurize = arguments['--featurize'] or 'downsample' # TODO(Alvin) default?
    solver = arguments['--solver']

    random_mode = arguments['random']
    encode_mode = arguments['encode']
    solve_mode = arguments['solve']
    play_mode = arguments['play']
    n_train_episodes = int(arguments['--n_train_episodes'])

    env_id = arguments['--env_id']
    env = gym.make(env_id)

    params = arguments['<param>']
    if not isinstance(params, list):
        params = [params]
    if arguments['--params'] and not params:
        params = arguments['--params'].split(',')


    if random_mode:
        featurize = solver = 'random'
        play_mode = True

    if featurize == 'pca':
        Featurizer = PCA
    elif featurize == 'downsample':
        Featurizer = Downsample
    elif featurize == 'random':
        Featurizer = Random
    elif featurize == 'convolveNormal':
        Featurizer = ConvolveNormal
    elif featurize == 'convolveNormalSum':
        Featurizer = ConvolveNormalSum
    elif featurize == 'convolveSlice':
        Featurizer = ConvolveSlice
    else:
        raise UserWarning('Invalid encoding provided. Must be one of: pca, downsample, random, convolveNormal, convolveNormalSum, convolveSlice')

    if solver == 'ols':
        Solver = OLS
    elif solver == 'rols':
        Solver = RegularizedOLS
    elif solver == 'random':
        Solver = RandomSolve
    else:
        raise UserWarning('Invalid solver provided.')

    raw_path = os.path.join(root, 'raw', '*.npz')  # temporarily hard-coded
    featurizer = Featurizer(name, root, env)
    solver = Solver(name, root, featurizer)

    if encode_mode:
        X, Y = get_data(path=raw_path, n_train_episodes=n_train_episodes)
        featurizer.encode(X, Y, params)

    if solve_mode:
        solver.solve()

    if play_mode:
        n_episodes = arguments['--n_episodes'] or 1
        time_id = str(time.time())[-5:]
        print(' * New time id %s' % time_id)

        env = wrappers.Monitor(env, os.path.join(solver.play_dir, time_id),
                               video_callable=lambda _: arguments['--record'])

        if params:
            source_path_fmt = os.path.join(solver.solve_dir, '%s.npz')
            paths = [source_path_fmt % param for param in params]
            parameters = params
        else:
            source_path = os.path.join(solver.solve_dir, '*.npz')
            paths = glob.iglob(source_path)
            parameters = ['.'.join(os.path.basename(path).split('.')[:-1])
                          for path in paths]
            if not parameters:
                raise UserWarning('No solved models found. Did you forget to run `solve`?')
        average_rewards, total_rewards = [], []
        for param in parameters:
            feature_model = featurizer.load_model(param)
            solve_model = solver.load_model(param)
            episode_rewards = []
            total_rewards.append((param , episode_rewards))

            best_mean_reward = 0
            for i in range(int(n_episodes)):
                observation = env.reset()
                rewards = 0
                while True:
                    obs = observation.reshape((1, *observation.shape))
                    featurized = featurizer.phi(obs, feature_model)
                    action = solver.predict(featurized, solve_model)
                    observation, reward, done, info = env.step(action)
                    if done:
                        env.reset()
                        episode_rewards.append(env.get_episode_rewards()[-1])
                        if i % 100 == 0 and i > 0:
                            best_mean_reward = max(best_mean_reward, np.mean(env.get_episode_rewards[-100:]))
                        break
            print(' * (%s) Best Mean Reward: %f' % (param, best_mean_reward))
            average_rewards.append(best_mean_reward)

        for k, rewards in total_rewards:
            path = os.path.join(solver.play_dir, '%s-%f.txt' % (
                featurize, float(k)))
            with open(path, 'w') as f:
                f.write(','.join(map(str, rewards)))

        with open(os.path.join(solver.play_dir, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for param, reward in zip(parameters, average_rewards):
                writer.writerow([param, reward])


if __name__ == '__main__':
    main()

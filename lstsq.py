"""Experiment using least squares to mimick 'optimal' policies.

Usage:
    lstsq.py encode <param> <param>... [options]
    lstsq.py solve [options]
    lstsq.py play [--n_episodes=<n>] [options]
    lstsq.py random [--n_episodes=<n>] [options]

Options:
    --env_id=<id>   ID of environment to play in
    --name=<name>   Name of trial - looks for <root>/<name>/*.npz [default: raw]
    --root=<root>   Root of data path [default: ./data/]
    --featurize=<f> Name of featurization technique [default: downsample]
    --params=<s>    Comma-separated string of params to use
    --solver=<s>    Name of solver to use [default: ols]
    --record        Whether or not to record
"""

import docopt
import numpy as np
import os
import os.path
import glob
import csv
import gym
import random
import cv2
import pickle
import time

from skimage.transform import rescale
from scipy.sparse.linalg import svds
from typing import Tuple
from gym import wrappers

from featurize.downsample import Downsample
from featurize.pca import PCA
from featurize.random import Random
from featurize.convolveNormal import ConvolveNormal
from featurize.convolveNormalSum import ConvolveNormalSum
from featurize.convolveSlice import ConvolveSlice
from solve.ols import OLS
from solve.random import Random as RandomSolve

from utils import get_data


def phi_random(
        X: np.array,
        k: float) -> np.array:
    """Run a random agent by generating a random featurization."""
    new_w = np.round(img_w * k)
    new_h = np.round(img_h * k)
    new_d = int(new_w * new_h * img_c)
    return np.random.random((X.shape[0], new_d))


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

    env_id = arguments['--env_id'] or 'SpaceInvadersNoFrameskip-v4'
    env = gym.make(env_id)
    img_h, img_w, img_c = env.observation_space.shape

    params = arguments['<param>'] or arguments['--params'].split(',')

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
        raise UserWarning('Invalid encoding provided. Must be one of: pca, downsample, random')

    if solver == 'ols':
        Solver = OLS
    elif solver == 'random':
        Solver = RandomSolve
    else:
        raise UserWarning('Invalid solver provided.')

    raw_path = os.path.join(root, name, '*.npz')
    featurizer = Featurizer(name, root, env)
    solver = Solver(name, root, featurizer)

    if encode_mode:
        X, Y = get_data(path=raw_path)
        featurizer.encode(X, Y, params)

    if solve_mode:
        solver.solve()

    if play_mode:
        n_episodes = arguments['--n_episodes'] or 1
        time_id = str(time.time())[-5:]
        print(' * New time id %s' % time_id)

        if arguments['--record']:
            env = wrappers.Monitor(env, os.path.join(solver.play_dir, time_id))

        if params:
            source_path_fmt = os.path.join(solver.solve_dir, '%s.npz')
            paths = [source_path_fmt % param for param in params]
            parameters = params
        else:
            source_path = os.path.join(solver.solve_dir, '*.npz')
            paths = glob.iglob(source_path)
            parameters = ['.'.join(os.path.basename(path).split('.')[:-1])
                          for path in paths]
        average_rewards, total_rewards = [], []
        for param in parameters:
            episode_rewards = []
            total_rewards.append((param , episode_rewards))
            feature_model = featurizer.load_model(param)
            solve_model = solver.load_model(param)

            for _ in range(int(n_episodes)):
                observation = env.reset()
                rewards = 0
                while True:
                    obs = observation.reshape((1, *observation.shape))
                    featurized = featurizer.phi(obs, feature_model)
                    action = solver.predict(featurized, solve_model)
                    observation, reward, done, info = env.step(action)
                    rewards += reward
                    if done:
                        episode_rewards.append(rewards)
                        break
            average_reward = sum(episode_rewards) / float(len(episode_rewards))
            print(' * (%s) Average Reward: %f' % (param, average_reward))
            average_rewards.append(average_reward)
        if not parameters:
            raise UserWarning('No solved models found. Did you forget to run `solve`?')

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

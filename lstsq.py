"""Experiment using least squares to mimick 'optimal' policies.

Usage:
    lstsq.py encode <param> <param>... [options]
    lstsq.py solve [options]
    lstsq.py play [--env_id=<id>] [--n_episodes=<n>] [options]

Options:
    --name=<name>   Name of trial - looks for <root>/<name>/*.npz [default: raw]
    --root=<root>   Root of data path [default: ./data/]
    --featurize=<f> Name of featurization technique [default: downsample]
    --ks=<ks>       Comma-separated string of ks to use
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

from skimage.transform import rescale
from scipy.sparse.linalg import svds
from typing import Tuple

from featurize.downsample import Downsample
from featurize.pca import PCA
from featurize.random import Random
from featurize.convolveNormal import ConvolveNormal
from featurize.convolveNormalSum import ConvolveNormalSum
from featurize.convolveSlice import ConvolveSlice
from solve.ols import OLS

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

    encode_mode = arguments['encode']
    solve_mode = arguments['solve']
    play_mode = arguments['play']

    env_id = arguments['--env_id'] or 'SpaceInvadersNoFrameskip-v4'
    env = gym.make(env_id)
    img_h, img_w, img_c = env.observation_space.shape

    params = arguments['<param>']

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

    raw_path = os.path.join(root, name, '*.npz')
    X, Y = get_data(path=raw_path)
    featurizer = Featurizer(name, root, env)
    solver = OLS(name, root, featurizer)

    if encode_mode:
        featurizer.encode(X, Y, params)

    if solve_mode:
        solver.solve()

    if play_mode:
        n_episodes = arguments['--n_episodes'] or 1

        source_path = os.path.join(solver.solve_dir, '*.npz')
        ks, average_rewards, total_rewards = [], [], []
        for path in glob.iglob(source_path):
            with np.load(path) as datum:
                w = datum['arr_0']
            param = '.'.join(os.path.basename(path).split('.')[:-1])
            episode_rewards = []
            total_rewards.append((param , episode_rewards))

            model = featurizer.load_model(param)

            for _ in range(int(n_episodes)):
                observation = env.reset()
                rewards = 0
                while True:
                    obs = observation.reshape((1, *observation.shape))
                    featurized = featurizer.phi(obs, model)
                    action = np.argmax(featurized.dot(w))
                    observation, reward, done, info = env.step(action)
                    rewards += reward
                    if done:
                        episode_rewards.append(rewards)
                        break
            average_reward = sum(episode_rewards) / float(len(episode_rewards))
            print(' * (%s) Average Reward: %f' % (param, average_reward))
            ks.append(param)
            average_rewards.append(average_reward)
        if not ks:
            raise UserWarning('No solved models found. Did you forget to run `solve`?')

        for k, rewards in total_rewards:
            path = os.path.join(solver.solve_dir, '%s-%f.txt' % (
                featurize, float(k)))
            with open(path, 'w') as f:
                f.write(','.join(map(str, rewards)))

        with open(os.path.join(solver.solve_dir, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for k, reward in zip(ks, average_rewards):
                writer.writerow([k, reward])


if __name__ == '__main__':
    main()

"""Experiment using least squares to mimick 'optimal' policies.

Usage:
    lstsq.py all [options]
    lstsq.py pca <k> <k>... [options]
    lstsq.py downsample <k> <k>... [options]
    lstsq.py ols [options]
    lstsq.py play [--env_id=<id>] [--n_episodes=<n>] [options]

Options:
    --name=<name>   Name of trial - looks for <root>/<name>/*.npz [default: raw]
    --root=<root>   Root of data path [./data/]
    --featurize=<f> Name of featurization technique [default: downsample]
"""

import docopt
import numpy as np
import os
import os.path
import glob
import csv
import gym
import random

from skimage.transform import rescale
from scipy.sparse.linalg import svds
from typing import Tuple


def get_data(path: str, one_hotted: bool=False) -> Tuple[np.array, np.array]:
    """Grab data from provided path."""
    X, Y = None, None
    for fpath in glob.iglob(path):
        if fpath.endswith('.npy'):
            A = np.load(fpath)
        else:
            with np.load(fpath) as datum:
                A = datum['arr_0']
        x, y = A[:,:-2], A[:,-2]
        X = x if X is None else np.vstack((X, x))
        Y = y if Y is None else np.vstack((Y, y))
    if one_hotted:
        return X, one_hot(Y)
    return X, Y


def one_hot(Y: np.array) -> np.array:
    """One hot the provided list of classes."""
    num_classes = len(np.unique(Y))
    return np.eye(num_classes)[Y.astype(int)]


def main():

    arguments = docopt.docopt(__doc__)

    root = arguments['--root']
    name = arguments['--name']
    featurize = arguments['--featurize']
    raw_path = os.path.join(root, name, '*.npz')
    pca_path = os.path.join(root, name + '-pca', '*.npz')
    downsample_path = os.path.join(root, name + '-downsample', '*.npz')
    ols_path = os.path.join(root, name + '-ols', '*.npz')
    play_path = os.path.join(root, name + '-play', '*.npz')

    all_mode = arguments['all']
    pca_mode = arguments['pca']
    ols_mode = arguments['ols']
    play_mode = arguments['play']

    if pca_mode:
        X, Y = get_data(path=raw_path)
        n = X.shape[0]

        ks = map(int, arguments['<k>'])
        U, s, VT = svds(X, k=max(ks))
        dirname = os.path.dirname(pca_path)

        for k in ks:
            projX = U.dot(np.diag(s[:k]))
            data = np.hstack((projX, Y, np.zeros(n)))
            np.savez_compressed(data, os.path.join(dirname, k))

        np.savez_compressed(U, os.path.join(dirname, 'U'))
        np.savez_compressed(s, os.path.join(dirname, 's'))

    if downsample_mode:
        X, Y = get_data(path=raw_path)
        n = X.shape[0]

        ks = map(float, arguments['<k>'])
        dirname = os.path.dirname(downsample_path)

        for k in ks:
            data = np.hstack((rescale(X, k), Y, np.zeros(n)))
            np.savez_compressed(data, os.path.join(dirname, k))

    if ols_mode:
        dirname = os.path.dirname(ols_path)
        ks, accuracies = [], []

        source_path = os.path.join(root, featurize, '*.npz')
        for path in glob.iglob(source_path):
            k = float('.'.join(path.split('.')[:-1]))
            X, Y = get_data(path=path)
            w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(one_hot(Y))

            guesses = np.argmax(projX.dot(w), axis=1)
            accuracy = np.sum(guesses == Y_raw) / float(Y.shape[0])
            print(' * Accuracy for %d: %f' % (k, accuracy))

            np.savez_compressed(w, os.path.join(dirname, k))
            ks.append(k)
            accuracies.append(accuracy)

        with open(os.path.join(dirname, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for k, accuracy in zip(ks, accuracies):
                writer.write([k, accuracy])

    if play_mode:
        env_id = arguments['--env_id'] or 'SpaceInvadersNoFrameskip-v4'
        n_episodes = arguments['--n_episodes'] or 10

        env = gym.make(env_id)
        random.seed(0)
        np.random.seed(0)

        ks, total_rewards = [], []
        for path in glob.iglob(ols_path):
            with np.load(path) as datum:
                w = datum['arr_0']
            k = float('.'.join(path.split('.')[:-1]))
            observation = np.zeros(U.shape[0])
            episode_rewards = []
            for _ in range(int(n_episodes)):
                rewards = 0
                while True:
                    if featurize == 'downsample':
                        observation = rescale(featurize, k)
                        action = observation.dot(w)
                    observation, reward, done, info = env.step(action)
                    rewards += reward
                    if done:
                        episode_rewards.append(rewards)
                        break
            average_reward = sum(episode_rewards) / float(len(episode_rewards))
            print(' * (%f) Average Reward: %f' % (k, average_reward))
            ks.append(k)
            total_rewards.append(average_reward)

        with open(os.path.join(dirname, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for k, reward in zip(ks, total_rewards):
                writer.write([k, reward])

    # Human play for SpaceInvadersNoFrameskip-v4
    # For 4 possible actions
    # Least squares + PCA reduction to 25 dims: 51.5%
    # Least squares + PCA reduction to 100 dims: 59.5%

if __name__ == '__main__':
    main()

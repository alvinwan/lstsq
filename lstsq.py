"""Experiment using least squares to mimick 'optimal' policies.

Usage:
    lstsq.py all [options]
    lstsq.py pca <k> <k>... [options]
    lstsq.py downsample <k> <k>... [options]
    lstsq.py ols [options]
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
        x, y = A[:, :-2], A[:, -2].reshape((-1, 1))
        X = x if X is None else np.vstack((X, x))
        Y = y if Y is None else np.vstack((Y, y))
    if one_hotted:
        return X, one_hot(Y)
    return X, Y


def one_hot(Y: np.array) -> np.array:
    """One hot the provided list of classes."""
    num_classes = len(np.unique(Y))
    return np.eye(num_classes)[Y.astype(int)].reshape((Y.shape[0], num_classes))


def phi_downsample(
        X: np.array,
        k: float,
        img_w: int,
        img_h: int,
        img_c: int) -> np.array:
    """Downsample a set of images."""
    new_w = np.round(img_w * k)
    new_h = np.round(img_h * k)
    new_d = int(new_w * new_h * img_c)
    newX = np.zeros((X.shape[0], new_d))
    for i in range(X.shape[0]):
        image = X[i].reshape((img_w, img_h, img_c)).astype(np.uint8)
        newX[i] = cv2.resize(image, (0, 0), fx=k, fy=k).reshape((1, -1))
    return newX


def phi_pca(
        X: np.array,
        k: float,
        img_w: int,
        img_h: int,
        img_c: int) -> np.array:
    """Run PCA to find projection into subspace."""
    return X


def phi_random(
        X: np.array,
        k: float,
        img_w: int,
        img_h: int,
        img_c: int) -> np.array:
    """Run a random agent by generating a random featurization."""
    new_w = np.round(img_w * k)
    new_h = np.round(img_h * k)
    new_d = int(new_w * new_h * img_c)
    return np.random.random((X.shape[0], new_d))


def main():

    arguments = docopt.docopt(__doc__)

    root = arguments['--root']
    name = arguments['--name']
    featurize = arguments['--featurize'] or 'downsample' # TODO(Alvin) default?
    raw_path = os.path.join(root, name, '*.npz')
    pca_path = os.path.join(root, name + '-pca', '*.npz')
    downsample_path = os.path.join(root, name + '-downsample', '*.npz')
    ols_path = os.path.join(root, name + '-ols', '*.npz')
    play_path = os.path.join(root, name + '-play', '*.npz')

    all_mode = arguments['all']
    pca_mode = arguments['pca']
    downsample_mode = arguments['downsample']
    ols_mode = arguments['ols']
    play_mode = arguments['play']

    env_id = arguments['--env_id'] or 'SpaceInvadersNoFrameskip-v4'
    env = gym.make(env_id)
    img_h, img_w, img_c = env.observation_space.shape

    if pca_mode:
        X, Y = get_data(path=raw_path)
        n = X.shape[0]

        ks = map(int, arguments['<k>'])
        U, s, VT = svds(X, k=max(ks))
        dirname = os.path.dirname(pca_path)
        placeholder = np.zeros((n, 1))

        for k in ks:
            projX = U.dot(np.diag(s[:k]))
            data = np.hstack((projX, Y, placeholder))
            np.savez_compressed(os.path.join(dirname, k), data)

        np.savez_compressed(U, os.path.join(dirname, 'U'))
        np.savez_compressed(s, os.path.join(dirname, 's'))

    if downsample_mode:
        X, Y = get_data(path=raw_path)
        n = X.shape[0]
        placeholder = np.zeros((n, 1))

        ks = map(float, arguments['<k>'])
        dirname = os.path.dirname(downsample_path)
        os.makedirs(dirname, exist_ok=True)

        for k in ks:
            newX = phi_downsample(X, k, img_w, img_h, img_c)
            data = np.hstack((newX, Y, placeholder))
            np.savez_compressed(os.path.join(dirname, str(k)), data)

    if ols_mode:
        dirname = os.path.dirname(ols_path)
        os.makedirs(dirname, exist_ok=True)
        ks, accuracies = [], []

        source_path = os.path.join(root, '%s-%s' % (name, featurize), '*.npz')
        for path in glob.iglob(source_path):
            k = float('.'.join(os.path.basename(path).split('.')[:-1]))
            X, Y = get_data(path=path)
            w = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(one_hot(Y))

            guesses = np.argmax(X.dot(w), axis=1)
            accuracy = np.sum(guesses == np.ravel(Y)) / float(Y.shape[0])
            print(' * Accuracy for %f: %f' % (k, accuracy))

            ks.append(k)
            accuracies.append(accuracy)
            np.savez_compressed(os.path.join(dirname, str(k)), w)

        with open(os.path.join(dirname, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for k, accuracy in zip(ks, accuracies):
                writer.writerow([k, accuracy])

    if play_mode:
        n_episodes = arguments['--n_episodes'] or 10
        dirname = os.path.dirname(play_path)
        os.makedirs(dirname, exist_ok=True)

        random.seed(0)
        np.random.seed(0)

        if featurize == 'downsample':
            phi = phi_downsample
        elif featurize == 'pca':
            phi = phi_pca
        elif featurize == 'random':
            phi = phi_random
        else:
            raise UserWarning('Unknown featurization:', featurize)

        ks, average_rewards, total_rewards = [], [], []
        for path in glob.iglob(ols_path):
            with np.load(path) as datum:
                w = datum['arr_0']
            k = float('.'.join(os.path.basename(path).split('.')[:-1]))
            episode_rewards = []
            total_rewards.append((k, episode_rewards))
            for _ in range(int(n_episodes)):
                observation = env.reset()
                rewards = 0
                while True:
                    obs = observation.reshape((1, *observation.shape))
                    featurized = phi(obs, k, img_w, img_h, img_c)
                    action = np.argmax(np.round(featurized.dot(w)))
                    observation, reward, done, info = env.step(action)
                    rewards += reward
                    if done:
                        episode_rewards.append(rewards)
                        break
            average_reward = sum(episode_rewards) / float(len(episode_rewards))
            print(' * (%f) Average Reward: %f' % (k, average_reward))
            ks.append(k)
            average_rewards.append(average_reward)

        for k, rewards in total_rewards:
            path = os.path.join(dirname, '%s-%f.txt' % (featurize, k))
            with open(path, 'w') as f:
                f.write(','.join(map(str, rewards)))

        with open(os.path.join(dirname, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for k, reward in zip(ks, average_rewards):
                writer.writerow([k, reward])


if __name__ == '__main__':
    main()

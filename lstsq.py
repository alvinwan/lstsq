"""Experiment using least squares to mimick 'optimal' policies.

Usage:
    lstsq.py all [options]
    lstsq.py encode <param> <param>... [options]
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
import pickle

from skimage.transform import rescale
from scipy.sparse.linalg import svds
from typing import Tuple

from featurize.downsample import Downsample
from featurize.pca import PCA
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

    raw_path = os.path.join(root, name, '*.npz')
    pca_path = os.path.join(root, name + '-pca', '*.npz')
    downsample_path = os.path.join(root, name + '-downsample', '*.npz')
    ols_path = os.path.join(root, name + '-ols', '*.npz')
    play_path = os.path.join(root, name + '-play', '*.npz')

    all_mode = arguments['all']
    encode_mode = arguments['encode']
    ols_mode = arguments['ols']
    play_mode = arguments['play']

    env_id = arguments['--env_id'] or 'SpaceInvadersNoFrameskip-v4'
    env = gym.make(env_id)
    img_h, img_w, img_c = env.observation_space.shape

    params = arguments['<param>']

    if featurize == 'pca':
        Featurizer = PCA
    elif featurize == 'downsample':
        Featurizer = Downsample
    else:
        raise UserWarning('Invalid encoding provided. Must be one of: %s' % featurizer.keys())

    X, Y = get_data(path=raw_path)
    featurizer = Featurizer(name, root, env)

    if encode_mode:
        featurizer.encode(X, Y, params)

    if ols_mode:
        solver = OLS(name, root, featurizer)
        solver.solve()

    if play_mode:
        n_episodes = arguments['--n_episodes'] or 10
        ols_dirname = os.path.dirname(ols_path)
        dirname = os.path.dirname(play_path)
        os.makedirs(dirname, exist_ok=True)

        if featurize == 'downsample':
            phi = phi_downsample
            data = {'img_w': img_w, 'img_h': img_h, 'img_c': img_c}
        elif featurize == 'pca':
            phi = phi_pca
            data = {'model': {}}
        elif featurize == 'random':
            phi = phi_random
            data = {}
        else:
            raise UserWarning('Unknown featurization:', featurize)

        source_path = os.path.join(ols_dirname, featurize, '*.npz')
        ks, average_rewards, total_rewards = [], [], []
        for path in glob.iglob(source_path):
            with np.load(path) as datum:
                w = datum['arr_0']
            k = float('.'.join(os.path.basename(path).split('.')[:-1]))
            episode_rewards = []
            total_rewards.append((k, episode_rewards))

            if featurize == 'pca':
                k = int(k)
                pca_dirname = os.path.dirname(pca_path)
                filename = os.path.join(pca_dirname, '%d-model.pkl' % k)
                with open(filename, 'rb') as f:
                    data['model'][k] = pickle.load(f)

            for _ in range(int(n_episodes)):
                observation = env.reset()
                rewards = 0
                while True:
                    obs = observation.reshape((1, *observation.shape))
                    featurized = phi(obs, k, **data)
                    action = np.argmax(featurized.dot(w))
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

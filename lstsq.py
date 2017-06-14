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
    --player=<p>            Type of play [default: simple]
"""

import docopt
import numpy as np
import os
import os.path
import gym
import random

from featurize.downsample import Downsample
from featurize.pca import PCA
from featurize.random import Random
from featurize.convolveNormal import ConvolveNormal
from featurize.convolveNormalSum import ConvolveNormalSum
from featurize.convolveSlice import ConvolveSlice
from solve.ols import OLS
from solve.rols import RegularizedOLS
from solve.random import Random as RandomSolve
from play.simple import SimplePlay

from utils import get_data


def main():

    random.seed(0)
    np.random.seed(0)

    arguments = docopt.docopt(__doc__)

    root = arguments['--root']
    name = arguments['--name']
    featurize = arguments['--featurize'] or 'downsample' # TODO(Alvin) default?
    solver = arguments['--solver']
    player = arguments['--player']

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

    if player == 'simple':
        Player = SimplePlay
    else:
        raise UserWarning('Invalid player provided.')

    raw_path = os.path.join(root, 'raw', '*.npz')  # temporarily hard-coded
    featurizer = Featurizer(name, root, env)
    solver = Solver(name, root, featurizer)

    if encode_mode:
        X, Y = get_data(path=raw_path, n_train_episodes=n_train_episodes)
        featurizer.encode(X, Y, params)

    if solve_mode:
        solver.solve()

    if play_mode:
        n_episodes = int(arguments['--n_episodes']) or 1
        player = Player(env, solver, featurizer, arguments['--record'], params)
        record = player.run(n_episodes)
        player.log(record)


if __name__ == '__main__':
    main()

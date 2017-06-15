"""Experiment using least squares to mimick 'optimal' policies.

Usage:
    lstsq.py encode <param> [options]
    lstsq.py encode <param> <param>... [options]
    lstsq.py solve [options]
    lstsq.py play [--n_episodes=<n>] [options]
    lstsq.py random [--n_episodes=<n>] [options]

Options:
    --env_id=<id>           ID of environment to play in [default: SpaceInvadersNoFrameskip-v4]
    --name=<name>           Name of trial - looks for <root>/<name>/*.npz [default: raw]
    --root=<root>           Root of data path [default: ./data/]
    --featurize=<f>         Name of featurization technique [default: pca]
    --params=<s>            Comma-separated string of params to use
    --solver=<s>            Name of solver to use [default: ols]
    --record                Whether or not to record
    --n_train_episodes=<n>  Number of episodes to train on [default: -1]
    --player=<p>            Type of play [default: simple]
"""

import docopt
import numpy as np
import gym
import random

from path import Path
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
from play.dagger import DaggerPlay

from utils import get_data


def main():

    random.seed(0)
    np.random.seed(0)

    arguments = docopt.docopt(__doc__)

    featurization = arguments['--featurize'] or 'downsample' # TODO(Alvin) default?
    solver = arguments['--solver']
    player = arguments['--player']

    env = gym.make(arguments['--env_id'])

    params = arguments['<param>']
    if not isinstance(params, list):
        params = [params]
    if arguments['--params'] and not params:
        params = arguments['--params'].split(',')

    if arguments['random']:
        featurization = solver = 'random'
        arguments = arguments.copy()
        arguments['play'] = True
        # Replace featurization/solver with random player

    if featurization == 'pca':
        Featurizer = PCA
    elif featurization == 'downsample':
        Featurizer = Downsample
    elif featurization == 'random':
        Featurizer = Random
    elif featurization == 'convolveNormal':
        Featurizer = ConvolveNormal
    elif featurization == 'convolveNormalSum':
        Featurizer = ConvolveNormalSum
    elif featurization == 'convolveSlice':
        Featurizer = ConvolveSlice
    else:
        raise UserWarning('Invalid encoding provided. Must be one of: pca, '
                          'downsample, random, convolveNormal, '
                          'convolveNormalSum, convolveSlice')

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
    elif player == 'dagger':
        Player = DaggerPlay
    else:
        raise UserWarning('Invalid player provided.')

    path = Path(arguments['--name'], featurization, solver, arguments['--root'])

    featurizer = Featurizer(path, env)
    solver = Solver(path, featurizer)

    if arguments['encode']:
        X, Y = get_data(
            path=path.data,
            n_train_episodes=int(arguments['--n_train_episodes']))
        featurizer.encode(X, Y, params)

    if arguments['solve']:
        solver.solve()

    if arguments['play']:
        n_episodes = int(arguments['--n_episodes'] or 1)
        player = Player(path, env, solver, featurizer, arguments['--record'], params)
        record = player.run(n_episodes)
        player.log(record)


if __name__ == '__main__':
    main()

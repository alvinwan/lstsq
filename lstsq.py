"""Experiment using least squares to mimick 'optimal' policies.

Usage:
    lstsq.py (process84|processraw) <dataset1> <dataset2>
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
    --feature-params=<s>    Comma-separated string of params to use for featurization [default: 8]
    --solver-params=<s>     Comma-separated string of params to use for solver [default: 1e-5,1e-3,1e-1,1,1e1,1e2]
    --solver=<s>            Name of solver to use [default: ols]
    --record                Whether or not to record
    --n_train_episodes=<n>  Number of episodes to train on [default: -1]
    --player=<p>            Type of play [default: simple]
    --dataset=<dataset>     Name of dataset [default: raw]
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
from utils import wrap_custom


def main():

    random.seed(0)
    np.random.seed(0)

    arguments = docopt.docopt(__doc__)

    featurization = arguments['--featurize'] or 'downsample' # TODO(Alvin) default?
    solver = arguments['--solver']
    player = arguments['--player']

    env = wrap_custom(gym.make(arguments['--env_id']))

    feature_params = arguments['<param>']
    if not isinstance(feature_params, list):
        feature_params = [feature_params]
    if arguments['--feature-params'] and not feature_params:
        feature_params = arguments['--feature-params'].split(',')
    solver_params = arguments['--solver-params'].split(',')

    if arguments['random']:
        featurization = solver = 'random'
        arguments = arguments.copy()
        arguments['play'] = True
        # Replace featurization/solver with random player

    print(' * [Info] Using featurization %s' % featurization)
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

    print(' * [Info] Using solver %s' % solver)
    if solver == 'ols':
        Solver = OLS
    elif solver == 'rols':
        Solver = RegularizedOLS
    elif solver == 'random':
        Solver = RandomSolve
    else:
        raise UserWarning('Invalid solver provided.')

    print(' * [Info] Using player %s' % player)
    if player == 'simple':
        Player = SimplePlay
    elif player == 'dagger':
        Player = DaggerPlay
    else:
        raise UserWarning('Invalid player provided.')

    path = Path(
        arguments['--name'], featurization, solver, arguments['--root'],
        arguments['--dataset'])

    featurizer = Featurizer(path, env)
    solver = Solver(path, featurizer)

    if arguments['process84']:
        import os.path
        import glob
        from utils import _process_frame84
        source_dir = os.path.join(arguments['--root'], arguments['<dataset1>'])
        source = os.path.join(source_dir, '*.npz')
        dest_dir = os.path.join(arguments['--root'], arguments['<dataset2>'])
        os.makedirs(dest_dir, exist_ok=True)
        for full_path in glob.iglob(source):
            path = os.path.basename(full_path)
            full_dest_path = os.path.join(dest_dir, path)
            with np.load(full_path) as f:
                data = f['arr_0']
            new_data = []
            for row in data:
                obs = _process_frame84(row[:-2]).reshape((1, -1))
                new_data.append(np.hstack((obs, row[-2].reshape((1, 1)), row[-1].reshape((1, 1)))))
            np.savez_compressed(full_dest_path, np.vstack(new_data))

    if arguments['encode']:
        X, Y = get_data(
            path=path.data,
            n_train_episodes=int(arguments['--n_train_episodes']))
        featurizer.encode(X, Y, feature_params)

    if arguments['solve']:
        import os.path
        for feature_param in feature_params:
            X, Y = get_data(path=os.path.join(path.encoded_dir, feature_param + '.npz'))
            solver.solve(X, Y, feature_param, solver_params)

    if arguments['play']:
        if not feature_params:
            raise UserWarning('No parameters specified!')
        n_episodes = int(arguments['--n_episodes'] or 1)
        print(n_episodes)
        player = Player(path, env, solver, featurizer, arguments['--record'], feature_params, solver_params)
        record = player.run(n_episodes)
        player.log(record)


if __name__ == '__main__':
    main()

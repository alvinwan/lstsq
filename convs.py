"""Test convolutional featurization, using pictureweb.

Usage:
    convs.py (featurize|train)

Options:
    --layer     Which layer of the net to train on [default: state]
    --envid     Atari environment ID [default: SpaceInvaders-v0]
"""

import argparse
import glob
import numpy as np
import sys
import sklearn.metrics

from scipy.linalg import solve
from pictureweb.pictureweb.conv.coates_ng_help import grab_patches
from pictureweb.pictureweb.conv._conv import conv_compute_output_shape
from pictureweb.pictureweb.conv._conv import _conv_tf
from pictureweb.pictureweb.conv.filter_gen import make_empirical_filter_gen_no_mmap

n = 90
batch_feature_size = 64
num_feature_batches = 8
data_batch_size = 100
patch_size = 10
pool_size = 70


arguments = sys.argv


def output_shape():

    data = np.random.normal(0.0, 10.0, size=(n, 84, 84, 3))
    data = np.transpose(data, axes=(0, 3, 1, 2))
    print(data.shape)

    out_shape = conv_compute_output_shape(
        data,
        batch_feature_size,
        num_feature_batches,
        data_batch_size,
        patch_size=patch_size,
        pool_size=pool_size)
    print(out_shape)


def featurize(layer, envid):
    """Featurize using convs, filters are random patches from game"""
    fmt = '%s-atari-%s/*_state.npy' % (layer, envid)
    raw_data = [np.load(path) for path in sorted(glob.iglob(fmt))]
    states = [raw[:, :-2].reshape(-1, 84, 84, 3) for raw in raw_data]
    states = [np.transpose(state, axes=(0, 3, 1, 2)) for state in states]
    data = np.vstack(states)
    filter_gen = make_empirical_filter_gen_no_mmap(
        grab_patches(data, patch_size=patch_size))
    out, _ = _conv_tf(
        data,
        filter_gen,
        batch_feature_size,
        num_feature_batches,
        data_batch_size,
        patch_size=patch_size,
        pool_size=pool_size
    )
    out_path_x = '%s-atari-%s/X_%d.npy' % (layer, envid, len(raw_data))
    np.save(out_path_x, out)

    out_path_y = '%s-atari-%s/Y_%d.npy' % (layer, envid, len(raw_data))
    np.save(out_path_y, np.vstack([raw[:, -2][:, None] for raw in raw_data]))


def train(N=100):
    """Run rols. Print accuracy"""
    X = np.load('state-atari-SpaceInvaders-v0/X_%d.npy' % N)
    Y = np.load('state-atari-SpaceInvaders-v0/Y_%d.npy' % N)

    Y_oh = np.eye(6)[np.ravel(Y.astype(int))]
    X = X.reshape((X.shape[0], -1))
    Y = Y.reshape((Y.shape[0], -1))

    print('Solving for w...')
    w = solve(X.T.dot(X) + 1 * np.eye(X.shape[1]), X.T.dot(Y_oh))
    np.save('state-atari-SpaceInvaders-v0/w_%d.npy' % N, w)

    accuracy = sklearn.metrics.accuracy_score(np.argmax(X.dot(w), axis=1), Y)
    print('Accuracy:', accuracy)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='The function to run', choices=('featurize', 'train'))
    parser.add_argument(
        '--layer', help='Name of layer output from A3c', default='state')
    parser.add_argument(
        '--envid', help='ID for Atari environment', default='SpaceInvaders-v0')
    args = parser.parse_args()

    if args.command == 'featurize':
        featurize(args.layer, args.envid)
    else:
        train()


if __name__ == '__main__':
    main()

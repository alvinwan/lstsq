"""Test convolutional featurization, using pictureweb."""

import argparse
import glob
import numpy as np
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'layer', help='Name of layer output from A3c', default='state')
    parser.add_argument(
        'envid', help='ID for Atari environment', default='SpaceInvaders-v0')
    args = parser.parse_args()

    fmt = '%s-atari-%s/*_state.npy' % (args.layer, args.envid)
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
    out_path_x = '%s-atari-%s/X_%d.npy' % (args.layer, args.envid, len(raw_data))
    np.save(out_path_x, out)

    out_path_y = '%s-atari-%s/Y_%d.npy' % (args.layer, args.envid, len(raw_data))
    np.save(out_path_y, np.vstack([state[:, :-2] for state in states]))


if __name__ == '__main__':
    main()

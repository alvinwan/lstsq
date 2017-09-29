"""Minimal script for featurizing any input using conv_tf."""

import numpy as np

from pictureweb.pictureweb.conv.coates_ng_help import grab_patches
from pictureweb.pictureweb.conv._conv import conv_compute_output_shape
from pictureweb.pictureweb.conv._conv import _conv_tf
from pictureweb.pictureweb.conv.filter_gen import make_empirical_filter_gen_no_mmap


__all__ = ('conv',)


def conv(
    data, batch_feature_size=64, num_feature_batches=8,
    data_batch_size=100, patch_size=10, pool_size=150):
    """Run conv_tf on it."""
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
    return out


def output_shape(shape, batch_feature_size=64, num_feature_batches=8,
    data_batch_size=100, patch_size=10, pool_size=150):
    """Use to test and verify output shape."""

    data = np.random.normal(0.0, 10.0, size=shape)
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
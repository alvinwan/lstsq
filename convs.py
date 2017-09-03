"""Test convolutional featurization, using pictureweb."""

import numpy as np
from pictureweb.pictureweb.conv.coates_ng_help import grab_patches
from pictureweb.pictureweb.conv._conv import conv_compute_output_shape
from pictureweb.pictureweb.conv._conv import _conv


####################
# Get output shape #
####################

n = 90
batch_feature_size = 512
num_feature_batches = 8
data_batch_size = 30
patch_size = 10
pool_size = 50

data = np.zeros((n, 84, 84, 3))
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

###############
# Actual conv #
###############

out = _conv(
    data,
    batch_feature_size,
    num_feature_batches,
    data_batch_size,
    patch_size=patch_size,
    pool_size=pool_size
)
print(out.shape)
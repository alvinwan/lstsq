"""Test convolutional featurization, using pictureweb."""

import numpy as np
from pictureweb.pictureweb.conv.coates_ng_help import grab_patches
from pictureweb.pictureweb.conv._conv import conv_compute_output_shape


####################
# Get output shape #
####################

n = 90

data = np.zeros((n, 84, 84, 3))
out = conv_compute_output_shape(data, 512, 8, 30, patch_size=10, pool_size=50)
print(out)



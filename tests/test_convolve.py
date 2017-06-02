import pytest
import numpy as np

from featurize.convolveNormal import ConvolveNormal

def test_convolve_3d_with_2d_dims():
    """Test that 2d filter applied to 3d has expected dimensions."""
    n, w, h, c, size = 5, 10, 10, 3, 5
    images = np.random.random((n, w, h, c))
    filter_ = np.random.random((size, size))
    featurized = ConvolveNormal.convolve3d(images, filter_)
    assert featurized.shape[0] == images.shape[0]

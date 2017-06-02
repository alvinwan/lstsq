from .interface import FeaturizeInterface
from scipy.signal import convolve2d
from typing import List
from .convolveNormal import ConvolveNormal

import os.path
import numpy as np


class ConvolveNormalSum(ConvolveNormal):
    """Run state through convolutions, where filters are random normals."""

    def __init__(self, name: str, root: str, env):
        super(ConvolveNormalSum, self).__init__(name, 'convolve', root, env)

    def phi(self, X: np.ndarray, filters: List[np.array]) -> np.array:
        """Convolve with random normals."""
        X = X.reshape((X.shape[0], *self.image_shape))
        return np.add.sum(ConvolveNormal.convolve3d(X, filter_) for filter_ in filters)

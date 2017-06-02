from .interface import FeaturizeInterface
from scipy.signal import convolve2d
from typing import List

import os.path
import numpy as np


class ConvolveNormal(FeaturizeInterface):
    """Run state through convolutions, where filters are random normals."""

    def __init__(self, name: str, root: str, env):
        super(Convolve, self).__init__(name, 'convolve', root, env)

    def phi(self, X: np.ndarray, filters: List[np.array]) -> np.array:
        """Convolve with random normals."""
        return np.hstack((convolve2d(X, filter_) for filter_ in filters))

    def train(self, _, __, param: str):
        size, num_filters = map(int, param.split(','))
        return [np.random.normal(0.0, 85.0, (size, size))
                for _ in range(num_filters))]

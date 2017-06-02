from .interface import FeaturizeInterface
from scipy.signal import convolve2d
from typing import List
from .convolveNormal import ConvolveNormal

import os.path
import numpy as np


class ConvolveSlice(ConvolveNormal):
    """Run state through convolutions, where filters are random normals."""

    def __init__(self, name: str, root: str, env):
        super(ConvolveSlice, self).__init__(name, root, env, 'convolveSlice')
        self.image_shape = env.observation_space.shape

    def train(self, X: np.array, __, param: str):
        X = X.reshape((X.shape[0], *self.image_shape))
        size, num_filters = map(int, param.split(','))
        filters = []
        for _ in range(num_filters):
            idx = np.random.randint(0, n)
            i = np.random.randint(0, w - size)
            j = np.random.randint(0, h - size)
            c = np.random.randint(0, c)
            filter_ = X[idx, i:i+size, j:j+size, c].reshape((1, size, size))
            filters.append(filter_)
        return filters

from path import Path
from .convolveNormal import ConvolveNormal

import numpy as np


class ConvolveSlice(ConvolveNormal):
    """Run state through convolutions, where filters are random normals."""

    def __init__(self, path: Path, env):
        super(ConvolveSlice, self).__init__(path, env)
        self.image_shape = env.observation_space.shape
        self.w, self.h, self.c = self.image_shape

    def train(self, X: np.array, __, param: str):
        X = X.reshape((X.shape[0], *self.image_shape))
        size, num_filters = map(int, param.split(','))
        n = X.shape[0]
        filters = []
        for _ in range(num_filters):
            idx = np.random.randint(0, n)
            i = np.random.randint(0, self.w - size)
            j = np.random.randint(0, self.h - size)
            c = np.random.randint(0, self.c)
            filter_ = X[idx, i:i+size, j:j+size, c].reshape((1, size, size))
            filters.append(filter_)
        return np.vstack(filters)

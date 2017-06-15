from typing import List
from .convolveNormal import ConvolveNormal

import numpy as np


class ConvolveNormalSum(ConvolveNormal):
    """Run state through convolutions, where filters are random normals."""

    def phi(self, X: np.ndarray, filters: List[np.array]) -> np.array:
        """Convolve with random normals."""
        X = X.reshape((X.shape[0], *self.image_shape))
        return np.add.reduce([ConvolveNormal.convolve3d(X, filter_) for filter_ in filters])

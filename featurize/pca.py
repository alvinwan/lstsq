from .interface import FeaturizeInterface
from typing import List
from sklearn.decomposition import PCA as skPCA

import os.path
import numpy as np


class PCA(FeaturizeInterface):
    """Project into subspace."""

    def __init__(self, name: str, root: str, env):
        super(PCA, self).__init__(name, 'pca', root, env)

    def phi(self, X: np.ndarray, model) -> np.array:
        """Downsample an image."""
        return model.transform(X)

    def train(self, X: np.array, _, param: str):
        k = int(param)
        model = skPCA(n_components=k)
        model.fit(X)
        return model

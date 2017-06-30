from .interface import FeaturizeInterface
from sklearn.decomposition import PCA as skPCA
from utils import PCA as cuPCA

import numpy as np


class PCA(FeaturizeInterface):
    """Project into subspace."""

    def phi(self, X: np.ndarray, model) -> np.array:
        """Use PCA to project onto subspace."""
        return model.transform(X.reshape((X.shape[0], -1)))

    def train(self, X: np.array, _, param: str):
        k = int(param)
        model = skPCA(n_components=k)
        model.fit(X)
        return model

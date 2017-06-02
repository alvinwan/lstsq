from .interface import FeaturizeInterface
from typing import List
from sklearn.decomposition import PCA as skPCA

import os.path
import pickle
import numpy as np


class PCA(FeaturizeInterface):
    """Project into subspace."""

    def __init__(self, name: str, root: str, env):
        super(PCA, self).__init__(name, 'pca', root, env)

    def load_model(self, param: str):
        pass

    def save_model(self, model, param: str):
        filename = os.path.join(self.encoded_dir, '%s-model.pkl' % param)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(' * Wrote %s model to %s' % (param, filename))

    def phi(self, X: np.ndarray, model) -> np.array:
        """Downsample an image."""
        return model.transform(X)

    def train(self, X: np.array, _, param: str):
        k = int(param)
        model = skPCA(n_components=k)
        model.fit(X)
        return model

    def save_encoded(self, X: np.ndarray, Y: np.array, param: str):
        n = X.shape[0]
        placeholder = np.zeros((n, 1))

        data = np.hstack((X, Y, placeholder))
        np.savez_compressed(os.path.join(self.encoded_dir, param), data)

    def load_encoded(self):
        pass

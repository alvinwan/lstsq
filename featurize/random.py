from .interface import FeaturizeInterface
from typing import List

import os.path
import pickle
import numpy as np


class Random(FeaturizeInterface):
    """Random agent."""

    def __init__(self, name: str, root: str, env):
        super(Random, self).__init__(name, 'random', root, env)

    def load_model(self, param: str):
        pass

    def save_model(self, model, param: str) -> np.array:
        pass

    def phi(self, X: np.ndarray, model) -> np.array:
        """Downsample an image."""
        return np.random.random(X.shape)

    def train(self, X: np.array, _, param: str):
        pass

    def save_encoded(self, X: np.ndarray, Y: np.array, param: str):
        raise UserWarning('Why are you saving random encodings? Just `play` with the random agent.')

    def load_encoded(self):
        pass

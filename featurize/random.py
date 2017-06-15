from .interface import FeaturizeInterface

import numpy as np


class Random(FeaturizeInterface):
    """Random agent."""

    def load_model(self, param: str):
        pass

    def save_model(self, model, param: str) -> np.array:
        pass

    def phi(self, X: np.ndarray, _) -> np.array:
        return X

    def train(self, X: np.array, Y: np.array, param: str):
        pass

    def save_encoded(self, X: np.ndarray, Y: np.array, param: str):
        raise UserWarning('Why are you saving random encodings? Just `play` with the random agent.')

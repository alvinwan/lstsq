from .interface import SolveInterface
from featurize.interface import FeaturizeInterface
from utils import one_hot

import numpy as np
import os.path


class Random(SolveInterface):

    def predict(self, X: np.array, num_actions: int):
        """Predict using the new data. Return nx1 column vector."""
        return np.random.randint(0, num_actions)

    def train(self, X: np.array, Y: np.array, solver_param: str) -> np.array:
        pass

    def save_model(self, model, feature_param: str, solver_param: str):
        pass

    def load_model(self, feature_param: str, solver_param: str) -> np.array:
        return 4

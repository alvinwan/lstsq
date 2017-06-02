from .interface import SolveInterface
from featurize.interface import FeaturizeInterface
from utils import one_hot

import numpy as np
import os.path


class OLS(SolveInterface):

    def __init__(self, name: str, root: str, featurizer: FeaturizeInterface):
        super(OLS, self).__init__(name, 'ols', root, featurizer)

    def predict(self, X: np.array, model):
        """Predict using the new data. Return nx1 column vector."""
        return np.argmax(X.dot(model), axis=1).reshape((-1, 1))

    def train(self, X: np.array, Y: np.array) -> np.array:
        return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(one_hot(Y))

    def save_model(self, model, param: str):
        np.savez_compressed(os.path.join(self.solve_dir, param), model)

    def load_model(self, param: str) -> np.array:
        with np.load(os.path.join(self.solve_dir, param + '.npz')) as data:
            model = data['arr_0']
        return model

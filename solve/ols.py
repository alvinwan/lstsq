from .interface import SolveInterface
from featurize.interface import FeaturizeInterface
from utils import one_hot

import numpy as np
import os.path


class OLS(SolveInterface):

    def __init__(self, name: str, root: str, featurizer: FeaturizeInterface):
        super(OLS, self).__init__(name, 'ols', root, featurizer)

    def train(self, X: np.array, Y: np.array, _) -> np.array:
        return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(one_hot(Y))

    def evaluate(self, X: np.array, model) -> np.array:
        return X.dot(model)

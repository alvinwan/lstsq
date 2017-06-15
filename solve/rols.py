"""L2 regularized least squares"""

from .ols import OLS
from utils import one_hot

import numpy as np


class RegularizedOLS(OLS):

    can_batch_train = True

    def train(self, X: np.array, Y: np.array, lambda_: str) -> np.array:
        I = np.eye(X.shape[1])
        return np.linalg.pinv(X.T.dot(X) + float(lambda_) * I) \
            .dot(X.T).dot(one_hot(Y))

    def evaluate(self, X: np.array, model) -> np.array:
        return X.dot(model)

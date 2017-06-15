from path import Path
from featurize.interface import FeaturizeInterface
from utils import get_data

import numpy as np
import os
import os.path
import glob
import csv


class SolveInterface:
    """Interface for solving"""

    def __init__(self, path: Path, featurize: FeaturizeInterface):
        """Initialize the featurization path.

        :param featurize: featurization technique
        """
        self.path = path
        self.featurize = featurize

    def train(self, X: np.array, Y: np.array, param: str):
        """Train and return the model"""
        raise NotImplementedError()

    def predict(self, X: np.array, model):
        """Predict using the new data. Return nx1 column vector."""
        return np.argmax(self.evaluate(X, model), axis=1).reshape((-1, 1))

    def evaluate(self, X: np.array, model) -> np.array:
        raise NotImplementedError()

    def solve(self):
        """Solve for model and save model to disk."""
        params, accuracies = [], []
        for path in glob.iglob(self.path.encoded):
            param = '.'.join(os.path.basename(path).split('.')[:-1])
            X, Y = get_data(path=path)
            n = float(X.shape[0])
            model = self.train(X, Y, param)
            yhat = self.predict(X, model)
            accuracy = np.sum(yhat == Y) / n
            print(' * Accuracy for %s: %f' % (param, accuracy))
            self.save_model(model, param)
            params.append(param)
            accuracies.append(accuracy)
        with open(os.path.join(self.path.solve_dir, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for param, accuracy in zip(params, accuracies):
                writer.writerow([param, accuracy])

    def save_model(self, model, param: str):
        """Save the model to disk."""
        np.savez_compressed(os.path.join(self.path.solve_dir, param), model)

    def load_model(self, param: str) -> np.array:
        """Load the model"""
        with np.load(os.path.join(self.path.solve_dir, param + '.npz')) as data:
            model = data['arr_0']
        return model

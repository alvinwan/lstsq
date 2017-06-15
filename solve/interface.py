from path import Path
from featurize.interface import FeaturizeInterface
from typing import List

import numpy as np
import os
import os.path
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

    def solve(
            self,
            X: np.array,
            Y: np.array,
            feature_param: str='',
            solver_params: List[str] = ()):
        """Solve for model and save model to disk."""
        accuracies = []
        for param in solver_params:
            n = float(X.shape[0])
            model = self.train(X, Y, param)
            yhat = self.predict(X, model)
            accuracy = np.sum(yhat == Y) / n
            print(' * Accuracy for %s, %s: %f' % (feature_param, param, accuracy))
            self.save_model(model, feature_param, param)
            accuracies.append(accuracy)
        with open(os.path.join(self.path.solve_dir, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for param, accuracy in zip(solver_params, accuracies):
                writer.writerow([feature_param, param, accuracy])

    def save_model(self, model, feature_param: str, param: str):
        """Save the model to disk."""
        if feature_param:
            param = '%s-%s' % (feature_param, param)
        np.savez_compressed(os.path.join(self.path.solve_dir, param), model)

    def load_model(self, feature_param: str, param: str) -> np.array:
        """Load the model"""
        if feature_param:
            param = '%s-%s' % (feature_param, param)
        with np.load(os.path.join(self.path.solve_dir, param + '.npz')) as data:
            model = data['arr_0']
        return model

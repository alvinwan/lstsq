from typing import List
from featurize.interface import FeaturizeInterface
from utils import get_data

import numpy as np
import os
import os.path
import glob
import csv


class SolveInterface:
    """Interface for solving"""

    fmt_solve = '%s-solve/%s/*.npz'
    fmt_play = '%s-play/%s/*.npz'

    def __init__(self, name: str, technique: str, root: str, featurize: FeaturizeInterface):
        """Initialize the featurization path.

        :param name: Name of scope for trial
        :param technique: Name of solution technique used
        :param root: Root for all data
        :param featurize: featurization technique
        """
        self.name = name
        self.root = root
        self.technique = technique
        self.featurize = featurize

        os.makedirs(self.solve_dir, exist_ok=True)
        os.makedirs(self.play_dir, exist_ok=True)

    @property
    def __base_path(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def solve_path(self) -> str:
        return self.fmt_solve % (self.__base_path, self.featurize.technique)

    @property
    def solve_dir(self) -> str:
        return os.path.dirname(self.solve_path)

    # TODO(Alvin): These are misplaced - create a player class?
    @property
    def play_path(self) -> str:
        return self.fmt_play % (self.__base_path, self.featurize.technique)

    @property
    def play_dir(self) -> str:
        return os.path.dirname(self.play_path)

    def train(self, X: np.array, Y: np.array, param: str):
        """Train and return the model"""
        raise NotImplementedError()

    def predict(self, X: np.array, model):
        """Predict using the new data. Return nx1 column vector."""
        return np.argmax(self.evaluate(X, model), axis=1).reshape((-1, 1))

    def solve(self):
        """Solve for model and save model to disk."""
        params, accuracies = [], []
        for path in glob.iglob(self.featurize.encoded_path):
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
        with open(os.path.join(self.solve_dir, 'results.csv'), 'w') as f:
            writer = csv.writer(f)
            for param, accuracy in zip(params, accuracies):
                writer.writerow([param, accuracy])

    def save_model(self, model, param: str):
        """Save the model to disk."""
        np.savez_compressed(os.path.join(self.solve_dir, param), model)

    def load_model(self, param: str) -> np.array:
        """Load the model"""
        with np.load(os.path.join(self.solve_dir, param + '.npz')) as data:
            model = data['arr_0']
        return model

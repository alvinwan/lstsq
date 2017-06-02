import os.path
import os
import pickle
import numpy as np

from typing import Tuple
from typing import List


class FeaturizeInterface:
    """Interface for a featurization method"""

    fmt_encoded = '%s-enc/%s/*.npz'
    fmt_model = '%s-model/%s/*.npz'

    def __init__(self, name: str, technique: str, root: str, env):
        """Initialize the featurization path.

        :param name: Name of scope for trial
        :param technique: Name of featurization technique used
        :param root: Root for all data
        :param env: OpenAI Gym environment
        """
        self.name = name
        self.root = root
        self.technique = technique
        self.env = env

        os.makedirs(self.encoded_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.image_shape = env.observation_space.shape
        self.img_w, self.img_h, self.img_c = self.image_shape

    @property
    def __base_path(self) -> str:
        return os.path.join(self.root, self.name)

    @property
    def encoded_path(self) -> str:
        return self.fmt_encoded % (self.__base_path, self.technique)

    @property
    def encoded_dir(self) -> str:
        return os.path.dirname(self.encoded_path)

    @property
    def model_path(self) -> str:
        return self.fmt_model % (self.__base_path, self.technique)

    @property
    def model_dir(self) -> str:
        return os.path.dirname(self.model_path)

    def encode(self, X: np.array, Y: np.array, params: List[str]):
        """Encode all samples and save to disk."""
        for param in params:
            model = self.train(X, Y, param)
            self.save_encoded(self.phi(X, model), Y, param)
            self.save_model(model, param)

    def train(self, X: np.array, Y: np.array, param: str):
        """Train a model for the provided task, and return the model."""
        raise NotImplementedError

    def load_model(self, param: str):
        """Load model from model dir/. Updates self."""
        filename = os.path.join(self.encoded_dir, '%s-model.pkl' % param)
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def save_model(self, model, param: str) -> np.array:
        """Save model to model dir."""
        filename = os.path.join(self.encoded_dir, '%s-model.pkl' % param)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(' * Wrote featurization model (%s) to %s' % (param, filename))

    def save_encoded(self, X: np.array, Y: np.array, param: str):
        """Save an encoded dataset with provided hyperparameters."""
        n = X.shape[0]
        placeholder = np.zeros((n, 1))

        data = np.hstack((X, Y, placeholder))
        encoded_path = os.path.join(self.encoded_dir, param)
        np.savez_compressed(encoded_path, data)
        print(' * Wrote encoded data (%s) to %s' % (param, encoded_path))

    def phi(self, X: np.array, model) -> np.array:
        """Featurize the provided set of sample. Returns 1xd row vector."""
        raise NotImplementedError()

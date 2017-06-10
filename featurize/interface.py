import os.path
import os
import pickle
import numpy as np

from typing import Tuple
from typing import List


class FeaturizeInterface:
    """Interface for a featurization method"""

    fmt_encoded = '%s-enc/%s/%s/*.npz'
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
        self.featurized_X = None

        os.makedirs(self.encoded_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.image_shape = env.observation_space.shape
        self.img_w, self.img_h, self.img_c = self.image_shape

    @property
    def __base_path(self) -> str:
        return os.path.join(self.root, self.name)

    def encoded_path(self, param: str) -> str:
        return self.fmt_encoded % (self.__base_path, self.technique, param)

    def encoded_dir(self, param: str) -> str:
        return os.path.dirname(self.encoded_path(param))

    @property
    def model_path(self) -> str:
        return self.fmt_model % (self.__base_path, self.technique)

    @property
    def model_dir(self) -> str:
        return os.path.dirname(self.model_path)

    def encode(self, X: np.array, Y: np.array, params: List[str]):
        """Encode all samples and save to disk."""
        for param in params:
            model = self.load_model(param)
            self.featurized_X = self.phi(X, model)
            self.save_encoded(self.featurized_X, Y, param)
            self.save_model(model, param)

    def train(self, X: np.array, Y: np.array, param: str):
        """Train a model for the provided task, and return the model."""
        raise NotImplementedError

    def load_model(self, param: str):
        """Load model from model dir/. Updates self."""
        filename = os.path.join(self.encoded_dir(param), 'model.pkl')
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def save_model(self, model, param: str) -> np.array:
        """Save model to model dir."""
        filename = os.path.join(self.encoded_dir(param), 'model.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(' * Wrote featurization model (%s) to %s' % (param, filename))

    def save_encoded(
            self,
            X: np.array,
            Y: np.array,
            param: str,
            original_path: str):
        """Save an encoded sample with provided hyperparameters."""
        n = X.shape[0]
        placeholder = np.zeros((n, 1))

        data = np.hstack((X, Y, placeholder))
        original_name = os.path.basename(original_path).split('.')[0]
        encoded_path = os.path.join(self.encoded_dir(param), original_name)
        np.savez_compressed(encoded_path, data)
        print(' * Wrote encoded data (%s/%s) to %s' % (
            param, original_name, encoded_path))

    def load_encoded(self, param: str):
        """Load encoded dataset with provided hyperparameters"""
        encoded_path = os.path.join(self.encoded_dir(param), )
        with np.load(encoded_path) as data:
            return data['arr_0']

    def phi(self, X: np.array, model) -> np.array:
        """Featurize the provided set of sample. Returns 1xd row vector."""
        raise NotImplementedError()

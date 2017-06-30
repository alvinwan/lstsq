import os.path
import os
import pickle
import numpy as np

from path import Path
from typing import List


class FeaturizeInterface:
    """Interface for a featurization method"""

    def __init__(self, path: Path, env):
        """Initialize the featurization path.

        :param env: OpenAI Gym environment
        """
        self.env = env
        self.path = path

        self.image_shape = env.observation_space.shape
        self.img_w, self.img_h, self.img_c = self.image_shape

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
        filename = os.path.join(self.path.encoded_dir, '%s-model.pkl' % param)
        print(' * [Info] Loading model', filename)
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model

    def save_model(self, model, param: str) -> np.array:
        """Save model to model dir."""
        filename = os.path.join(self.path.encoded_dir, '%s-model.pkl' % param)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(' * Wrote featurization model (%s) to %s' % (param, filename))

    def save_encoded(self, X: np.array, Y: np.array, param: str):
        """Save an encoded dataset with provided hyperparameters."""
        n = X.shape[0]
        placeholder = np.zeros((n, 1))

        data = np.hstack((X, Y, placeholder))
        encoded_path = os.path.join(self.path.encoded_dir, '%s' % (param))
        np.savez_compressed(encoded_path, data)
        print(' * Wrote encoded data (%s) to %s' % (param, encoded_path))

    def phi(self, X: np.array, model) -> np.array:
        """Featurize the provided set of sample. Returns 1xd row vector."""
        raise NotImplementedError()

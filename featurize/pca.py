from .interface import FeaturizeInterface
from typing import List


class PCA(FeaturizeInterface):
    """Project into subspace."""

    def __init__(self, name: str, root: str, env):
        super(PCA, self).__init__(name, 'pca', root, env)

    def load_model(self):
        pass

    def save_model(self, model):
        filename = os.path.join(dirname, '%d-model.pkl' % k)
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(' * Wrote %d model to %s' % (k, filename))

    def phi(self, X: np.ndarray, model) -> np.array:
        """Downsample an image."""
        return model.transform(X.reshape((1, -1)))

    def train(self, X: np.array, _, param: str):
        k = int(param)
        model = PCA(n_components=k)
        model.fit(X)
        return model

    def save_encoded(self, X: np.ndarray, Y: np.array, param: str):
        n = X.shape[0]
        placeholder = np.zeros((n, 1))
        
        data = np.hstack((X, Y, placeholder))
        np.savez_compressed(os.path.join(dirname, param), data)

    def load_encoded(self):
        pass

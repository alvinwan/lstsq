from .interface import SolveInterface
from featurize.interface import FeaturizeInterface
from utils import one_hot


class OLS(SolveInterface):

    def __init__(self, name: str, root: str, featurizer: FeaturizeInterface):
        super(OLS, self).__init__(name, root, 'ols', featurizer)

    def predict(self, X: np.array, model):
        """Predict using the new data. Return nx1 column vector."""
        return np.argmax(X.dot(w), axis=1).reshape((-1, 1))

    def train(self, X: np.array, Y: np.array) -> np.array:
        return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(one_hot(Y))

    def save_model(self, model, param: str):
        np.savez_compressed(os.path.join(self.solve_dir, self.featurize.name, param), w)

    def load_model(self, param: str) -> np.array:
        with open(os.path.join(self.solve_dir, ))

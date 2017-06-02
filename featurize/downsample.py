from .interface import FeaturizeInterface
from typing import List


class Downsample(FeaturizeInterface):
    """Downsample the raw pixels."""

    def __init__(self, name: str, root: str):
        super(Downsample, self).__init__(name, 'downsample', root)

    def load_model(self, env):
        self.image_shape = env.observation_space.shape
        self.img_w, self.img_h, self.img_c = self.image_shape

    def save_model(self, env):
        pass

    def phi(self, X: np.ndarray, k: float) -> np.array:
        """Downsample an image."""
        new_w = np.round(self.img_w * k)
        new_h = np.round(self.img_h * k)
        new_d = int(new_w * new_h * self.img_c)
        newX = np.zeros((X.shape[0], new_d))
        for i in range(X.shape[0]):
            image = X[i].reshape(self.image_shape).astype(np.uint8)
            newX[i] = cv2.resize(image, (0, 0), fx=k, fy=k).reshape((1, -1))
        return newX

    def train(self, X: np.array, _, k: str) -> float:
        return float(k)

    def save_encoded(self, X: np.array, Y: np.array, param: str):
        n = X.shape[0]
        placeholder = np.zeros((n, 1))

        data = np.hstack((X, Y, placeholder))
        np.savez_compressed(os.path.join(dirname, param, data))

    def load_encoded(self, params: List[str]):
        pass

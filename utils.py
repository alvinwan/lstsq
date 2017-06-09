from typing import Tuple

import numpy as np
import glob


def one_hot(Y: np.array) -> np.array:
    """One hot the provided list of classes."""
    num_classes = len(np.unique(Y))
    return np.eye(num_classes)[Y.astype(int)].reshape((Y.shape[0], num_classes))


def get_data(
        path: str,
        one_hotted: bool=False,
        n_train_episodes: int=-1) -> Tuple[np.array, np.array]:
    """Grab data from provided path."""
    X, Y = None, None
    by_score = lambda p: int(p.split('_')[-1].split('.')[0] or 0)
    for i, fpath in enumerate(sorted(
            glob.iglob(path), key=by_score, reverse=True)):
        if fpath.endswith('.npy'):
            A = np.load(fpath)
        else:
            with np.load(fpath) as datum:
                A = datum['arr_0']
        x, y = A[:, :-2], A[:, -2].reshape((-1, 1))
        X = x if X is None else np.vstack((X, x))
        Y = y if Y is None else np.vstack((Y, y))
        if i + 1 == n_train_episodes:
            break
    if one_hotted:
        return X, one_hot(Y)
    return X, Y

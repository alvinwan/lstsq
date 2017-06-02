def one_hot(Y: np.array) -> np.array:
    """One hot the provided list of classes."""
    num_classes = len(np.unique(Y))
    return np.eye(num_classes)[Y.astype(int)].reshape((Y.shape[0], num_classes))

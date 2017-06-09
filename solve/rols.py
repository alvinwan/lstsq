"""L2 regularized least squares"""


from .ols import OLS


class RegularizedOLS(OLS):

    can_batch_train = True

    def __init__(self, name: str, root: str, featurizer: FeaturizeInterface):
        super(RegularizedOLS, self).__init__(name, 'rols', root, featurizer)

    def train(self, X: np.array, Y: np.array, lambda_: str) -> np.array:
        I = np.eye(X.shape[1])
        return np.linalg.pinv(X.T.dot(X) + float(lambda_) * I) \
            .dot(X.T).dot(one_hot(Y))

    def evaluate(self, X: np.array, model) -> np.array:
        return X.dot(model)

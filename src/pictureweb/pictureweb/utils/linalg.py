import numpy as np
try:
    import scipy.linalg
    from sklearn.kernel_approximation import Nystroem
except:
    pass
from numba import jit
from . import misc
import concurrent.futures as fs
import math

@jit(nopython=True, nogil=True)
def fast_exp(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = math.exp(x[i,j])
    return x


def computeDistanceMatrix(XTest, XTrain, sq_norms_train=None):
    XTrain = XTrain.reshape(XTrain.shape[0], -1)
    XTest = XTest.reshape(XTest.shape[0], -1)
    if sq_norms_train == None:
        XTrain_norms = (np.linalg.norm(XTrain, axis=1) ** 2)[:, np.newaxis]
    else:
        XTrain_norms = sq_norms_train

    XTest_norms = (np.linalg.norm(XTest, axis=1) ** 2)[:, np.newaxis]
    K = XTest.dot(XTrain.T)
    K *= -2
    K += XTrain_norms.T
    K += XTest_norms
    return K

def computeRBFGramMatrix(XTest, XTrain, gamma=1, sq_norms_train=None):
    gamma = -1.0 * gamma
    return fast_exp(gamma*computeDistanceMatrix(XTest, XTrain, sq_norms_train))

def make_smart_precondition(X_train, lambdav, gamma, n_components):
    '''Make a preconditioner for rbf kernel ridge regression by using nystroem
       features  '''
    print("Computing Nystroem svd")
    nystroem = Nystroem(gamma=gamma, n_components=n_components)
    nystroem.fit(X_train)
    print("Computing Nystroem features")
    X_train_lift = nystroem.transform(X_train)
    print("Computing ZTZ")
    ztz = X_train_lift.T.dot(X_train_lift)
    ztz_reg = ztz + lambdav * np.eye(ztz.shape[0]).astype('float32')
    print("Computing Cholesky")
    L = np.linalg.cholesky(ztz_reg)
    U = scipy.linalg.solve(L, X_train_lift.T)
    print(U.shape)
    def prc(x):
        return (1.0/lambdav)*(x - U.T.dot(U.dot(x)))
    return prc, U

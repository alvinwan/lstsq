"""Get and Eval model"""

import numpy as np
from scipy.linalg import solve
from sklearn.metrics import accuracy_score

env_id = 'SpaceInvaders-v0'
model_id = '1_conv'

X = np.load('compute-210x160-%s/X_%s.npy' % (env_id, model_id))
X = X.reshape((X.shape[0], -1))
Y = np.load('compute-210x160-%s/Y_%s.npy' % (env_id, model_id))

Y_oh = np.eye(6)[np.ravel(Y).astype(int)]

reg = 1
I = np.eye(X.shape[1])
w = solve(X.T.dot(X) + reg * I, X.T.dot(Y_oh))
np.save('compute-210x160-%s/w_%s.npy' % (env_id, model_id), w)

accuracy = accuracy_score(np.argmax(X.dot(w), axis=1), Y)
print(accuracy)

"""Get and Eval model"""

import numpy as np
from scipy.linalg import solve
from sklearn.metrics import accuracy_score
import sys

# parse cli
arguments = sys.argv
model_id = '1_conv'
env_id = 'SpaceInvaders-v0'
if len(arguments) > 1:
    model_id = arguments[1]
if len(arguments) > 2:
    env_id = arguments[2]

X = np.load('compute-210x160-%s/X_%s.npy' % (env_id, model_id))
Y = np.load('compute-210x160-%s/Y_%s.npy' % (env_id, model_id))

X = X.reshape((X.shape[0], X.shape[-1]))
Y = Y.reshape((Y.shape[0], 1))

Y_oh = np.eye(6)[np.ravel(Y).astype(int)]

w = solve(X.T.dot(X), X.T.dot(Y_oh))
np.save('compute-210x160-%s/w_%s.npy' % (env_id, model_id), w)

accuracy = accuracy_score(np.argmax(X.dot(w), axis=1), Y)
print(accuracy)

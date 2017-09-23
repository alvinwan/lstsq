"""Get and Eval model"""

import numpy as np
from scipy.linalg import solve
from sklearn.metrics import accuracy_score

X = np.load('compute-atari-SpaceInvaders-v0/X_100_fc0.npy').reshape((-1, 512))
Y = np.load('compute-atari-SpaceInvaders-v0/Y_100_conv.npy')[3:-1,] #[:X.shape[0]]

Y_oh = np.eye(6)[np.ravel(Y).astype(int)]

w = solve(X.T.dot(X), X.T.dot(Y_oh))
np.save('compute-atari-SpaceInvaders-v0/w_100_fc0.npy', w)

accuracy = accuracy_score(np.argmax(X.dot(w), axis=1), Y)
print(accuracy)

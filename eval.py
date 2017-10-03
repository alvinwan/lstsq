"""Get and Eval model"""

import numpy as np
from scipy.linalg import solve
from sklearn.metrics import accuracy_score

X = np.load('compute-210x160-SpaceInvaders-v0/X_1_conv.npy')
X = X.reshape((X.shape[0], -1))
Y = np.load('compute-210x160-SpaceInvaders-v0/Y_1_conv.npy')

Y_oh = np.eye(6)[np.ravel(Y).astype(int)]

reg = 1
I = np.eye(X.shape[1])
w = solve(X.T.dot(X) + reg * I, X.T.dot(Y_oh))
np.save('compute-210x160-SpaceInvaders-v0/w_1_fc0.npy', w)

accuracy = accuracy_score(np.argmax(X.dot(w), axis=1), Y)
print(accuracy)

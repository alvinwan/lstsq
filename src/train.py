"""Get and Eval model"""

import numpy as np
from scipy.linalg import solve
from sklearn.metrics import accuracy_score
import sys
import gym
import warnings
import os

warnings.filterwarnings('ignore')

# parse cli
arguments = sys.argv
model_id = '42570_canny'  # <featurization hyperparameter>_<featurization name>
env_id = 'SpaceInvaders-v0'
if len(arguments) > 1:
    model_id = arguments[1]
if len(arguments) > 2:
    env_id = arguments[2]

env = gym.make(env_id)

X = np.load('../compute-210x160-%s/X_%s.npy' % (env_id, model_id))[()]  # assuming X is sparse
Y = np.load('../compute-210x160-%s/Y_%s.npy' % (env_id, model_id))

X = X.reshape((X.shape[0], -1)).astype(np.float32)
Y = Y.reshape((Y.shape[0], 1)).astype(np.float32)

print(X.shape, Y.shape, X.dtype, Y.dtype)

Y_oh = np.eye(env.action_space.n)[np.ravel(Y).astype(int)]

regs = (1e-7, 1e-5, 1e-3, 1e-1, 1, 1e1, 1e2, 1e3, 1e5)
I = np.eye(X.shape[1])
print('I initialized')
XTX = X.T.dot(X)
print('XTX initialized')
XTY = X.T.dot(Y_oh)
print('XTY initialiezd')

test_path = '../compute-210x160-%s/X_%s_test.npy' % (env_id, model_id)
X_test = Y_test = None
if os.path.exists(test_path):
    X_test = np.load('../compute-210x160-%s/X_%s_test.npy' % (env_id, model_id))
    Y_test = np.load('../compute-210x160-%s/Y_%s_test.npy' % (env_id, model_id))

results = []
for reg in regs:
    try:
        w = solve(XTX + reg*I, XTY)
        acc = accuracy_score(np.argmax(X.dot(w), axis=1), Y)
    except np.linalg.linalg.LinAlgError:
        w = None
        acc = 0
    test_acc = 0
    if X_test is not None:
        test_acc = accuracy_score(np.argmax(X_test.dot(w), axis=1), Y_test)
    print('Regularization:', reg, '// Accuracy:', acc, '// Val Accuracy:', test_acc)
    results.append((w, acc, test_acc))

if X_test is not None:
    w, acc, test_acc = max(results, key=lambda t: t[2])
else:
    w, acc, test_acc = max(results, key=lambda t: t[1])
np.save('../compute-210x160-%s/w_%s.npy' % (env_id, model_id), w)

print('Best train accuracy:', acc, '(saving model...)')
print('Best val accuracy:', test_acc)
print(' & '.join(map(str, regs)))
print((' | '.join([str(acc * 100) + '%' for w, acc, _ in results])).replace('0% ', 'I '))

"""Get and Eval model"""

import numpy as np
from scipy.linalg import solve
from sklearn.metrics import accuracy_score
import sys
import gym

# parse cli
arguments = sys.argv
model_id = '1_conv'
env_id = 'SpaceInvaders-v0'
if len(arguments) > 1:
    model_id = arguments[1]
if len(arguments) > 2:
    env_id = arguments[2]

env = gym.make(env_id)

X = np.load('compute-210x160-%s/X_%s.npy' % (env_id, model_id))
Y = np.load('compute-210x160-%s/Y_%s.npy' % (env_id, model_id))

X = X.reshape((X.shape[0], -1))
Y = Y.reshape((Y.shape[0], 1))

Y_oh = np.eye(env.action_space.n)[np.ravel(Y).astype(int)]

regs = (1e-7, 1e-5, 1e-3, 1e-1, 1, 1e1, 1e2, 1e3, 1e5)
I = np.eye(X.shape[1])
results = []
for reg in regs:
    try:
        w = solve(X.T.dot(X) + reg*I, X.T.dot(Y_oh))
        acc = accuracy_score(np.argmax(X.dot(w), axis=1), Y)
    except np.linalg.linalg.LinAlgError:
        w = None
        acc = 0
    print('Regularization:', reg, '// Accuracy:', acc)
    results.append((w, acc))
w, acc = max(results, key=lambda t: t[1])
np.save('compute-210x160-%s/w_%s.npy' % (env_id, model_id), w)
print('Best accuracy:', acc, '(saving model...')

print(' & '.join(regs))
print(' | '.join([acc for w, acc in results]))
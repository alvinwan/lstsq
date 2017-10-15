"""Get and Eval model"""

import numpy as np
from scipy.linalg import solve
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr
from scipy.sparse.linalg import lsmr
import sys
import gym
import warnings
import os

warnings.filterwarnings('ignore')

# parse cli
arguments = sys.argv
model_id = '11_blobprost'
env_id = 'SpaceInvaders-v0'
if len(arguments) > 1:
    model_id = arguments[1]
if len(arguments) > 2:
    env_id = arguments[2]

env = gym.make(env_id)

X = np.load('compute-210x160-%s/X_%s.npy' % (env_id, model_id))[()]
Y = np.load('compute-210x160-%s/Y_%s.npy' % (env_id, model_id))[()].todense()

Y_oh = np.eye(env.action_space.n)[np.ravel(Y).astype(int)]

print('X', X.shape, 'Y', Y.shape, 'Y_oh', Y_oh.shape)

regs = (1e-7, 1e-5, 1e-3, 1e-1, 1, 1e1, 1e2, 1e3, 1e5)
I = eye(X.shape[1])
results = []
#print('Converting X to csc_matrix')
#X = csc_matrix(X)
#print('Computing XTX...')
#XTX = X.T.dot(X)
#print('Computing XTY...')
#XTY = X.T.dot(Y_oh)
#print('Finished precomputation...')
for reg in regs:
    try:
        #w = spsolve(XTX + reg*I, XTY)
        print('Trying', reg)
        w = []
        for j in range(env.action_space.n):
            w.append(lsmr(X, Y_oh[:, j])[0])
            print('Solved action %d for %s' % (j, str(reg)))
        w = w.vstack(w).T
        print(w.shape)
        acc = accuracy_score(np.argmax(X.dot(w), axis=1), Y)
    except np.linalg.linalg.LinAlgError:
        w = None
        acc = 0
        istop = 0
    print('Regularization:', reg, '// Accuracy:', acc, '// isTop:', istop)
    results.append((w, acc))
w, acc = max(results, key=lambda t: t[1])
np.save('compute-210x160-%s/w_%s.npy' % (env_id, model_id), w)
print('Best train accuracy:', acc, '(saving model...)')
print(' & '.join(map(str, regs)))
print((' | '.join([str(acc * 100) + '%' for w, acc in results])).replace('0% ', 'I '))

test_path = 'compute-210x160-%s/X_%s_test.npy' % (env_id, model_id)
if os.path.exists(test_path):
    X_test = np.load('compute-210x160-%s/X_%s_test.npy' % (env_id, model_id))
    Y_test = np.load('compute-210x160-%s/Y_%s_test.npy' % (env_id, model_id))
    acc = accuracy_score(np.argmax(X_test.dot(w), axis=1), Y_test)
    print('Validation accuracy:', acc)

import sys
sys.path.insert(0, "..")
from numba import jit
import numpy as np
import time
import opt
import concurrent.futures as fs
import math
from sklearn import metrics
import utils
from utils import misc
from scipy import linalg


@jit(nopython=True, nogil=True)
def _fast_exp(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = math.exp(x[i,j])
    return x

def fast_exp(X, max_threads=70):
    return __parallel_execute__(_fast_exp, X, max_threads)

@jit(nopython=True, nogil=True)
def _fast_rand(X):
    return np.random.randn(*X.shape)

def fast_rand(shape, max_threads=70):
    X = np.zeros(shape)
    return __parallel_execute__(_fast_rand, X, max_threads)

def __parallel_execute__(f, X, max_threads=70):
    '''Parallelize any numba jit-ed (nogil) function on matrices f '''
    idxs = misc.chunk_idxs(X.shape[0], max_threads)
    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for (sidx, eidx) in idxs:
            futures.append(executor.submit(f, X[sidx:eidx, :]))
        fs.wait(futures)
    return results

def softmax_pred(wx):
    wx = fast_exp(wx)
    wx /= (np.sum(wx, axis=1)[:,np.newaxis])
    return wx

def softmax_gn(X_train, y_train, X_test, y_test, XtX, step_size=10, multiplier=1e-8, numiter=50, verbose=True):
        gmat = (1.0/X_train.shape[0])*(XtX)
        lambdav = multiplier
        gmat = gmat + lambdav*np.eye(gmat.shape[0])
        num_classes = int(np.max(y_train) + 1)
        w = np.zeros((num_classes, gmat.shape[0]))
        num_samples = X_train.shape[0]
        y_train_hot = np.eye(num_classes)[y_train.astype('int')]
        for k in range(numiter):
            t = time.time()
            train_preds  = w.dot(X_train.T).T
            e = time.time()
            print("step 1 took {0}".format(e - t))
            print("Compting train_preds 2")
            t = time.time()
            train_preds -= np.max(train_preds, axis=1)[:,np.newaxis]
            e = time.time()
            fast_exp(train_preds)
            train_preds /= (np.sum(train_preds, axis=1)[:,np.newaxis])
            train_preds = y_train_hot - train_preds
            grad = (1.0/num_samples)*(X_train.T.dot(train_preds).T) - lambdav*w
            w += step_size*(linalg.solve(gmat, grad.T, sym_pos=True)).T
            y_train_pred = np.argmax(softmax_pred(w.dot(X_train.T).T), axis=1)
            y_test_pred = np.argmax(softmax_pred(w.dot(X_test.T).T), axis=1)
            train_acc = metrics.accuracy_score(y_train, y_train_pred)
            test_acc = metrics.accuracy_score(y_test, y_test_pred)
            if (verbose):
              print("Iter: {0}, Train Accuracy: {1}, Test Accuracy: {2}".format(k, train_acc, test_acc))
        return y_train_pred, y_test_pred, w

'''
def softmax_kernel_gn(K_train, y_train, K_test, y_test, step_size=1, multiplier=1e-1, numiter=50, verbose=True):
        lambdav = multiplier
        diag_idx = np.diag_indices(K_train.shape[0])
        K_train[diag_idx] += lambdav
        num_classes = int(np.max(y_train) + 1)
        w = np.zeros((K_train.shape[0], num_classes))
        num_samples = K_train.shape[0]
        y_train_hot = np.eye(num_classes)[y_train.astype('int')]
        for k in range(numiter):
            t = time.time()
            print("Computing train_preds 1")
            train_preds  = K_train.dot(w)
            e = time.time()
            print("step 1 took {0}".format(e - t))
            print("Compting train_preds 2")
            t = time.time()
            train_preds -= np.max(train_preds, axis=1)[:,np.newaxis]
            e = time.time()
            fast_exp(train_preds)
            train_preds /= (np.sum(train_preds, axis=1)[:,np.newaxis])
            train_preds = y_train_hot - train_preds
            print("RESIDUAL ", np.linalg.norm(train_preds))
            grad = (1.0/num_samples)*(train_preds) - lambdav*w
            print("Solving...")
            w += step_size*scipy.linalg.solve(K_train, grad, sym_pos=True)
            y_train_pred = np.argmax(K_train.dot(w), axis=1)
            y_test_pred = np.argmax(K_test.dot(w), axis=1)
            train_acc = metrics.accuracy_score(y_train, y_train_pred)
            test_acc = metrics.accuracy_score(y_test, y_test_pred)
            if (verbose):
              print("Iter: {0}, Train Accuracy: {1}, Test Accuracy: {2}".format(k, train_acc, test_acc))
        K_train[diag_idx] -= lambdav
        return y_train_pred, y_test_pred, w
'''

def softmax_block_gn(X_train, y_train, X_test, y_test, multiplier=1e-2, numiter=4,block_size=4000, epochs=10, verbose=True):
        ''' Fix some coordinates '''
        total_features = X_train.shape[1]
        num_blocks = math.ceil(total_features/block_size)
        num_classes = int(np.max(y_train) + 1)
        w = np.zeros((num_classes, X_train.shape[1]))
        num_samples = X_train.shape[0]
        y_train_hot = np.eye(num_classes)[y_train.astype('int')]
        lambdav = multiplier
        for e in range(epochs):
                shuffled_features = np.random.choice(total_features, total_features, replace=False)
                for b in range(int(num_blocks)):
                        block_features = shuffled_features[b*block_size:min((b+1)*block_size, total_features)]
                        X_train_block = X_train[:, block_features]
                        X_test_block = X_test[:, block_features]
                        w_block = w[:, block_features]
                        gmat = (1.0/X_train_block.shape[0])*(X_train_block.T.dot(X_train_block))
                        gmat = gmat + lambdav*np.eye(gmat.shape[0]);
                        w_full = np.zeros((10, X_train.shape[1]))
                        for k in range(numiter):
                            train_preds  = w.dot(X_train.T).T # datapoints x 10
                            train_preds -= np.max(train_preds, axis=1)[:,np.newaxis]
                            train_preds = fast_exp(train_preds)
                            train_preds /= (np.sum(train_preds, axis=1)[:,np.newaxis])
                            train_preds *= -1
                            train_preds += y_train_hot
                            grad = (1.0/num_samples)*(X_train_block.T.dot(train_preds).T) - lambdav*w_block # blocksize x 1
                            w_block = w_block + (np.linalg.solve(gmat, grad.T)).T
                            w[:, block_features] = w_block
                        y_train_pred = np.argmax(w.dot(X_train.T).T, axis=1)
                        y_test_pred = np.argmax(w.dot(X_test.T).T, axis=1)
                        train_acc = metrics.accuracy_score(y_train, y_train_pred)
                        test_acc = metrics.accuracy_score(y_test, y_test_pred)
                        if (verbose):
                                print("Epoch: {0}, Block: {3}, Train Accuracy: {1}, Test Accuracy: {2}".format(e, train_acc, test_acc, b))


'''
def softmax_kernel_block_gn(K_train, y_train, K_test, y_test, step_size=10, multiplier=1e-2, numiter=4,block_size=4000, epochs=10, verbose=True):
        total_features = K_train.shape[1]
        num_blocks = math.ceil(total_features/block_size)
        num_classes = int(np.max(y_train) + 1)
        w = np.zeros((K_train.shape[1], num_classes))
        num_samples = K_train.shape[0]
        y_train_hot = np.eye(num_classes)[y_train.astype('int')]
        lambdav = multiplier
        for e in range(epochs):
                shuffled_features = np.random.choice(total_features, total_features, replace=False)
                for b in range(int(num_blocks)):
                        block_features = shuffled_features[b*block_size:min((b+1)*block_size, total_features)]
                        K_train_block = K_train[:, block_features]
                        w_block = w[block_features, :]
                        Kbb = K_train_block[block_features, :]
                        idxs = np.diag_indices(Kbb.shape[0])
                        Kbb[idxs] += lambdav
                        for k in range(numiter):
                            # Fix this!!
                            y_hat = K_train_block.T.dot(w)
                            print(y_hat.shape)
                            y_hat -= np.max(y_hat, axis=1)[:,np.newaxis]
                            y_hat = fast_exp(y_hat)
                            y_hat /= (np.sum(y_hat, axis=1)[:,np.newaxis])
                            y_hat *= -1
                            y_hat += y_train_hot[block_features]
                            grad = (1.0/num_samples)*(y_hat) - lambdav*w_block # blocksize x 1
                            print(grad.shape)
                            print(Kbb.shape)
                            w_block = w_block + step_size*(scipy.linalg.solve(Kbb, grad, sym_pos=True))
                            w[block_features, :] = w_block
                            y_train_pred = np.argmax(K_train.dot(w), axis=1)
                            y_test_pred = np.argmax(K_test.dot(w), axis=1)
                            train_acc = metrics.accuracy_score(y_train, y_train_pred)
                            test_acc = metrics.accuracy_score(y_test, y_test_pred)
                            print("Epoch: {0}, Block: {3}, Train Accuracy: {1}, Test Accuracy: {2}".format(e, train_acc, test_acc, b))
'''

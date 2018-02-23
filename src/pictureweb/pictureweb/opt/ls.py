import sys
sys.path.append("..")
import scipy.linalg
import sklearn.metrics as metrics
import numpy as np
from numba import jit
import concurrent.futures as fs
from ..utils.misc import chunk_idxs
import math





def evaluateDualModel(kMatrix, model, TOT_FEAT=1):
    kMatrix *= TOT_FEAT
    y = kMatrix.dot(model)
    kMatrix /= TOT_FEAT
    return y


def learnPrimal(trainData, labels, W=None, reg=0.1, XTX=None, XTy=None):
    '''Learn a model from trainData -> labels '''

    trainData = trainData.reshape(trainData.shape[0],-1)
    X = trainData
    n = trainData.shape[0]
    if (W == None):
        W = np.ones(n)[:, np.newaxis]


    if (XTX == None):
        X = np.ascontiguousarray(trainData).reshape(trainData.shape[0], -1)
        print("X SHAPE ", trainData.shape)
        print("Computing XTX")
        XTWX = X.T.dot(X)
        idxes = np.diag_indices(XTWX.shape[0])
        print("Done Computing XTX")
    else:
        idxes = np.diag_indices(XTX.shape[0])
        XTWX = XTX
        if (not np.all(W == np.ones(n)[:, np.newaxis])):
            W = np.ones(n)[:, np.newaxis]
            print("Warning W provided but XTX also provided so ignoring W")

    XTWX[idxes] += reg
    y = np.eye(max(labels) + 1)[labels]
    if (XTy == None):
        XTWy = X.T.dot(W * y)
    else:
        XTWy = XTy
    model = scipy.linalg.solve(XTWX, XTWy)
    XTWX[idxes] -= reg
    return model

def trainAndEvaluateDualModel(KTrain, KTest, labelsTrain, labelsTest, reg=0.1):
    model = learnDual(KTrain,labelsTrain, reg=reg)
    predTrainWeights = evaluateDualModel(KTrain, model)
    predTestWeights = evaluateDualModel(KTest, model)
    labelsTrainPred = np.argmax(predTrainWeights, axis=1)
    labelsTestPred = np.argmax(predTestWeights, axis=1)
    train_acc = metrics.accuracy_score(labelsTrain, labelsTrainPred)
    test_acc = metrics.accuracy_score(labelsTest, labelsTestPred)
    return (train_acc, test_acc, model)

def learnDual(gramMatrix, labels, reg=0.1, TOT_FEAT=1, NUM_TRAIN=1):
    ''' Learn a model from K matrix -> labels '''
    print ("Learning Dual Model")
    y = np.eye(max(labels) + 1)[labels]
    idxes = np.diag_indices(gramMatrix.shape[0])
    gramMatrix /= float(TOT_FEAT)
    print("reg is " + str(reg))
    gramMatrix[idxes] += (reg)
    model = scipy.linalg.solve(gramMatrix, y, sym_pos=True)
    gramMatrix[idxes] -= (reg)
    gramMatrix *= TOT_FEAT
    return model


def trainAndEvaluatePrimalModel(XTrain, XTest, labelsTrain, labelsTest, reg=0.0, W=None, XTX=None, XTy=None):
    model = learnPrimal(XTrain, labelsTrain, reg=reg, W=W, XTX=XTX, XTy=XTy)
    yTrainHat = XTrain.dot(model)[:,1]
    yTestHat = XTest.dot(model)[:,1]

    yTrainPred = np.argmax(XTrain.dot(model), axis=1)
    yTestPred = np.argmax(XTest.dot(model), axis=1)

    train_acc = metrics.accuracy_score(yTrainPred, labelsTrain)
    test_acc = metrics.accuracy_score(yTestPred, labelsTest)
    return (train_acc, test_acc, model)


def top_k_accuracy(labels, y_pred, k=5):
    top_k_preds = get_top_k(y_pred, k=k)
    if (len(labels.shape) == 1):
        labels = labels[:, np.newaxis]
    correct = np.sum(np.any(top_k_preds == labels, axis=1))
    return correct/float(labels.shape[0])

def top_5_acc(labels, y_pred):
    top_5 = []
    y_pred = y_pred.copy()
    for i in range(5):
        top = np.argmax(y_pred, axis=1)
        top_5.append(top)
        y_pred[:, top] = float('-inf')
    top_5 = np.vstack(top_5).T
    return np.sum(np.any(top_5 == labels, axis=1))/float(labels.shape[0])

def top_k_accuracy(labels, y_pred, k=5):
    top_k_preds = get_top_k(y_pred, k=k)
    if (len(labels.shape) == 1):
        labels = labels[:, np.newaxis]
    correct = np.sum(np.any(top_k_preds == labels, axis=1))
    return correct/float(labels.shape[0])

def get_top_k(y_pred, k=5, threads=70):
    with fs.ThreadPoolExecutor(max_workers=threads) as executor:
        idxs = chunk_idxs(y_pred.shape[0], threads)
        futures = []
        for (sidx, eidx) in idxs:
            futures.append(executor.submit(_get_top_k, y_pred[sidx:eidx, :], k))
        fs.wait(futures)
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    return results

@jit(nopython=True, nogil=True)
def _get_top_k(y_pred, k=5):
    top_k_preds = np.ones((y_pred.shape[0], k))
    top_k_pred_weights = np.ones((y_pred.shape[0], k))

    top_k_pred_weights *= -99999999
    for i in range(y_pred.shape[0]):
        top_k = top_k_preds[i, :]
        top_k_pred_weights_curr = top_k_pred_weights[i, :]
        for j in range(y_pred.shape[1]):
            in_top_k = False
            for elem in top_k_pred_weights_curr:
                in_top_k = in_top_k | (y_pred[i,j] > elem)
            if (in_top_k):
                min_idx = 0
                for z in range(top_k_pred_weights_curr.shape[0]):
                    if top_k_pred_weights_curr[min_idx] > top_k_pred_weights_curr[z]:
                        min_idx = z
                top_k[min_idx] = j
                top_k_pred_weights_curr[min_idx] = y_pred[i,j]

    return top_k_preds

def block_kernel_solve(K, y, block_size=4000, epochs=1, lambdav=0.1, verbose=True, prc=lambda x: x, eval_fn=None):
        '''Solve (K + \lambdaI)x = y
            in a block-wise fashion
        '''

        # compute some constants
        num_samples = K.shape[0]
        num_blocks = math.ceil(num_samples/block_size)
        x = np.zeros((K.shape[0], y.shape[1]))
        loss = 0
        for e in range(epochs):
                shuffled_coords = np.random.choice(num_samples, num_samples, replace=False)
                for b in range(int(num_blocks)):
                        # pick a block
                        block = shuffled_coords[b*block_size:min((b+1)*block_size, num_samples)]

                        # pick a subset of the kernel matrix (note K can be mmap-ed)
                        K_block = prc(K[:, block])
                        y_block = prc(y[block, :])

                        # This is a matrix vector multiply very efficient can be parallelized
                        # (even if K is mmaped)

                        # calculate
                        R = np.zeros(y_block.shape)
                        for b2 in range(int(num_blocks)):
                            if b2 == b: continue
                            block_b2 = shuffled_coords[b2*block_size:min((b2+1)*block_size, num_samples)]
                            R += K_block[block_b2, :].T.dot(x[block_b2, :])


                        Kbb = K_block[block, :]
                        print(R)

                        print(Kbb.shape)
                        print(y_block - R)
                        # Add term to regularizer
                        idxes = np.diag_indices(Kbb.shape[0])

                        Kbb[idxes] += lambdav
                        print(lambdav)
                        print("solving block {0}".format(b))
                        x_block = scipy.linalg.solve(Kbb, y_block - R, sym_pos=True)
                        Kbb[idxes] -= lambdav
                        # update model
                        x[block] = x_block
                        if (verbose):
                            status = "Epoch: {0}, Block: {1}, EvalFnOutput {2}".format(e, b, eval_fn(x))
                            print(status)
        return x



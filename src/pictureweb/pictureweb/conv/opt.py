from scipy import linalg
import sklearn.metrics as metrics
import numpy as np
from numba import jit
import concurrent.futures as fs


def evaluateDualModel(kMatrix, model, TOT_FEAT=1):
    kMatrix *= TOT_FEAT
    y = kMatrix.dot(model)
    kMatrix /= TOT_FEAT
    return y


def learnPrimal(trainData, labels, W=None, reg=0.1, XTX=None):
    '''Learn a model from trainData -> labels '''

    trainData = trainData.reshape(trainData.shape[0],-1)
    n = trainData.shape[0]
    X = np.ascontiguousarray(trainData, dtype=np.float32).reshape(trainData.shape[0], -1)
    if (W == None):
        W = np.ones(n)[:, np.newaxis]

    if (XTX == None):
        print("X SHAPE ", trainData.shape)
        print("Computing XTX")
        sqrtW = np.sqrt(W)
        X *= sqrtW
        XTWX = X.T.dot(X)
        print("Done Computing XTX")
        idxes = np.diag_indices(XTWX.shape[0])
    else:
        XTWX = XTX
        if (not np.all(W == np.ones(n)[:, np.newaxis])):
            W = np.ones(n)[:, np.newaxis]
            print("Warning W provided but XTX also provided so ignoring W")

    XTWX[idxes] += reg
    y = np.eye(max(labels) + 1)[labels]
    XTWy = X.T.dot(W * y)
    model = linalg.solve(XTWX, XTWy)
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
    return (train_acc, test_acc)

def learnDual(gramMatrix, labels, reg=0.1, TOT_FEAT=1, NUM_TRAIN=1):
    ''' Learn a model from K matrix -> labels '''
    print ("Learning Dual Model")
    y = np.eye(max(labels) + 1)[labels]
    idxes = np.diag_indices(gramMatrix.shape[0])
    gramMatrix /= float(TOT_FEAT)
    print("reg is " + str(reg))
    gramMatrix[idxes] += (reg)
    model = linalg.solve(gramMatrix, y)
    gramMatrix[idxes] -= (reg)
    gramMatrix *= TOT_FEAT
    return model


def trainAndEvaluatePrimalModel(XTrain, XTest, labelsTrain, labelsTest, reg=0.0, W=None, XTX=None):
    model = learnPrimal(XTrain, labelsTrain, reg=reg, W=W, XTX=XTX)
    yTrainHat = XTrain.dot(model)[:,1]
    yTestHat = XTest.dot(model)[:,1]

    yTrainPred = np.argmax(XTrain.dot(model), axis=1)
    yTestPred = np.argmax(XTest.dot(model), axis=1)

    train_acc = metrics.accuracy_score(yTrainPred, labelsTrain)
    test_acc = metrics.accuracy_score(yTestPred, labelsTest)
    return (train_acc, test_acc)

def chunk_idxs(size, chunks):
    chunk_size  = int(np.ceil(size/chunks))
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))

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

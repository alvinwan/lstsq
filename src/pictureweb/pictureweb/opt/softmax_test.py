import numpy as np
from sklearn import metrics
import softmax

@profile
def softmax_gn(X_train, y_train, X_test, y_test, XTX, w_init=None, step_size=1, multiplier=1e-2, num_classes=1000, numiter=50, verbose=True):
        ''' Implementation of gauss-newton quassi-newton optimization algorithm
            with softmax objective
        '''
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        gmat = (1.0/X_train.shape[0])*XTX
        lambdav = multiplier*np.trace(gmat)/gmat.shape[0]
        num_idx = np.diag_indices(gmat.shape[0])
        gmat[num_idx] += lambdav
        w = np.zeros((num_classes, gmat.shape[0]))
        if (w_init != None):
            w = w_init
        num_samples = X_train.shape[0]
        onehot = lambda x: np.eye(num_classes)[x]
        class_matrix = np.eye(num_classes)
        y_train_hot = class_matrix[y_train]
        y_test_hot = class_matrix[y_test]
        for k in range(numiter):
            print("Computing Trian Preds")
            train_preds  = w.dot(X_train.T).T # 1million x 1000
            train_acc = metrics.accuracy_score(y_train, np.argmax(train_preds, axis=1))
            print("TRAIN ACC ", train_acc)

            print("Normalizing trian preds")
            train_preds = train_preds - np.max(train_preds, axis=1)[:,np.newaxis]
            train_preds = softmax.fast_exp(train_preds)
            train_preds = train_preds/(np.sum(train_preds, axis=1)[:,np.newaxis])
            train_preds = y_train_hot - train_preds

            print("Computing gradient")
            grad = (1.0/num_samples)*(X_train.T.dot(train_preds).T) - lambdav*w

            print("Solving for step")
            w = w + step_size * (np.linalg.solve(gmat, grad.T, sym_pos=True)).T

        return w


alexnet_train = np.load("/data/vaishaal/pictureweb/pywren_kernels/imagenet_train_alexnet_fc7.npz")
X_train = alexnet_train["X_train"]
y_train = alexnet_train["y_train"]
X_test = np.load("/data/vaishaal/pictureweb/pywren_kernels/imagenet_test_features_alexnet_fc7.npy")
y_test  = np.load("/data/vaishaal/pictureweb/pywren_kernels/imagenet_test_labels.npy")
XtX = X_train.T.dot(X_train)

softmax_gn(X_train,y_train,X_test,y_test, XtX, multiplier=1e-3, step_size=10, numiter=3)

import os
import os.path
import numpy as np
import time
import threading

from utils import run_threads
from utils import one_hot


N_ACTIONS = 6
D=7056
t0=time.time()
DIR = 'raw-atari-unpacked'
SAVE_DIR = 'raw-atari-precompute'

global_xtx = np.zeros((D, D))
global_xtx_set = False

XTX_THREAD_PARTIAL = 'raw-atari-pca-xtx-%d-%d'
XTX_THREAD = 'raw-atari-pca-xtx-%d'
XTX_FINAL = 'raw-atari-pca-xtx-final.npy'
V_N_DIMS = 'raw-atari-pca-v%d.npy'

global_pxtpx = np.zeros((P, P))
global_pxty = np.zeros((P, N_ACTIONS))

PXTPX_THREAD_PARTIAL = 'raw-atari-pca-pxtpx-%d-%d'
PXTY_THREAD_PARTIAL = 'raw-atari-pca-pxty-%d-%d'
PXTPX_THREAD = 'raw-atari-pca-pxtpx-%d'
PXTY_THREAD = 'raw-atari-pca-pxty-%d'
PXTPX_FINAL = 'raw-atari-pca-pxtpx-final.npy'
PXTY_FINAL = 'raw-atari-pca-pxty-final.npy'

def run_1():
    """
    Step 1 - precompute
    """
    global global_xtx, global_xtx_set, t0
    t0 = time.time()
    run_threads(run_1_thread)
    np.save(os.path.join(SAVE_DIR, XTX_FINAL), global_xtx)
    global_xtx_set = True


def run_1_thread(thread_id, paths, start):
    """
    bar{X}^T\bar{X} = (X-u)^T(X-u) = \sum_i (x_i-u_i)^T(x_i-u_i)
    """
    xtx = np.zeros((D,D))
    for i, path in enumerate(paths, start=start):
        if i % 100 == 0:
            print('[%d] Iteration: %d . Average(s): %.3f' % (thread_id, i, (time.time() - start) / float(i)))
        if i%4000 == 0:
            np.save(os.path.join(SAVE_DIR, XTX_THREAD_PARTIAL % (thread_id, i)), xtx)
        di = np.load(os.path.join(DIR, path))
        x, y = di[:,:D], di[:,-2].reshape((-1, 1))
        u = np.mean(x, axis=0)
        bar_x = x - u
        xtx += bar_x.T.dot(bar_x)
    print('[%d] Total time: %.3f' % (thread_id, time.time() - t0))
    np.save(os.path.join(SAVE_DIR, XTX_THREAD % thread_id), xtx)

    global global_xtx
    global_xtx += xtx


def run_2(dims=range(15,20)):
    """
    Step 2
    Run svd on \bar{X}^T\bar{X} to get V
    where \bar{X} = X-u
    """
    global global_xtx, global_xtx_set
    XTX = global_xtx if global_xtx_set else np.load(os.path.join(SAVE_DIR, XTX_FINAL))
    lambdas, vs = np.linalg.eig(XTX)
    for i in dims:
       codomain = vs[:,:i]
       np.save(os.path.join(SAVE_DIR, V_N_DIMS % i), codomain)


def run_3(n_dims=15):
    """
    Step 3
    """
    global global_pxtpx, t0
    t0 = time.time()
    V = np.load(os.path.join(SAVE_DIR, V_N_DIMS % i))
    run_threads(run_3_thread, extra_args=(V,))
    np.save(os.path.join(SAVE_DIR, PXTPX_FINAL), global_pxtpx)


def run_3_thread(thread_id, paths, start, V):
    """
    Compute projected
    \tilde{X}^T\tilde{X} = \sum_i \tilde{X}_i^T\tilde{X}_i
    \tilde{X}^Ty = \sum_i \tilde{X}_i^Ty
    where \bar{X} = X-u, \tilde{X} = \phi(X) = \bar{X}V
    """
    pxtpx = np.zeros((P,P))
    pxty = np.zeros((P, N_ACTIONS))
    for i, path in enumerate(paths, start=start):
        if i % 100 == 0:
            print('[%d] Iteration: %d . Average(s): %.3f' % (thread_id, i, (time.time() - start) / float(i)))
        if i % 4000 == 0:
            np.save(os.path.join(SAVE_DIR, PXTPX_THREAD_PARTIAL % (thread_id, i)), pxtpx)
            np.save(os.path.join(SAVE_DIR, PXTY_THREAD_PARTIAL % (thread_id, i)), pxty)
        di = np.load(os.path.join(DIR, path))
        x, y = di[:,:D], di[:,-2].reshape((-1, 1))
        u = np.mean(x, axis=0)
        bar_x = x - u
        phi_x = bar_x.dot(V)
        pxtpx += phi_x.T.phi_x
        pxty += phi_x.T.dot(one_hot(y))

    print('[%d] Total time: %.3f' % (thread_id, time.time() - t0))
    np.save(os.path.join(SAVE_DIR, PXTPX_THREAD % thread_id), pxtpx)
    np.save(os.path.join(SAVE_DIR, PXTY_THREAD % thread_id), pxty)

    global global_pxtpx, global_pxty
    global_pxtpx += pxtpx
    global_pxty += pxty

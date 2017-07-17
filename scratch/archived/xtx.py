import os
import os.path
import numpy as np
import time
import threading

t0 = time.time()

import sys

arguments = sys.argv

assert len(arguments) == 2, 'Requires number of episodes to use'

N_THREADS = 48
N_ACTIONS = 6
N = int(arguments[1])
D=512
start = time.time()
DIR = 'fc5'
SAVE_DIR = 'fc5-precompute'
global_xtx = np.zeros((D, D))
global_xty = np.zeros((D, N_ACTIONS))

XTX_THREAD = 'fc5-xtx-%d-%d'
XTX_FINAL = 'fc5-xtx-%d-final'

XTY_THREAD = 'fc5-xty-%d-%d'
XTY_FINAL = 'fc5-xty-%d-final'

def one_hot(y):
    Y = np.eye(N_ACTIONS)[y.astype(int)]
    return Y.reshape((Y.shape[0], Y.shape[2]))

def run(thread_id, paths, start):
    xtx = np.zeros((D,D))
    xty = np.zeros((D, N_ACTIONS))
    t2 = time.time()
    for i, path in enumerate(paths, start=start):
        if i % 100 == 0:
            print(i, time.time() - t2)
            t2 = time.time()
        full_path = os.path.join(DIR, path)
        di = np.load(full_path)
        x, y = di[:,:D], di[:,-2].reshape((-1, 1))
        xty += x.T.dot(one_hot(y))
        xtx += x.T.dot(x)
    print(time.time() - t0)
    np.save(os.path.join(SAVE_DIR, XTX_THREAD % (N, thread_id)), xtx)
    np.save(os.path.join(SAVE_DIR, XTY_THREAD % (N, thread_id)), xty)
    print('saving took', time.time() - t2)
    global global_xtx , global_xty
    global_xtx += xtx
    global_xty += xty


def main():
    paths = list(os.listdir(DIR))[:N]
    n_paths_per_thread = int(np.ceil(len(paths) / N_THREADS))
    print('Number threads:', N_THREADS)
    print('Paths per thread:', n_paths_per_thread)
    #   run(0, paths, 0)
    #   return
    threads = []
    for i in range(N_THREADS):
        thread_paths = paths[i*n_paths_per_thread: (i+1)*n_paths_per_thread]
        thread = threading.Thread(target=run, args=(i, thread_paths, i*n_paths_per_thread))
        thread.start()
        threads.append(thread)
        if i % 14 == 0:
            print(i, 'threads started')
    print(N_THREADS, 'threads started')
    for i in range(N_THREADS):
        threads[i].join()
    global global_xtx, global_xty
    np.save(os.path.join(SAVE_DIR, XTX_FINAL % N), global_xtx)
    np.save(os.path.join(SAVE_DIR, XTY_FINAL % N), global_xty)

if __name__ == '__main__':
   main()

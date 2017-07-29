"""

xtx.py 1000 : Run on 1000 regular atari episodes
xtx.py 1000 9000 : Run on 1000 regular, 9000 dagger episodes
xtx.py restore <xtx path> <xty path> : Initiatlize xtx, xty to paths
"""

import os
import os.path
import numpy as np
import time
import threading
import glob

t0 = time.time()

import sys

arguments = sys.argv

assert len(arguments) >= 2, 'Requires number of episodes to use'

assert len(arguments) < 3 or 'dagger' in arguments, 'Gotta add the "dagger" keyword bruh! i.e., bash script.sh 1000 9000 dagger, currently: %s' % ' '.join(arguments)

DAGGER = 'dagger' in arguments
DAGGER_DIR = 'spaceinvaders-dagger'
DAGGER2_DIR = 'spaceinvaders-dagger2'
N_THREADS = 48
N_ACTIONS = 6
N = int(arguments[1])
N_d = int(arguments[2]) if len(arguments) >= 3 and arguments[2].isnumeric() else N
D=512
start = time.time()
LAYER = 'prelu'
DIR = 'spaceinvaders-%s' % LAYER
SAVE_DIR = 'spaceinvaders-precompute'

restore = None

iterator = iter(arguments)
if 'restore' in iterator:
    global_xtx = np.load(next(iterator))
    global_xty = np.load(next(iterator))
else:
    global_xtx = np.zeros((D, D))
    global_xty = np.zeros((D, N_ACTIONS))

XTX_THREAD = LAYER + '-xtx-%d-%d'
XTX_FINAL = LAYER + '-xtx-%d-final'

XTY_THREAD = LAYER + '-xty-%d-%d'
XTY_FINAL = LAYER + '-xty-%d-final'

if DAGGER:
    suffix = '-dagger-%d' % N_d
    XTX_THREAD += suffix
    XTX_FINAL += suffix
    XTY_THREAD += suffix
    XTY_FINAL += suffix

def one_hot(y):
    Y = np.eye(N_ACTIONS)[y.astype(int)]
    return Y.reshape((Y.shape[0], Y.shape[2]))

def run(thread_id, paths, start):
    xtx = np.zeros((D,D))
    xty = np.zeros((D, N_ACTIONS))
    t2 = time.time()
    for i, path in enumerate(paths, start=start):
        if i % 100 == 0 and i > 0:
            print(i, time.time() - t2)
            t2 = time.time()
        full_path = os.path.join(DIR, path)
        if not os.path.exists(full_path):
             full_path = path
        di = np.load(full_path)
        x, y = di[:,:D], di[:,-2].reshape((-1, 1))
        xty += x.T.dot(one_hot(y))
        xtx += x.T.dot(x)
    if len(paths) > 200:
       print(time.time() - t0)
       np.save(os.path.join(SAVE_DIR, XTX_THREAD % (N, thread_id)), xtx)
       np.save(os.path.join(SAVE_DIR, XTY_THREAD % (N, thread_id)), xty)
       print('saving took', time.time() - t2)
    global global_xtx , global_xty
    global_xtx += xtx
    global_xty += xty


def main():
    paths = list(os.listdir(DIR))[:N]
    print('Num normal episodes used for training:', N)
    if DAGGER:
        print('Num dagger episodes use for training:', N_d)
        print('DAGGAH!!')
        dagger_paths = list(glob.glob('spaceinvaders-dagger/*/*.npy'))[:N]
        if len(dagger_paths) < N:
            print('NOT ENOUGH DAGGER PATHS!', len(dagger_paths))
        paths += dagger_paths
    if DAGGER_NEW:

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
    print(N_THREADS, 'threads started')
    for i in range(N_THREADS):
        threads[i].join()
    global global_xtx, global_xty
    global_xtx_path = os.path.join(SAVE_DIR, XTX_FINAL % N)
    global_xty_path = os.path.join(SAVE_DIR, XTY_FINAL % N)
    print('Saved to', global_xtx_path, 'and', global_xty_path)
    np.save(global_xtx_path, global_xtx)
    np.save(global_xty_path, global_xty)

if __name__ == '__main__':
   main()

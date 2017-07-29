"""

Usage:
    dagger.py <i>
    dagger.py <n_atari> <n_dagger> <i>
"""

import os
import os.path
import numpy as np
import time
import threading
import glob
import argparse
import sys

#############
# XTX & XTY #
#############

D = 512
N_ACTIONS = 6

global_xtx = np.zeros((D, D))
global_xty = np.zeros((D, N_ACTIONS))

def one_hot(y):
    Y = np.eye(N_ACTIONS)[y.astype(int)]
    return Y.reshape((Y.shape[0], Y.shape[2]))


def run(thread_id, paths, start):
    print('Thread', thread_id, 'started')
    xtx, xty = np.zeros((D,D)), np.zeros((D, N_ACTIONS))
    for i, di in enumerate(map(np.load, paths), start=start):
        if i % 100 == 0 and i > 0:
            print(i)
        x, y = di[:,:D], di[:,-2].reshape((-1, 1))
        xty += x.T.dot(one_hot(y))
        xtx += x.T.dot(x)
    print('Thread', thread_id, 'finished')
    global global_xtx , global_xty
    global_xtx += xtx
    global_xty += xty


def xtx_and_xty(
        idx_dagger_run=0,
        dagger_gameplay_dir='prelu-dagger-spaceinvaders',
        atari_gameplay_dir='prelu-atari-spaceinvaders',
        save_dir='spaceinvaders-precompute',
        n_threads=48,
        n_atari=1000,
        n_dagger=1000,
        xtx_filename='xtx',
        xty_filename='xty'):
    if idx_dagger_run == 0:
        paths = list(glob.glob('%s/*.npy' % atari_gameplay_dir))[:n_atari]
        print('Num normal episodes used for training:', n_atari)
    print('Num dagger episodes use for training:', n_dagger)
    print('Dagger run training index:', idx_dagger_run)
    dagger_paths = dp = list(glob.glob('%s/%d/*.npy' % (dagger_gameplay_dir, idx_dagger_run)))[:n_dagger]
    assert len(dp) >= n_dagger, 'NOT ENOUGH DAGGER PATHS! %d' % len(dp)
    paths += dp

    num_thread_paths = ntp = int(np.ceil(len(paths) / n_threads))
    threads = []
    for i in range(n_threads):
        thread = threading.Thread(target=run, args=(i, paths[i*ntp: (i+1)*ntp], i*ntp))
        thread.start()
        threads.append(thread)
    for i in range(n_threads):
        threads[i].join()

    global global_xtx, global_xty
    global_xtx_path = os.path.join(save_dir, xtx_filename)
    global_xty_path = os.path.join(save_dir, xty_filename)
    np.save(global_xtx_path, global_xtx)
    np.save(global_xty_path, global_xty)
    print('Saved to', global_xtx_path, 'and', global_xty_path)


#####
# W #
#####


def w(
    save_dir='spaceinvaders-precompute',
    xtx_filename='xtx',
    xty_filename='xty',
    w_filename='w'):

    xtx = np.load(os.path.join(save_dir, xtx_filename))
    xty = np.load(os.path.join(save_dir, xty_filename))

    w = np.linalg.pinv(xtx).dot(xty)
    w_path = os.path.join(save_dir, w_filename)
    np.save(w_path, w)
    print('Saved to', w_path)


def main(n_atari=1000, n_dagger=1000, idx_dagger_run=0):
    xtx_filename = 'xtx-%d.npy' % idx_dagger_run
    xty_filename = 'xty-%d.npy' % idx_dagger_run
    w_filename = 'w-%d.npy' % idx_dagger_run

    xtx_and_xty(
        n_atari=n_atari,
        n_dagger=n_dagger,
        xtx_filename=xtx_filename,
        xty_filename=xty_filename)
    w(
        xtx_filename=xtx_filename,
        xty_filename=xty_filename,
        w_filename=w_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('i', type=int, help='index of dagger run')
    parser.add_argument('--n_atari', type=int, help='Number of atari episodes to train on', default=None)
    parser.add_argument('--n_dagger', type=int, help='Number of dagger episodes to train on', default=None)
    args = parser.parse_args()

    if args.n_atari is not None and args.n_dagger is not None:
        main(
            idx_dagger_run=args.i,
            n_atari=args.n_atari,
            n_dagger=args.n_dagger)
    else:
        main(idx_dagger_run=args.i)

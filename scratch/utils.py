import numpy as np
import threading


def one_hot(y):
   Y = np.eye(N_ACTIONS)[y.astype(int)]
   return Y.reshape((Y.shape[0], Y.shape[2]))


def run_threads(target, dirname='raw-atari-unpacked', n_threads=72, extra_args=()):
    paths = list(os.listdir(dirname))
    n_paths_per_thread = int(np.ceil(len(paths) / n_threads))
    print('Number threads:', n_threads)
    print('Paths per thread:', n_paths_per_thread)
    threads = []
    for i in range(n_threads):
        thread_paths = paths[i*n_paths_per_thread: (i+1)*n_paths_per_thread]
        thread = threading.Thread(target=run, args=(i, thread_paths, i*n_paths_per_thread) + extra_args)
        thread.start()
        threads.append(thread)
        if i % 14 == 0:
            print(i, 'threads started')
    print(n_threads, 'threads started')
    for i in range(n_threads):
      threads[i].join()

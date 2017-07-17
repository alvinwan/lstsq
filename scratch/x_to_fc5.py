import os
import os.path
import numpy as np
import time
import threading

N_THREADS = 48
N_ACTIONS = 6
D=7056
start = time.time()
DIR = 'raw'
SAVE_DIR = 'fc5'

from dqn import *
session_config_kwargs = {
    'gpu_options': tf.GPUOptions(per_process_gpu_memory_fraction=0.5)}


def one_hot(y):
   Y = np.eye(N_ACTIONS)[y.astype(int)]
   return Y.reshape((Y.shape[0], Y.shape[2]))

def run(thread_id, paths, start, *dqn):
   for i, path in enumerate(paths, start=start):
      full_new_path = os.path.join(SAVE_DIR, path)
      if i % 100 == 0:
         print(i, time.time() - t0)
      if os.path.exists(full_new_path):
          continue
      full_path = os.path.join(DIR, path)
      di = np.load(full_path)
      x, y, r = di[:,:D], di[:,-2].reshape((-1, 1)), di[:,-1].reshape((-1, 1))
    #   import pdb; pdb.set_trace()

      x = x.reshape((-1, 84, 84, 1))
      x0 = x[:-3]
      x1 = x[1:-2]
      x2 = x[2:-1]
      x3 = x[3:]
      input_x = np.concatenate((x0, x1, x2, x3), axis=3)
      fc5 = x_to_fc5(input_x, *dqn)
    #   import pdb; pdb.set_trace()

      new_di = np.concatenate((fc5, y[:-3], r[:-3]), axis=1)
      np.save(full_new_path, new_di)
   print(time.time() - t0)


def main():
   dqn = get_dqn(session_config_kwargs=session_config_kwargs)
   paths = list(os.listdir(DIR))
   n_paths_per_thread = int(np.ceil(len(paths) / N_THREADS))
   print('Number threads:', N_THREADS)
   print('Paths per thread:', n_paths_per_thread)
   # run(0, paths, 0, *dqn)
   # return
   threads = []
   for i in range(N_THREADS):
      thread_paths = paths[i*n_paths_per_thread: (i+1)*n_paths_per_thread]
      thread = threading.Thread(target=run, args=(i, thread_paths, i*n_paths_per_thread, *dqn))
      thread.start()
      threads.append(thread)
      if i % 14 == 0:
         print(i, 'threads started')
   print(N_THREADS, 'threads started')
   for i in range(N_THREADS):
      threads[i].join()

if __name__ == '__main__':
   main()

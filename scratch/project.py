import os
import os.path
import numpy as np
import time
import threading


N_THREADS = 72
D=7056
start = time.time()
DIR = 'raw-atari-unpacked'
SAVE_DIR = 'raw-atari-precompute'

P = 15
global_xtvs = []
global_xvtxv = np.zeros((P, P))
global_xvty = np.zeros((P, 1))

V = np.load(os.path.join(SAVE_DIR, 'raw-atari-v%d.npy' % P)).astype(np.float64)

def run(thread_id, paths, start):
   xtvs =[]
   xvtxv = np.zeros((P, P))
   xvty = np.zeros((P, 1))
   n_corrupted = 0
   for i, path in enumerate(paths, start=start):
      if i % 100 == 0:
         print(i, n_corrupted, time.time() - start)
      if i%4000 == 0:
         t2 = time.time()
         if xtvs:
            np.save(os.path.join(SAVE_DIR, 'raw-atari-x%d-%d-%d' % (P, thread_id, i)), np.vstack((xtvs)))
         np.save(os.path.join(SAVE_DIR, 'raw-atari-xtx%d-%d-%d' % (P, thread_id, i)), xvtxv)
         np.save(os.path.join(SAVE_DIR, 'raw-atari-xty%d-%d-%d' % (P, thread_id, i)), xvty)
         print('saved', i, 'in', time.time()-t2)
      full_path = os.path.join(DIR, path)
      try:
          di = np.load(full_path)
          x, y = di[:,:D], di[:,-1].reshape((-1, 1))
          xv = x.dot(V)
          xtvs.append(x.dot(V))
          xvtxv += xv.T.dot(xv)
          xvty += xv.T.dot(y)
 #         np.load(full_path)
      except Exception as e:
          n_corrupted += 1
          print(e)
   print(n_corrupted)
   print(time.time() - start)
   t2 = time.time()
   xtv = np.vstack(xtvs)
   np.save(os.path.join(SAVE_DIR, 'raw-atari-x%d-%d' % (P, thread_id)), xtv)
   np.save(os.path.join(SAVE_DIR, 'raw-atari-xtx%d-%d' % (P, thread_id)), xvtxv)
   np.save(os.path.join(SAVE_DIR, 'raw-atari-xty%d-%d' % (P, thread_id)), xvty)
   print('saving took', time.time() - t2)
   global global_xtvs, global_xvtxv, global_xvty
   global_xtvs.append(xtv)
   global_xvtxv += xvtxv
   global_xvty += xvty


def main():
   paths = list(os.listdir(DIR))
   n_paths_per_thread = int(np.ceil(len(paths) / N_THREADS))
   print('Number threads:', N_THREADS)
   print('Paths per thread:', n_paths_per_thread)
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
   global global_xtvs, global_xvtxv, global_xvty
   global_xtv = np.vstack(global_xtvs)
   np.save(os.path.join(SAVE_DIR, 'raw-atari-xtv-final'), global_xtv)
   np.save(os.path.join(SAVE_DIR, 'raw-atari-xvtxv-final'), global_xvtxv)
   np.save(os.path.join(SAVE_DIR, 'raw-atari-xvty-final'), global_xvty)

if __name__ == '__main__':
   main()

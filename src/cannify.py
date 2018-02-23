from featurization.canny import featurize_density
import numpy as np
import glob
from multiprocessing import Pool
from math import ceil

src = '/data/alvin/lstsq/state-210x160-Centipede-v0/*999_*.npy'
dest = '/data/alvin/lstsq/compute-210x160-Centipede-v0/%s_%s_canny%s.npy'

B_list = []
Y_list = []
times = []
batch_size = 20
num_threads = 100
for path in glob.iglob(src):
    A_src = np.load(path)
    A = A_src[:,:-2].reshape((-1, 250, 160, 3))
    Y_list.append(A_src[:,-2:-1])

    p = Pool(num_threads)
    chunk_size = batch_size * num_threads
    for i in range(int(ceil(A.shape[0] / chunk_size))):
        print(i * chunk_size)
        B_list.extend(p.map(featurize_density, A[i*chunk_size:(i+1)*chunk_size]))
    assert len(B_list) >= A.shape[0]

B = np.array(B_list)
Y = np.vstack(Y_list)
n = len(B)
n_train = int(n * 0.9)

np.save(dest % ('X', str(n), ''), B[:n_train])
np.save(dest % ('Y', str(n), ''), Y[:n_train])
np.save(dest % ('X', str(n), '_test'), B[n_train:])
np.save(dest % ('Y', str(n), '_test'), Y[n_train:])
print('Saving to', dest % ('X', str(n), ''))

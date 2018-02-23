from featurization.canny import featurize_density
import numpy as np
import time
import glob

src = '/data/alvin/lstsq/state-210x160-Centipede-v0/*999*.npy'
dest = '/data/alvin/lstsq/compute-210x160-Centipede-v0/%s_%s_canny%s.npy'

B_list = []
Y_list = []
times = []
for path in glob.iglob(src):
    A_src = np.load(path)
    A = A_src[:,:-2].reshape((-1, 210, 160, 3))
    Y_list.append(A_src[:,-2:-1])

    for i, a in enumerate(A):
        t0 = time.time()  # each iter takes 9ms
        if i % 100 == 0:
            print(i, np.mean(times))
        B_list.append(featurize_density(a))
        times.append(time.time() - t0)

B = np.array(B_list)
Y = np.vstack(Y_list)
n = len(B)
n_train = int(n * 0.9)

np.save(dest % ('X', str(n), ''), B[:n_train])
np.save(dest % ('Y', str(n), ''), Y[:n_train])
np.save(dest % ('X', str(n), '_test'), B[n_train:])
np.save(dest % ('Y', str(n), '_test'), Y[n_train:])
print('Saving to', dest % ('X', str(n), ''))

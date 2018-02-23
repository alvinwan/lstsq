from featurization.canny import featurize_density
import numpy as np
import time

src = '/data/alvin/lstsq/state-210x160-SpaceInvaders-v0/00235_01170_0.npy'
dest = '/data/alvin/lstsq/compute-210x160-SpaceInvaders-v0/%s_%s_canny%s.npy'

A_src = np.load(src)
A = A_src[:,:-2].reshape((-1, 210, 160, 3))
Y = A_src[:,-2:-1]

B_list = []
times = []
for i, a in enumerate(A):
    t0 = time.time()  # each iter takes 9ms
    if i % 100 == 0:
        print(i, np.mean(times))
    B_list.append(featurize_density(a))
    times.append(time.time() - t0)
B = np.array(B_list)
n = len(B)
n_train = int(n * 0.9)

np.save(dest % ('X', str(n), ''), B[:n_train])
np.save(dest % ('Y', str(n), ''), Y[:n_train])
np.save(dest % ('X', str(n), '_test'), B[n_train:])
np.save(dest % ('Y', str(n), '_test'), Y[n_train:])
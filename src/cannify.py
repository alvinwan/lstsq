from featurization.canny import featurize
import numpy as np
import time


src = '/data/alvin/lstsq/state-210x160-SpaceInvaders-v0/00235_01170_0.npy'
dest = '/data/alvin/lstsq/compute-210x160-SpaceInvaders-v0/%s_%s_canny.npy'

A_src = np.load(src)
A = A_src[:,:-2].reshape((-1, 210, 160, 3))
Y = A_src[:,-2:]

B_list = []
times = []
for i, a in enumerate(A):
    t0 = time.time()
    if i % 100 == 0:
        print(i, np.mean(times))
    B_list.append(featurize(a))
    times.append(time.time() - t0)
B = np.array(B_list)

np.save(dest % ('X', str(len(B))), B)
np.save(dest % ('Y', str(len(Y))), Y)

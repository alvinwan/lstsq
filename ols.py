import numpy as np
import glob
import cv2
import scipy.linalg

N = 1

all_data = []
all_labels = []
for i, path in enumerate(sorted(glob.iglob('state-210x160-SpaceInvaders-v0/*.npy'))):
    if i > N:
        break
    data = np.load(path)
    squished = []
    for datum in data[:, :-2].reshape((-1, 210, 160, 3)):
        squished.append(cv2.resize(datum, (84, 84), interpolation=cv2.INTER_LINEAR)[None])
    new_data = np.concatenate(squished, axis=0)
    all_data.append(new_data)
    all_labels.append(data[:, -2][:,None])
X = np.concatenate(all_data, axis=0)
np.save('compute-210x160-SpaceInvaders-v0/X_1_84x84.npy', X)

Y = np.concatenate(all_labels, axis=0)
np.save('compute-210x160-SpaceInvaders-v0/Y_1_84x84.npy', Y)


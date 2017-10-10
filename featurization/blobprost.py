from skimage.feature import blob_dog
from skimage.feature import canny
import time
import os
import numpy as np
import glob
from multiprocessing import Pool


__all__ = ('blob', 'blob_multi', 'prost')


def blob(frame, x_ways=8, y_ways=7, bins_per_color=1, with_prost=True):
    """Blob features for a single sample.

    :param frame: a single frame, hxwx3
    :param x_ways: Divide x dimension of sample by `x_ways`
    :param y_ways: Divide y dimension of sample by `y_ways`
    :param bins_per_color: Split each color into this many bins.
    :param with_prost: Add prost features
    :return: h2xw2x3 where h2=h/y_ways, w2=w/x_ways
    """
    all_features = []
    h, w = frame.shape[1: 3]
    bin_size = int(np.floor(255. / bins_per_color))
    nh, nw = int(np.ceil(h / y_ways)), int(np.ceil(w / x_ways))

    all_blobs = []
    for channel in range(frame.shape[2]):  # each channel for samples
      state = frame[:, :, channel]
      for bin_idx in range(bins_per_color):  # split each channel into multiple bins, evenly
        start, end = bin_idx * bin_size, (bin_idx + 1) * bin_size
        binned_state = np.zeros(state.shape)
        idxs = np.where(np.logical_and((state >= start), (state < end)))
        binned_state[idxs] = state[idxs]

        features, blobs = canny_blob(binned_state, x_ways=x_ways, y_ways=y_ways,  with_blobs=True)  # only look at values in this bin
        color = np.array([[channel * bins_per_color + bin_idx] * blobs.shape[0]]).T
        all_blobs.append(np.hstack((blobs[:, :2], color)))
        all_features.append(features)

    if with_prost:
        all_blobs = np.vstack(all_blobs)  # grab blob xs, ys
        all_features.append(prost(all_blobs.astype(np.uint8), bins_per_color=bins_per_color))
    return np.hstack(all_features)


def canny_blob(state, x_ways, y_ways, with_blobs=False):
    """Run blob detection on canny edges. Optionally compute pairwise distances."""
    edges = canny(state)
    h, w = state.shape[0], state.shape[1]
    blobs = blob_dog(edges, max_sigma=5, threshold=0.2)
    nh, nw = int(np.ceil(h / y_ways)), int(np.ceil(w / x_ways))
    y, x, r = blobs[:, 0] // y_ways, blobs[:, 1] // x_ways, blobs[:, 2] * np.sqrt(2)  # downsize blobs
    y, x = y.astype(int), x.astype(int)
    featurized = np.zeros((nh, nw))
    featurized[y, x] = r  # fill in lower-dimensional representation
    if with_blobs:
        return featurized.ravel(), np.vstack((y, x)).T
    return featurized.ravel()


def blob_multi(raw, n=48):
    """Blob features for a set of samples.

    :param raw: nxhxwx3, for n samples, axb images and 3 color channels
    :return: nxh2xw2x3, where h2 = h/7, w2 = w/8
    """
    p = Pool(n)
    results = p.map(blob, raw)
    return np.vstack(results)


def prost(all_blobs, bins_per_color=3):
    """Find all pairwise offset distances

    :param all_blobs: blobs in a frame, nx3
    """
    features = np.zeros((30,20,bins_per_color * 3,30,20,bins_per_color * 3))
    norms = np.linalg.norm(all_blobs, axis=1)[:, np.newaxis]
    D = norms + -2*all_blobs.dot(all_blobs.T) + norms.T
    for i, row in enumerate(D):
        y1, x1, c1 = all_blobs[i]
        for j, elem in enumerate(row):
            y2, x2, c2 = all_blobs[j]
            features[y1, x2, c1, y2, x2, c2] = elem
    return features.ravel()


if __name__ == '__main__':
  for i, path in enumerate(glob.iglob('../state-210x160-SpaceInvaders-v0/*.npy')):
    new_path = os.path.join('blobprost', os.path.basename(path))
    if os.path.exists(new_path):
        continue
    start = time.time()
    new = blob_multi(np.load(path)[::100, :-2].reshape((-1, 210, 160, 3)))
    new_data = np.vstack(new)
    np.save(new_path, new_data)
    print(i, time.time() - start, 'saved',  new_data.shape, 'to', new_path)

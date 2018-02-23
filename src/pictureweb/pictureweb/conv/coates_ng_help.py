from ..utils import misc
import numpy as np
import scipy
from numba import jit
import concurrent.futures as fs

def fast_row_mean(X, max_threads=32):
    idxs = misc.chunk_idxs(X.shape[0], max_threads)
    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for (sidx, eidx) in idxs:
            futures.append(executor.submit(lambda x: np.mean(x, axis=1), X[sidx:eidx, :]))
        fs.wait(futures)
        results = np.hstack(list(map(lambda x: x.result(), futures)))
    return results

def fast_row_norm(X, max_threads=32):
    idxs = misc.chunk_idxs(X.shape[0], max_threads)
    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for (sidx, eidx) in idxs:
            futures.append(executor.submit(lambda x: np.linalg.norm(x, axis=1), X[sidx:eidx, :]))
        fs.wait(futures)
        results = np.hstack(list(map(lambda x: x.result(), futures)))
    return results


def normalize_images(images, min_divisor=1e-8, mode="CHW", feature_batch=None):
    images = images.astype('float32')
    images /= 255.0
    return images

def normalize_images_2(images, min_divisor=1e-8, mode="CHW", feature_batch=None):
    if (mode == "HWC"):
        images = images.transpose(0,3,1,2)
    images = images.astype('float32')
    images /= 255.0

    orig_shape = images.shape
    images = images.reshape(images.shape[0], -1)
    n_images = images.shape[0]
    # Zero mean every feature
    images = images - np.mean(images, axis=1)[:,np.newaxis]
    # Normalize
    image_norms = np.linalg.norm(images, axis=1)/55.0
    # Get rid of really small norms
    image_norms[np.where(image_norms < min_divisor)] = 1
    # Make features unit norm
    images_normalized = images/image_norms[:,np.newaxis]
    images_normalized = images_normalized.reshape(orig_shape)
    if (mode == "HWC"):
        images_normalized = images_normalized.transpose(0,2,3,1)
    return images_normalized


def normalize_patches(patches, min_divisor=1e-8, zca_bias=0.1, mean_rgb=np.array([0,0,0])):
    if (patches.dtype == 'uint8'):
        patches = patches.astype('float64')
        patches /= 255.0

    n_patches = patches.shape[0]
    orig_shape = patches.shape
    patches = patches.reshape(patches.shape[0], -1)
    # Zero mean every feature
    patches = patches - np.mean(patches, axis=1)[:,np.newaxis]

    # Normalize
    patch_norms = np.linalg.norm(patches, axis=1)

    # Get rid of really small norms
    patch_norms[np.where(patch_norms < min_divisor)] = 1

    # Make features unit norm
    patches = patches/patch_norms[:,np.newaxis]


    patchesCovMat = 1.0/n_patches * patches.T.dot(patches)

    (E,V) = np.linalg.eig(patchesCovMat)

    E += zca_bias
    sqrt_zca_eigs = np.sqrt(E)
    inv_sqrt_zca_eigs = np.diag(np.power(sqrt_zca_eigs, -1))
    global_ZCA = V.dot(inv_sqrt_zca_eigs).dot(V.T)
    patches_normalized = (patches).dot(global_ZCA).dot(global_ZCA.T)

    return patches_normalized.reshape(orig_shape).astype('float32')


@jit(nopython=True, nogil=True, cache=True)
def __grab_patches(images, patch_size=6, tot_patches=1e6, seed=0):
    np.random.seed(seed)
    tot_patches = int(tot_patches)
    im_idxs = np.random.choice(images.shape[0], tot_patches)
    idxs_x = np.random.choice(images.shape[3] - patch_size - 1, tot_patches)
    idxs_y = np.random.choice(images.shape[2] - patch_size - 1, tot_patches)
    idxs_x += int(np.ceil(patch_size/2))
    idxs_y += int(np.ceil(patch_size/2))
    patches = np.zeros((tot_patches, images.shape[1], patch_size, patch_size), dtype=np.float64)
    for i, (im_idx, idx_x, idx_y) in enumerate(zip(im_idxs, idxs_x, idxs_y)):
        out_patch = patches[i, :, :, :]
        grab_patch_from_idx(images[im_idx], idx_x, idx_y, patch_size, out_patch)
    return patches


@jit(nopython=True, nogil=True)
def grab_patch_from_idx(im, idx_x, idx_y, patch_size, outpatch):
    sidx_x = int(idx_x - patch_size/2)
    eidx_x = int(idx_x + patch_size/2)
    sidx_y = int(idx_y - patch_size/2)
    eidx_y = int(idx_y + patch_size/2)
    outpatch[:,:,:] = im[:, sidx_x:eidx_x, sidx_y:eidx_y,]
    return outpatch

def grab_patches(images, patch_size=6, tot_patches=5e5, seed=0, max_threads=50, dtype=np.uint8):
    idxs = misc.chunk_idxs(images.shape[0], max_threads)
    tot_patches = int(tot_patches)
    patches_per_thread = int(tot_patches/max_threads)
    np.random.seed(seed)
    seeds = np.random.choice(int(1e5), len(idxs), replace=False)
    with fs.ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = []
        for i,(sidx, eidx) in enumerate(idxs):
            futures.append(executor.submit(__grab_patches, images[sidx:eidx, :], 
                                           patch_size=patch_size,
                                           tot_patches=patches_per_thread,
                                           seed=seeds[i]
                                            ))
        results = np.vstack(list(map(lambda x: x.result(), futures)))
    return results




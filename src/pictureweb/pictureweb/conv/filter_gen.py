import numpy as np

def make_gaussian_filter_gen(bandwidth, patch_size=6, channels=3, seed=0):
    ps = patch_size
    np.random.seed(seed)
    def gaussian_filter_gen(num_filters):
        out = np.random.randn(num_filters, channels, ps, ps).astype('float32') * bandwidth
        return out
    return gaussian_filter_gen

def make_empirical_filter_gen_no_mmap(patches, MIN_VAR_TOL=0, seed=0, upsample=4):
    np.random.seed(seed)
    num_patches = patches.shape[0]
    all_idxs = np.random.choice(num_patches, num_patches, replace=False)
    curr_idx = [0]
    def empirical_filter_gen(num_filters):
        idxs = all_idxs[curr_idx[0]:curr_idx[0]+num_filters*upsample]
        curr_idx[0] += num_filters*upsample
        unfiltered = patches[idxs].astype('float32')
        old_shape = unfiltered.shape
        unfiltered = unfiltered.reshape(unfiltered.shape[0], -1)
        unfiltered_vars = np.var(unfiltered, axis=1)
        idxs = np.argsort(-1.0 * unfiltered_vars)
        unfiltered = unfiltered[idxs]
        filtered = unfiltered[np.where(unfiltered_vars > MIN_VAR_TOL)]
        out = filtered[:num_filters].reshape(num_filters, *old_shape[1:])
        return out
    return empirical_filter_gen

def make_empirical_filter_gen(patch_mmap, MIN_VAR_TOL=0, seed=0, upsample=4):
    np.random.seed(seed)
    num_patches = patch_mmap.shape[0]
    all_idxs = np.random.choice(num_patches, num_patches, replace=False)
    curr_idx = [0]
    def empirical_filter_gen(num_filters):
        patches = patch_mmap.load()
        idxs = all_idxs[curr_idx[0]:curr_idx[0]+num_filters*upsample]
        curr_idx[0] += num_filters*upsample
        unfiltered = patches[idxs].astype('float32')
        old_shape = unfiltered.shape
        unfiltered = unfiltered.reshape(unfiltered.shape[0], -1)
        unfiltered_vars = np.var(unfiltered, axis=1)
        idxs = np.argsort(-1.0 * unfiltered_vars)
        unfiltered = unfiltered[idxs]
        filtered = unfiltered[np.where(unfiltered_vars > MIN_VAR_TOL)]
        out = filtered[:num_filters].reshape(num_filters, *old_shape[1:])
        return out
    return empirical_filter_gen

def patchify_all_imgs(X, patch_shape, pad=True, pad_mode='constant', cval=0):
    out = []
    X = X.transpose(0,2,3,1)
    i = 0
    for x in X:
        dim = x.shape[0]
        patches = patchify(x, patch_shape, pad, pad_mode, cval)
        out_shape = patches.shape
        out.append(patches.reshape(out_shape[0]*out_shape[1], patch_shape[0], patch_shape[1], -1))
    return np.array(out)

def patchify(img, patch_shape, pad=True, pad_mode='constant', cval=0):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    #FIXME: Make first two coordinates of output dimension shape as img.shape always

    if pad:
        pad_size= (patch_shape[0]/2, patch_shape[0]/2)
        img = np.pad(img, (pad_size, pad_size, (0,0)),  mode=pad_mode, constant_values=cval)

    img = np.ascontiguousarray(img)  # won't make a copy if not needed

    X, Y, Z = img.shape
    x, y= patch_shape
    shape = ((X-x+1), (Y-y+1), x, y, Z) # number of patches, patch_shape
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed when indices
#    i,j,k,l are incremented by one
    strides = img.itemsize*np.array([Y*Z, Z, Y*Z, Z, 1])
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    return patches


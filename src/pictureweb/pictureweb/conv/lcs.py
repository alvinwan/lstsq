import numpy as np

def patchify(img, patch_shape, pad=False, pad_mode='constant', cval=0):
    ''' Function borrowed from:
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    '''
    #FIXME: Make first two coordinates of output dimension shape as img.shape always

    if pad:
        pad_size= (int(patch_shape[0]/2), int(patch_shape[0]/2))
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

def lcs_descriptor(im, patch_size=24, subpatch_scale=4, stride=4):
    out_size_x = (im.shape[0] - patch_size + 1)
    out_size_y = (im.shape[1] - patch_size + 1)
    subpatch_size = int(patch_size/subpatch_scale)
    # word size for each 24 x 24 patch is a mean and s.d for each subpatch
    word_size = 2*(subpatch_scale**2)
    lcs_words_shape = (out_size_x,out_size_y, word_size)
    patches = patchify(im, patch_shape=(subpatch_size, subpatch_size), pad_mode='constant', pad=True)
    patches_means = np.mean(patches, axis=(2,3))
    patches_stds = np.std(patches, axis=(2,3))
    all_means_lcs = patchify(patches_means, patch_shape=(patch_size, patch_size))[::stride, ::stride, ::subpatch_size, ::subpatch_size]
    all_stds_lcs = patchify(patches_stds, patch_shape=(patch_size, patch_size))[::stride, ::stride, ::subpatch_size, ::subpatch_size]
    all_means_lcs = all_means_lcs.reshape(all_means_lcs.shape[0]*all_means_lcs.shape[1], -1)
    all_stds_lcs = all_stds_lcs.reshape(all_stds_lcs.shape[0]*all_stds_lcs.shape[1], -1)
    lcs_features = np.hstack((all_means_lcs, all_stds_lcs))
    return lcs_features

from ..utils import misc
from ..utils import linalg
from .sharded_matrix import ShardedMatrix, ShardedSymmetricMatrix
from . import matmul
from ..utils.hash import *
import concurrent.futures as fs
import numpy as np
import math
try:
    import pywren
except:
    pass
import time
import boto3
import pickle



def compute_kernel_block(block_pair, K, X, gamma, num_features):
    ''' Compute a kernel block when design matrix is sharded on s3 '''
    start = time.time()
    block_0, block_1 = block_pair
    # kernel symmetric so only generate lower triangle
    flipped = False
    if (block_1 > block_0):
        block_0, block_1 = block_1, block_0
        flipped = True
    meta_time = time.time() - start
    start = time.time()
    # TODO: For now we assume X is only blocked one way
    block1 = X.get_block(block_0, 0)
    block2 = X.get_block(block_1, 0)

    block1 /= np.sqrt(num_features)
    block2 /= np.sqrt(num_features)

    download_time = time.time() - start
    start = time.time()
    K_block = linalg.computeRBFGramMatrix(block1, block2, gamma=gamma)
    kernel_time  = time.time() - start

    start = time.time()
    out_key = K.put_block(block_0, block_1, K_block)
    upload_time = time.time() - start
    return np.array([meta_time, download_time, kernel_time, upload_time])

def compute_linear_kernel_block(block_pair, K, X, exists=False):
    ''' Compute a kernel block when design matrix is sharded on s3 '''
    start = time.time()
    block_0, block_1 = block_pair
    # kernel symmetric so only generate lower triangle
    flipped = False
    if (block_1 > block_0):
        block_0, block_1 = block_1, block_0
        flipped = True
    meta_time = time.time() - start
    start = time.time()
    # TODO: For now we assume X is only blocked one way
    block1 = X.get_block(block_0, 0)
    block2 = X.get_block(block_1, 0)
    download_time = time.time() - start
    start = time.time()
    K_block = block1.dot(block2.T)
    kernel_time  = time.time() - start
    start = time.time()
    if (exists):
        K_block_old = K.get_block(block_0, block_1)
        K_block += K_block_old
    out_key = K.put_block(block_0, block_1, K_block)
    upload_time = time.time() - start
    return np.array([meta_time, download_time, kernel_time, upload_time])

def compute_rbf_kernel_blockwise(key, block_pairs, X, gamma, num_features):
    times = np.zeros(4)
    K = ShardedSymmetricMatrix(key, shape=(X.shape[0], X.shape[0]), bucket=X.bucket,
                      shard_size_0=X.shard_size_0, shard_size_1=X.shard_size_0, prefix=X.prefix)
    for bp in block_pairs:
        time = compute_kernel_block(bp, K, X, gamma, num_features)
        times += time
    times /= float(len(block_pairs))
    return times, K

def compute_linear_kernel_blockwise(block_pairs, X, exists=False):
    times = np.zeros(4)
    kernel_key = make_linear_hash(X.key)
    K = ShardedSymmetricMatrix(kernel_key, shape=(X.shape[0], X.shape[0]), bucket=X.bucket,
                      shard_size_0=X.shard_size_0, shard_size_1=X.shard_size_0, prefix=X.prefix)
    if (exists):
        K.get_block(0,0)

    for bp in block_pairs:
        time = compute_linear_kernel_block(bp, K, X, exists=exists)
        times += time
    times /= float(len(block_pairs))
    return times, K

def compute_sqnorms(X_train_sharded, blocks, axis=1):
    sqnorms = []
    for block in blocks:
        block_data = X_train_sharded.get_block(*block, flip=(not axis))
        sqnorm = np.pow(np.linalg.norm(block_data, axis=axis), 2)
        sqnorms.append(sqnorm)
    return sqnorms


def compute_rbf_kernel_blocked(K, XXT, blocked_sq_norms_train, blocked_sq_norms_test, block_idxs, gamma, num_features):
    if (blocked_sq_norms_test == None):
        blocked_sq_norms_test = blocked_sq_norms_train
    for block_idx in block_idxs:
        i,j = block_idx
        block = XXT.get_block(*block_idx)
        block *= -2
        block += blocked_sq_norms_test[i][:, np.newaxis]
        block += blocked_sq_norms_train[j][:, np.newaxis].T
        block *= -1*gamma
        block /= num_features
        K_block = np.exp(block, block)
        K.put_block(i, j, K_block)
    return K

def compute_quadratic_kernel_blocked(K, XXT, block_idxs, gamma):
    for block_idx in block_idxs:
        i,j = block_idx
        block = XXT.get_block(*block_idx)
        ab = gamma*block
        # quadratic kernel
        K_block = 2*ab
        K_block += ab*ab
        K_block += 1
        K.put_block(i, j, K_block)
    return K

def compute_quadratic_kernel_pywren(pwex, linear_kernel, X_train, X_test, gamma, tasks_per_job, num_jobs=None):

    key = "quadratic({0}, {1})".format(linear_kernel.key, gamma)
    shape = linear_kernel.shape
    shard_size_0 = linear_kernel.shard_size_0
    prefix = linear_kernel.prefix
    if (isinstance(linear_kernel, ShardedSymmetricMatrix)):
        K = ShardedSymmetricMatrix(key, shape=shape, bucket=linear_kernel.bucket,
                                shard_size_0=shard_size_0, shard_size_1=shard_size_0, prefix=prefix)
    else:
        K = ShardedMatrix(key, shape=shape, bucket=linear_kernel.bucket,
                                shard_size_0=shard_size_0, shard_size_1=shard_size_0, prefix=prefix)

    chunked_blocks = list(misc.chunk(list(misc.chunk(K.block_idxs, tasks_per_job)), num_jobs))

    exclude_modules = ["site-packages"]
    all_futures = []
    client = boto3.client('s3')
    straggler_futures = []
    straggler_args = []
    straggler_count = 0

    for i, c in enumerate(chunked_blocks):
        print("Submitting {0} jobs for chunk {1} out of {2} ".format(len(c), i, len(chunked_blocks)))
        futures = pwex.map(lambda x: compute_quadratic_kernel_blocked(K, linear_kernel, x, gamma), c, exclude_modules=exclude_modules)
        all_futures.append(futures)
        print("Waiting for {0} jobs for chunk {1} out of {2} ".format(len(c), i, len(chunked_blocks)))
        start_time = time.time()
        client = boto3.client('s3')
        while(True):
            fs_done, fs_not_done = pywren.wait(futures, pywren.ALWAYS)
            if (len(fs_done) > 0):
                # throw exceptions here
                fs_done[-1].result()
                fs_done[0].result()
            if (len(fs_not_done) == 0):
                break
            tot_time = time.time() - start_time
            print("Currently done {0} currently not done {1} ".format(len(fs_done), len(fs_not_done)))
            if (len(fs_done) > num_jobs*0.5):
                straggler_futures.append(fs_not_done)
                straggler_count += len(straggler_futures)
                break
            time.sleep(5)

    if (straggler_count > 0):
        for futures in straggler_futures:
            pywren.wait(futures)
            futures[0].result()
            futures[-1].result()
    return K


def compute_rbf_kernel_pywren(pwex, linear_kernel, X_train, X_test, gamma, tasks_per_job, num_jobs=None, sq_norms_train=None, sq_norms_test=None, num_features=1, **kwargs):
    blocked_sq_norms_train = {}
    blocked_sq_norms_test = {}
    if (sq_norms_train is None):
        sq_norms_train = matmul.compute_sq_norms_pywren(pwex, X_train, tasks_per_job)

    if (sq_norms_test is None and X_train.key != X_test.key):
        sq_norms_test = matmul.compute_sq_norms_pywren(pwex, X_test, tasks_per_job)

    num_out_blocks = len(linear_kernel.blocks)
    print("NUM OUT BLOCKS", num_out_blocks)

    if (num_jobs == None):
        # if there is no maximum number of jobs then just divide total number of (chunked) block matrix multiplies evenly
        num_jobs = int(num_out_blocks/float(tasks_per_job))

    d_blocks = linear_kernel._blocks(1)
    d_block_idxs = linear_kernel._block_idxs(1)
    for i,(sidx, eidx) in enumerate(d_blocks):
        blocked_sq_norms_train[d_block_idxs[i]] = sq_norms_train[sidx:eidx]

    if (X_train.key != X_test.key):
        d_blocks = linear_kernel._blocks(0)
        d_block_idxs = linear_kernel._block_idxs(0)
        for i,(sidx, eidx) in enumerate(d_blocks):
            blocked_sq_norms_test[d_block_idxs[i]] = sq_norms_test[sidx:eidx]
    else:
        blocked_sq_norms_test = None



    key = "rbf({0}, {1})".format(linear_kernel.key, gamma)
    shape = linear_kernel.shape
    shard_size_0 = linear_kernel.shard_size_0
    prefix = linear_kernel.prefix
    if (isinstance(linear_kernel, ShardedSymmetricMatrix)):
        K = ShardedSymmetricMatrix(key, shape=shape, bucket=linear_kernel.bucket,
                                shard_size_0=shard_size_0, shard_size_1=shard_size_0, prefix=prefix)
    else:
        K = ShardedMatrix(key, shape=shape, bucket=linear_kernel.bucket,
                                shard_size_0=shard_size_0, shard_size_1=shard_size_0, prefix=prefix)

    chunked_blocks = list(misc.chunk(list(misc.chunk(K.block_idxs, tasks_per_job)), num_jobs))

    exclude_modules = ["site-packages"]
    all_futures = []
    client = boto3.client('s3')
    straggler_futures = []
    straggler_args = []
    straggler_count = 0

    for i, c in enumerate(chunked_blocks):
        print("Submitting {0} jobs for chunk {1} out of {2} ".format(len(c), i, len(chunked_blocks)))
        futures = pwex.map(lambda x: compute_rbf_kernel_blocked(K, linear_kernel, blocked_sq_norms_train, blocked_sq_norms_test, x, gamma, num_features), c, exclude_modules=exclude_modules)
        all_futures.append(futures)
        print("Waiting for {0} jobs for chunk {1} out of {2} ".format(len(c), i, len(chunked_blocks)))
        start_time = time.time()
        client = boto3.client('s3')
        while(True):
            fs_done, fs_not_done = pywren.wait(futures, pywren.ALWAYS)
            if (len(fs_done) > 0):
                # throw exceptions here
                fs_done[-1].result()
                fs_done[0].result()
            if (len(fs_not_done) == 0):
                break
            tot_time = time.time() - start_time
            print("Currently done {0} currently not done {1} ".format(len(fs_done), len(fs_not_done)))
            if (len(fs_done) > num_jobs*0.5):
                straggler_futures.append(fs_not_done)
                straggler_count += len(straggler_futures)
                break
            time.sleep(5)

    if (straggler_count > 0):
        for futures in straggler_futures:
            pywren.wait(futures)
            futures[0].result()
            futures[-1].result()
    return K



def compute_linear_kernel_pywren(pwex, X_train_sharded, tasks_per_job, num_jobs=None):
    return compute_XTX_pywren(pwex, X, tasks_per_job, num_jobs=num_jobs, axis=1)


def compute_kernel_pywren(pwex, key, X_train_sharded, tasks_per_job, gamma, num_jobs=None, num_features=1):
    num_blocks = int(math.ceil(X_train_sharded.shape[0]/float(X_train_sharded.shard_size_0)))


    if (num_jobs == None):
        num_kernel_blocks = (num_blocks * num_blocks)/2 + float(num_blocks)/2
        num_jobs = int(num_kernel_blocks/float(tasks_per_job))

    chunked_blocks = misc.generate_chunked_block_pairs(num_blocks, tasks_per_job, num_jobs)
    all_futures = []
    for i,c in enumerate(chunked_blocks):
        print("Submitting jobs, chunk {0}".format(i))
        kernel_futures = pwex.map(lambda x: compute_rbf_kernel_blockwise(key, x, X_train_sharded, gamma, num_features), c)

        all_futures.append(kernel_futures)

    print("Waiting for jobs")
    for kernel_futures in all_futures:
        pywren.wait(kernel_futures)

    K_sharded = compute_rbf_kernel_blockwise([(num_blocks-1,num_blocks-1)], X_train_sharded, gamma)[1]
    return K_sharded

def fast_kernel_column_block_get(K, col_block, num_blocks=313, workers=5):
    s = time.time()
    with fs.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for i in range(0,15):
            futures.append(executor.submit(K.get_block, i, col_block))
        fs.wait(futures)
        results = list(map(lambda x: x.result(), futures))
    e = time.time()
    return np.vstack(results)


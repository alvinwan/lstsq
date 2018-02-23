from .distributed import MapResult, matrix_chunk, load_mmap, fast_kernel_row_block_async, fast_kernel_column_block_async, fast_kernel_row_blocks_get
import itertools
import numpy as np
import math
from ..utils import misc
from ..utils import hash
from .sharded_matrix import ShardedSymmetricMatrix, ShardedMatrix
try:
    import pywren
except:
    pass
import time
import concurrent.futures as fs
import boto3
import os
try:
    import watchtower
    import logging
except:
    pass

def compute_sq_norms_block(X,blocks, other_block=0):
    sq_norms = []
    print(blocks)
    print(other_block)
    for block in blocks:
        sq_norm = np.power(np.linalg.norm(X.get_block(block, other_block), axis=1), 2)
        sq_norms.append(sq_norm)

    out = np.hstack(sq_norms).T
    print(out.shape)
    return out


def compute_sq_norms_pywren(pwex,X,tasks_per_job=5):
    chunked_blocks = list(misc.chunk(X._block_idxs(axis=0), tasks_per_job))
    reduce_idxs = X._block_idxs(axis=1)
    print("reduce",reduce_idxs)

    sq_norms = np.zeros(X.shape[0])
    exclude_modules = ["site-packages"]
    s = time.time()
    for other_idx in reduce_idxs:
        def pywren_run(blocks):
            return compute_sq_norms_block(X, blocks, other_block=other_idx)

        all_futures = []
        print("Submitting job for chunk {0}".format(other_idx))
        futures = pwex.map(pywren_run, chunked_blocks, exclude_modules=exclude_modules)
        print("Waiting for chunk {0}".format(other_idx))
        pywren.wait(futures)
        sq_norms += np.hstack([f.result() for f in futures])
    return sq_norms

def compute_xyt_block(XYT, X, Y, block_pair_0, other_block=0, exists=False):
    ''' Compute a XtX block when X matrix is sharded on s3 
        (in both directions)
    '''
    start = time.time()
    return 0

def compute_xyt_block(XYT, X, Y, block_pair_0, other_block=0, exists=False):
    ''' Compute a XtX block when X matrix is sharded on s3 
        (in both directions)
    '''
    start = time.time()
    block_0, block_1 = block_pair_0
    print(block_0)
    print(block_1)

    block1 = X.get_block(block_0, other_block)
    block2 = Y.get_block(block_1, other_block)

    XYT_block = block1.dot(block2.T)
    return 0

def compute_xyt_blockwise_lambda(block_pairs, XYT, X, Y, reduce_idxs=[0]):
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        XYT_block = None
        for r in reduce_idxs:
            block1 = X.get_block(bidx_0, r)
            block2 = Y.get_block(bidx_1, r)
            if (XYT_block == None):
                XYT_block = block1.dot(block2.T)
            else:
                XYT_block += block1.dot(block2.T)
        XYT.put_block(bidx_0, bidx_1, XYT_block)

def compute_xyt_blockwise_sync(block_pairs, XYT, X, Y, reduce_idxs=[0]):
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        XYT_block = None
        for r in reduce_idxs:
            block1 = X.get_block(bidx_0, r)
            block2 = Y.get_block(bidx_1, r)
            if (XYT_block == None):
                XYT_block = block1.dot(block2.T)
            else:
                XYT_block += block1.dot(block2.T)
        XYT.put_block(block_0, block_1, XYT_block)
def compute_xyt_blockwise_sync(block_pairs, XYT, X, Y, reduce_idxs=[0], shm_size=50):
    import boto3
    import os
    pywren_call_start = time.time()
    #logger = logging.getLogger(XYT.key)
    boto3.setup_default_session(region_name='us-west-2')
    #logger.addHandler(watchtower.CloudWatchLogHandler("XYTGenerate", create_log_group=False))
    #logger.setLevel(logging.DEBUG)
    #logger.info("Starting to generate matrix blocks {0}".format(block_pairs))
    start_remount  = time.time()
    os.system("sudo mount -o remount,size={0}G /dev/shm".format(shm_size))
    end_remount = time.time()
    remount_time = end_remount - start_remount
    #logger.info("({0}) Remount Time: {1}".format(block_pairs, remount_time))
    for bp in block_pairs:
        bidx_0, bidx_1 = bp
        start_full = time.time()
        start_download = time.time()
        block_0 = fast_kernel_row_blocks_get(X, reduce_idxs, "/dev/shm/block_0", workers=22, dtype="float64", row_blocks=[bidx_0])
        block_1 = fast_kernel_row_blocks_get(Y, reduce_idxs, "/dev/shm/block_1", workers=22, dtype="float64", row_blocks=[bidx_1])
        end_download = time.time()
        download_time = end_download - start_download
        #logger.info("({0}) Download Time: {1}".format(bp, download_time))

        start_mul = time.time()
        XYT_block = block_0.dot(block_1.T)
        end_mul = time.time()
        mul_time = end_mul - start_mul
        #logger.info("({0}) Mul Time: {1}".format(bp, mul_time))

        start_up = time.time()
        XYT.put_block(bidx_0, bidx_1, XYT_block)
        end_up = time.time()
        up_time = end_up - start_up
        #logger.info("({0}) Up Time: {1}".format(bp, up_time))

        end_full = time.time()
        full_time  = start_full - end_full
        #logger.info("({0}) Full Time: {1}".format(bp, full_time))

    pywren_call_end  = time.time()
    #logger.info("Finished generating matrix blocks {0}, took {1} seconds".format(block_pairs, pywren_call_end - pywren_call_start))
    return 0


def compute_xyt_blockwise_async(block_pairs, XYT, X, Y, reduce_idxs=[0]):
    import boto3
    boto3.setup_default_session(region_name='us-west-2')
    start_total = time.time()
    executor = fs.ProcessPoolExecutor(max_workers=16)

    # set up the mmap_locations we need room to keep 4 in shared memory for MAXIMUM EFFICIENCY
    mmap_locs_0 = ["/dev/shm/block0_0", "/dev/shm/block0_1"]
    mmap_locs_1 = ["/dev/shm/block1_0", "/dev/shm/block1_1"]

    # start mmap loading shit
    mmap_idx = 0
    block_0_idx, block_1_idx = block_pairs[0]
    # grab the futures for first two blocks we are going to multiply
    block_0_futures = fast_kernel_row_block_async(X, reduce_idxs, row_blocks=[block_0_idx], mmap_loc=mmap_locs_0[mmap_idx], executor=executor)
    block_1_futures  = fast_kernel_row_block_async(Y, reduce_idxs, row_blocks=[block_1_idx], mmap_loc=mmap_locs_1[mmap_idx], executor=executor)
    try:
        logging.basicConfig(level=logging.DEBUG)
        #logger = logging.get#logger(XYT.key)
        #logger.addHandler(watchtower.CloudWatchLogHandler("XYTGenerate", create_log_group=False))
        #logger.setLevel(logging.DEBUG)
    except:
        pass
    for i, (block_0_idx, block_1_idx) in enumerate(block_pairs):
        #logger.info("Starting to generate matrix blocks ({0}, {1}) for {2}".format(block_0_idx, block_1_idx, XYT.key))
        start = time.time()
        start_get = time.time()
        # wait for the futures to return
        fs.wait(block_0_futures, return_when=fs.FIRST_EXCEPTION)
        fs.wait(block_1_futures, return_when=fs.FIRST_EXCEPTION)
        # throw exceptions if anything happens
        [f.result() for f in block_0_futures]
        [f.result() for f in block_1_futures]
        block_0 = load_mmap(*block_0_futures[0].result())
        block_1 = load_mmap(*block_1_futures[0].result())
        end_get = time.time()
        #logger.debug("Get Time " + str(end_get - start_get))
        mmap_idx = (mmap_idx + 1) % 2
        del block_0_futures
        del block_1_futures
        if (i + 1 != len(block_pairs)):
            next_block_0_idx, next_block_1_idx = block_pairs[i+1]
            block_0_futures = fast_kernel_row_block_async(X, reduce_idxs, row_blocks=[next_block_0_idx], mmap_loc=mmap_locs_0[mmap_idx], executor=executor)
            block_1_futures  = fast_kernel_row_block_async(Y, reduce_idxs, row_blocks=[next_block_1_idx], mmap_loc=mmap_locs_1[mmap_idx], executor=executor)

        start_mul = time.time()
        XYT_block = block_0.dot(block_1.T)
        end_mul = time.time()
        #logger.debug("MULTIME " + str(end_mul - start_mul))

        # BEGIN DEBUG CODE
        '''
        XYT_block_old = XYT.get_block(block_0_idx, block_1_idx)
        block_diff = np.abs(XYT_block_old - XYT_block)
        #logger.debug("Max diff between this and old {0}".format(np.max(block_diff)))
        #logger.debug("Avg diff between this and old {0}".format(np.mean(block_diff)))
        if (np.max(block_diff) > 1e-4 or np.mean(block_diff) > 1e-5):
            print(np.max(block_diff))
            print(np.mean(block_diff))
            raise Exception("Block {0},{1} do not match with old indices".format(block_0_idx, block_1_idx))
        # END DEBUG CODE
        '''
        start_upload = time.time()
        XYT.put_block(block_0_idx, block_1_idx, XYT_block)
        end_upload = time.time()
        #logger.debug("Total Upload time is {0}".format(end_upload - start_upload))
        end = time.time()
        #logger.debug("Total runtime for one block {0}".format(end - start))
        #logger.debug("Amortized runtime for {1} blocks {0}".format((end - start_total)/(i+1), i+1))
    end_total = time.time()
    #logger.debug("Total amortized per block time is {0}".format((end_total - start_total)/len(block_pairs)))
    #logger.info("Finished generating matrix blocks ({0}, {1}) for {2}".format(block_0_idx, block_1_idx, XYT.key))
    return 0

def add_matrices(dest, matrices, blocks):
    for block in blocks:
        x = None
        for matrix in matrices:
            if (x == None):
                x = matrix.get_block(*block)
            else:
                x += matrix.get_block(*block)
        dest.put_block(block[0], block[1], x)
    return dest


def delete_blocks(matrix, blocks):
    deletions = []
    for block in blocks:
        deletions.append(matrix.delete_block(*block))
    return deletions


def delete_matrices_pywren(pwex, matrices, tasks_per_job, num_jobs=None):
    # delete all the intermediate crap we brought upon the world
    all_futures = []
    client = boto3.client('s3')
    exclude_modules = ["site-packages"]

    for j,matrix in enumerate(matrices):
        block_idxs = matrix.block_idxs

        chunked_blocks = list(misc.chunk(list(misc.chunk(block_idxs, tasks_per_job)), num_jobs))

        for i,c in enumerate(chunked_blocks):
            #print(list(map(lambda x: delete_blocks(matrix, x), c)))
            all_futures.append((j,i,pwex.map(lambda x: delete_blocks(matrix, x), c, exclude_modules=exclude_modules)))

    for j,i,futures in all_futures:
        print("Waiting for deletion of chunk {0} in matrix {1}".format(i,j))
        pywren.wait(futures)

    for matrix in matrices:
        header_key = matrix.prefix + matrix.key + "/header"
        client.delete_object(Key=header_key, Bucket=matrix.bucket)
    return 0



def add_matrices_pywren(pwex, dest, matrices, tasks_per_job, num_jobs=None):
    all_blocks = dest.block_idxs
    if (num_jobs == None):
        num_jobs = len(all_blocks)/tasks_per_job
    chunked_jobs = list(misc.chunk(list(misc.chunk(all_blocks, tasks_per_job)), num_jobs))
    all_futures = []
    exclude_modules = ["site-packages"]
    for chunked_job in chunked_jobs:
        futures = pwex.map(lambda x: add_matrices(dest, matrices, x), chunked_job, exclude_modules=exclude_modules)

        pywren.wait(futures)
        all_futures.extend(futures)
    return all_futures[-1].result()


def compute_XYT_pywren(pwex, X, Y, tasks_per_job, num_jobs=None, axis=0, XYT=None, local=False):


    ''' Tl;dr map over axis in X and reduce over the other axis
        for example if we have a 1 million x 32k matrix with a shard size of 4096 
        in both directions, we will have 32k/4096 = 8 **sets** of 
        pywren jobs each of whom will emit/add to an 1 million x 1 million matrix in s3
        note this scheme is designed for "moderately skinny" matrices
        where a 4096 x D doesn't fit in main memory of a lambda instance
    '''



    num_blocks = int(math.ceil(X.shape[axis]/float(X.shard_sizes[axis])))
    num_out_blocks = (num_blocks * num_blocks)/2 + float(num_blocks)/2

    if (num_jobs == None):
        # if there is no maximum number of jobs then just divide total number of (chunked) block matrix multiplies evenly
        num_jobs = int(num_out_blocks/float(tasks_per_job))


    # since these are output blocks they stay constant
    chunked_blocks = misc.generate_chunked_block_pairs(num_blocks, tasks_per_job, num_jobs)

    # 0 -> 1 or 1 -> 0
    other_axis = int((axis-1)**2)

    reduce_idxs = X._block_idxs(axis=other_axis)
    matrices_to_reduce = []
    exists = False
    key = "XYT({0}, {1})".format(X.key, Y.key)
    XYT = ShardedSymmetricMatrix(key, shape=(X.shape[0], Y.shape[0]), bucket=X.bucket,
                          shard_size_0=X.shard_sizes[0], shard_size_1=Y.shard_sizes[0], prefix=X.prefix)

    #exclude_modules = ["site-packages"]
    exclude_modules = []
    def pywren_run(x):
            return compute_xyt_blockwise(x, XYT, X, Y, reduce_idxs=reduce_idxs)
    all_futures = []

    for i, c in enumerate(chunked_blocks):
        print("Submitting job for chunk {0} in axis {1} for chunk {2} in axis {3}".format(i, axis, other_idx, other_axis))
        futures = pwex.map(pywren_run, c, exclude_modules=exclude_modules)
        all_futures.append((i,futures))

    for i, futures in enumerate(all_futures):
        print("Waiting job for chunk {0} in axis {1} for chunk {2} in axis {3}".format(i, axis, other_idx, other_axis))
        pywren.wait(futures)
        [f.result() for f in futures]
    return XYT


def generate_key_name(X, Y):
    if (X.key == Y.key):
        key = "XXT({0})".format(X.key)
    else:
        key = "XYT({0}, {1})".format(X.key, Y.key)
    return key



def compute_XYT_pywren(pwex, X, Y, tasks_per_job, out_bucket, overwrite=False, num_jobs=None, XYT=None, local=False, custom_col=None, out_key=None, matmul_async=False, pywren_mode="standalone"):


    ''' Tl;dr map over axis in X and reduce over the other axis
        for example if we have a 1 million x 32k matrix with a shard size of 4096 
        in both directions, we will have 32k/4096 = 8 **sets** of 
        pywren jobs each of whom will emit/add to an 1 million x 1 million matrix in s3
        note this scheme is designed for "moderately skinny" matrices
        where a 4096 x D doesn't fit in main memory of a lambda instance
    '''
    # 0 -> 1 or 1 -> 0
    reduce_idxs = Y._block_idxs(axis=1)
    print("reduce",reduce_idxs)
    exists = False
    if (out_key == None):
        root_key = generate_key_name(X, Y)
    else:
        root_key = out_key

    if (X.key == Y.key):
        XYT = ShardedSymmetricMatrix(root_key, shape=(X.shape[0], X.shape[0]), bucket=out_bucket, shard_size_0=X.shard_sizes[0], shard_size_1=X.shard_sizes[0], prefix=X.prefix)
    else:
        XYT = ShardedMatrix(root_key, shape=(X.shape[0], Y.shape[0]), bucket=out_bucket, shard_size_0=X.shard_sizes[0], shard_size_1=Y.shard_sizes[0], prefix=X.prefix)

    print("XYT SHAPE", XYT.shape)
    num_out_blocks = len(XYT.blocks)
    exclude_modules = ["site-packages"]
    # since these are output blocks they stay constant
    if (num_jobs == None):
        # if there is no maximum number of jobs then just divide total number of (chunked) block matrix multiplies evenly
        num_jobs = int(num_out_blocks/float(tasks_per_job))



    print("Total number of output blocks", len(XYT.block_idxs))
    print("Total number of output blocks that exist", len(XYT.blocks_exist))

    if (overwrite):
        block_idxs_to_map = list(set(XYT.block_idxs))
    else:
        block_idxs_to_map = XYT.block_idxs_not_exist


    # block_idxs_to_map = list(itertools.product([144,15,101,21,55,12,154,97], XYT._block_idxs(0)))

    print("Number of output blocks to generate ", len(block_idxs_to_map))

    chunked_blocks = list(misc.chunk(list(misc.chunk(block_idxs_to_map, tasks_per_job)), num_jobs))


    def pywren_run(x):
        if (pywren_mode == "lambda"):
            return compute_xyt_blockwise_lambda(x, XYT, X, Y, reduce_idxs=reduce_idxs)
        elif (matmul_async):
            return compute_xyt_blockwise_async(x, XYT, X, Y, reduce_idxs=reduce_idxs)
        else:
            return compute_xyt_blockwise_sync(x, XYT, X, Y, reduce_idxs=reduce_idxs)
    all_futures = []
    print("LOCAL ? " + str(local))
    for i, c in enumerate(chunked_blocks):
        print("Starting test run..")
        print("Submitting job for chunk {0} in axis 0".format(i))
        if (local):
            list(map(pywren_run, c[:1]))
        else:
            s = time.time()
            print(len(c))
            futures = pwex.map(pywren_run, c, exclude_modules=exclude_modules)
            e = time.time()
            print("Pwex Map Time {0}".format(e - s))
            all_futures.append((i,futures))

    if (local):
        return XYT

    start_blocks = len(XYT.block_idxs_exist)
    start = time.time()
    if (overwrite):
        print("Waiting...")
        while(True):
            for i, futures, in all_futures:
                fs_done, fs_not_done = pywren.wait(futures, pywren.ALWAYS)
                if (len(fs_done) > 0):
                    print("SOMETHING FINISHED")
                # Throw exceptions here
                [f.result() for f in fs_done]
            time.sleep(10)

    else:
        while(True):
            for i, futures, in all_futures:
                fs_done, fs_not_done = pywren.wait(futures, pywren.ALWAYS)
                if (len(fs_done) > 0):
                    print("SOMETHING FINISHED")
                # Throw exceptions here
                [f.result() for f in fs_done]
            total_num_blocks = len(XYT.block_idxs)
            current_num_blocks = len(XYT.block_idxs_exist)
            print("{0} out of {1} blocks completed".format(current_num_blocks, total_num_blocks))
            elapsed = time.time() - start

            blocks_per_sec = (current_num_blocks - start_blocks)/elapsed
            print("Average blocks per second of {0}".format(blocks_per_sec))

            if (current_num_blocks == total_num_blocks and not overwrite):
                break

            print("Waiting...")
            time.sleep(30)
    return XYT






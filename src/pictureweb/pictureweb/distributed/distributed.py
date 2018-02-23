import sys
sys.path.insert(0, "..")
from ..utils import misc
from ..utils import hash
import numpy as np
import boto3
import io
import concurrent.futures as fs
import time
from . import sharded_matrix
import pywren


class MapResult(object):
    def __init__(self, key, bucket, arr):
        client = boto3.client('s3')
        self.key = key
        self.bucket = bucket
        bio = io.BytesIO()
        np.save(bio, arr)
        client.put_object(Key=key, Bucket=bucket, Body=bio.getvalue())
    def read(self):
        client = boto3.client('s3')
        return np.load(io.BytesIO(client.get_object(Bucket=self.bucket, Key=self.key)['Body'].read()))


def matrix_chunk(blocks_rows, blocks_columns, col_chunk_size):
    ''' Chunk indices for matrix vector multiply
        Will return a list of the form
        [ (block1, col_chunk_1), ... (block1, col_chunk_k)
          ...............................................
          (blockn, col_chunk_1), ... (blockn, col_chunk_k)
        ]
        and number of column chunks (for unchunking)
    '''
    all_chunks = []
    count = 0
    num_col_chunks = 0
    for i in range(blocks_rows):
        chunks_columns = list(chunk(list(range(blocks_columns)), col_chunk_size))
        num_col_chunks = len(chunks_columns)
        for c in chunks_columns:
            all_chunks.append((i,c))
    return all_chunks, num_col_chunks

def reduce_sum_rows(row):
    results = []
    row_sum = row[0].result().read()

    for i in row:
        row_sum += i.result().read()
    return row_sum


def generate_chunked_block_pairs(num_blocks, inner_chunk_size=10, outer_chunk_size=1000):
    all_pairs = list(itertools.product(range(num_blocks), range(num_blocks)))
    sorted_pairs = map(lambda x: tuple(sorted(x)), all_pairs)
    dedup_sorted_pairs = list(set(sorted_pairs))
    return list(chunk(list(chunk(dedup_sorted_pairs, inner_chunk_size)), outer_chunk_size))

def fast_kernel_row_block_async(K, col_blocks, executor=None, workers=23, mmap_loc="/dev/shm/block0", wait=False, dtype="float64", row_blocks=None):

    if (executor == None):
        executor = fs.ProcessPoolExecutor(max_workers=workers)


    if (row_blocks == None):
        row_blocks = K._block_idxs(0)

    total_block_width = 0
    total_block_height = 0
    for row_block in row_blocks:
        total_block_height += min(K.shard_sizes[0], K.shape[0] - row_block*K.shard_sizes[0])

    print(col_blocks)
    print(max(col_blocks))
    print("ARGMAX", np.argmax(col_blocks))
    for col_block in col_blocks:
        total_block_width += min(K.shard_sizes[1], K.shape[1] - col_block*K.shard_sizes[1])

    mmap_shape = (total_block_height,  total_block_width)
    print("MMAP SHAPE IS ", mmap_shape)
    s = time.time()
    X = np.memmap(mmap_loc, dtype=dtype, mode='w+', shape=mmap_shape)
    e = time.time()
    futures = []
    chunk_size = int(np.ceil(len(col_blocks)/workers))
    chunks = misc.chunk(col_blocks, chunk_size)
    col_offset = 0
    for c in chunks:
        futures.append(executor.submit(K.get_blocks_mmap, row_blocks, c, mmap_loc, mmap_shape, dtype=dtype, col_offset=col_offset, row_offset=0))
        col_offset += len(c)
    return futures


def fast_kernel_column_block_async(K, col_blocks, executor=None, workers=23, mmap_loc="/dev/shm/block0", wait=False, dtype="float64", row_blocks=None):

    if (executor == None):
        executor = fs.ProcessPoolExecutor(max_workers=workers)


    if (row_blocks == None):
        row_blocks = K._block_idxs(0)

    total_block_width = 0
    total_block_height = 0
    for row_block in row_blocks:
        total_block_height += min(K.shard_sizes[0], K.shape[0] - row_block*K.shard_sizes[0])

    for col_block in col_blocks:
        total_block_width += min(K.shard_sizes[1], K.shape[1] - col_block*K.shard_sizes[1])

    mmap_shape = (total_block_height,  total_block_width)
    s = time.time()
    np.memmap(mmap_loc, dtype=dtype, mode='w+', shape=mmap_shape)
    e = time.time()
    futures = []
    chunk_size = int(np.ceil(len(row_blocks)/workers))
    chunks = misc.chunk(row_blocks, chunk_size)
    row_offset = 0
    for c in chunks:
        futures.append(executor.submit(K.get_blocks_mmap, c, col_blocks, mmap_loc, mmap_shape, dtype=dtype, row_offset=row_offset, col_offset=0))
        row_offset += len(c)
    return futures

def load_mmap(mmap_loc, mmap_shape, mmap_dtype):
    return np.memmap(mmap_loc, dtype=mmap_dtype, mode='r+', shape=mmap_shape)

def fast_kernel_column_blocks_get(K, col_blocks, mmap_loc, workers=21, dtype="float64", row_blocks=None):
    futures = fast_kernel_column_block_async(K, col_blocks, mmap_loc=mmap_loc, workers=workers, dtype=dtype, row_blocks=row_blocks)
    fs.wait(futures)
    [f.result() for f in futures]
    return load_mmap(*futures[0].result())

def fast_kernel_row_blocks_get(K, col_blocks, mmap_loc, workers=21, dtype="float64", row_blocks=None):
    futures = fast_kernel_row_block_async(K, col_blocks, mmap_loc=mmap_loc, workers=workers, dtype=dtype, row_blocks=row_blocks)
    fs.wait(futures)
    [f.result() for f in futures]
    return load_mmap(*futures[0].result())

def get_column_block(X_sharded, column_blocks, dtype="float64", mmap_loc=None):
    hash_key = hash.hash_string(X_sharded.key + str(column_blocks))
    if (mmap_loc == None):
        mmap_loc = "/dev/shm/{0}".format(hash_key)
    return fast_kernel_column_blocks_get(X_sharded, \
                                  col_blocks=column_blocks, \
                                  row_blocks=X_sharded._block_idxs(0), \
                                  mmap_loc=mmap_loc, \
                                  dtype=dtype)

def get_row_block(X_sharded, row_blocks, dtype="float64", mmap_loc=None):
    hash_key = hash.hash_string(X_sharded.key + str(row_blocks))
    if (mmap_loc == None):
        mmap_loc = "/dev/shm/{0}".format(hash_key)
    return fast_kernel_row_blocks_get(X_sharded, \
                                  col_blocks=X_sharded._block_idxs(1), \
                                  row_blocks=row_blocks, \
                                  mmap_loc=mmap_loc, \
                                  dtype=dtype)

def get_local_matrix(X_sharded, dtype="float64", workers=22, mmap_loc=None):
    hash_key = hash.hash_string(X_sharded.key)
    if (mmap_loc == None):
        mmap_loc = "/dev/shm/{0}".format(hash_key)
    return fast_kernel_column_blocks_get(X_sharded, \
                                  col_blocks=X_sharded._block_idxs(1), \
                                  row_blocks=X_sharded._block_idxs(0), \
                                  mmap_loc=mmap_loc, \
                                  workers=workers, \
                                  dtype=dtype)

def column_shard_matrix_pywren(K_old, K_new, column_block, num_columns_per_block):
    column = fast_kernel_column_blocks_get(K_old, [column_block], mmap_loc="/dev/shm/column")
    offset = int((column_block*K_old.shard_size_1)/num_columns_per_block)
    idxs = misc.chunk_idxs(column.shape[1], column.shape[1]/num_columns_per_block)
    for i, (sidx, eidx) in enumerate(idxs):
        column_block = column[:, sidx:eidx]
        K_new.put_block(0, offset+i, column_block)
    return K_new



def column_shard_matrix(K, num_columns_per_block, pwex=None, local=True):
    ''' Take K, a matrix sharded into sub matrix blocks
        and return K_new, an identical matrix sharded into column blocks of size 
        num_columns_per_block.
        TODO (vaishaal): K.shard_size_1 must be divisble by num_columns_per_block
    '''
    key = "ColumnSharded({0}, {1})".format(K.key, num_columns_per_block)
    assert(K.shard_size_1 % num_columns_per_block == 0)
    K_new = sharded_matrix.ShardedMatrix(key, shape=K.shape, shard_size_0=K.shape[0], shard_size_1=num_columns_per_block, bucket=K.bucket)
    if (local):
        local_results = [column_shard_matrix_pywren(K, K_new, x, num_columns_per_block) for x in K._block_idxs(0)]
        K_new = local_results[0]
    else:
        futures = pwex.map(lambda x: column_shard_matrix_pywren(K, K_new, x, num_columns_per_block), K._block_idxs(1))
    return futures









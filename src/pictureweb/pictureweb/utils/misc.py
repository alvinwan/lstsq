import numpy as np
import itertools
from types import ModuleType
try:
    from importlib import reload
except:
    pass
from numba import jit
import math
import boto3

class MmapArray():
    def __init__(self, mmaped, mode=None,idxs=None):
        self.loc = mmaped.filename
        self.dtype = mmaped.dtype
        self.shape = mmaped.shape
        self.mode = mmaped.mode
        self.idxs = idxs
        if (mode != None):
            self.mode = mode

    def load(self):
        X = np.memmap(self.loc, dtype=self.dtype, mode=self.mode, shape=self.shape)
        if self.idxs != None:
            return X[self.idxs[0]:self.idxs[1]]
        else:
            return X



def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    if n == 0: return []
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_mmap(mmap_loc, mmap_shape, mmap_dtype):
    return np.memmap(mmap_loc, dtype=mmap_dtype, mode='r+', shape=mmap_shape)

def generate_chunked_block_pairs(num_blocks, inner_chunk_size=10, outer_chunk_size=1000):
    all_pairs = list(itertools.product(range(num_blocks), range(num_blocks)))
    sorted_pairs = map(lambda x: tuple(sorted(x)), all_pairs)
    dedup_sorted_pairs = list(set(sorted_pairs))
    return list(chunk(list(chunk(dedup_sorted_pairs, inner_chunk_size)), outer_chunk_size))

def chunk_idxs(size, chunks):
    chunk_size  = int(np.ceil(size/chunks))
    idxs = list(range(0, size+1, chunk_size))
    if (idxs[-1] != size):
        idxs.append(size)
    return list(zip(idxs[:-1], idxs[1:]))

@jit(nopython=True)
def fast_exp(K):
    for x in range(K.shape[0]):
        for y in range(K.shape[1]):
            K[x,y] = math.exp(K[x,y])
    return K


def rreload(module, depth=0, max_depth=2):
    """Recursively reload modules."""
    try:
        reload(module)
    except:
        pass
    if (depth == max_depth):
        return
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if type(attribute) is ModuleType:
            rreload(attribute, depth=depth+1)

def list_all_keys(bucket, prefix):
    client = boto3.client('s3')
    objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter=prefix)
    keys = list(map(lambda x: x['Key'], objects['Contents']))
    truncated = objects['IsTruncated']
    next_marker = objects.get('NextMarker')
    while (truncated):
        objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter=prefix, Marker=next_marker)
        truncated = objects['IsTruncated']
        next_marker = objects.get('NextMarker')
        keys += list(map(lambda x: x['Key'], objects['Contents']))
    return list(filter(lambda x: len(x) > 0, keys))

def block_key_to_block(key):
    try:
        block_key = key.split("/")[-1]
        blocks_split = block_key.split("_")
        b0_start = int(blocks_split[0])
        b0_end = int(blocks_split[1])
        b1_start = int(blocks_split[3])
        b1_end = int(blocks_split[4])
        return ((b0_start, b0_end), (b1_start, b1_end))
    except:
        return None


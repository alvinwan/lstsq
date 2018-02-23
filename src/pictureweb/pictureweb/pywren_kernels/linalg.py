import numpy as np
import boto3
import os
import hashlib
import io
import cloudpickle
import copy
import math
import time
import itertools
import pywren
from sklearn.kernel_approximation import RBFSampler, Nystroem
import scipy
import threading
import concurrent.futures as fs
from multiprocessing.pool import ThreadPool
import random




'''
A very rough linear algebra wrapper for pywren
Currently only supports:
* Symmetric Matrix Multiply (X.T.dot(X)
* Matrix vector multiply

'''

def hash_numpy_array(x):
    return hashlib.sha1(x.view(np.uint8)).hexdigest()

def hash_string(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def make_rbf_hash(s, gamma):
    return hashlib.sha1("rbf_kernel({0}, {1})".format(s, gamma).encode('utf-8')).hexdigest()

def make_linear_hash(s):
    return hashlib.sha1("linear_kernel({0})".format(s).encode('utf-8')).hexdigest()

def test(x):
    return x


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







class ShardedMatrix(object):
    def __init__(self, hash_key, data=None, shape=None, shard_size_0=4096, shard_size_1=None, bucket=None, prefix='pywren.linalg/', transposed = False, reshard=False, n_jobs=1, replication_factor=1):

        #TODO: Very hairy state rn, please cleanup

        if (hash_key == None):
            assert(not (data is None))
            matrix = data
            hash_key = hash_numpy_array(data)
            shape = matrix.shape
        elif (type(hash_key) == str and (not (data is None))):
            matrix = data
            shape = matrix.shape
        elif (type(hash_key) == str and data is None):
            matrix = None
            pass
        else:
            raise Exception("hash_key or data must be provided")

        if bucket == None:
            bucket = os.environ.get('PYWREN_LINALG_BUCKET')
            if (bucket == None):
                raise Exception("bucket not provided and environment variable PYWREN_LINALG_BUCKET not provided")

        self.bucket = bucket
        self.prefix = prefix
        self.key = hash_key
        self.n_jobs = n_jobs
        self.s3 = None
        self.replication_factor = replication_factor

        self.key_base = prefix + self.key + "/"
        header = self.__read_header__()
        if (header == None and shape == None):
            raise Exception("header doesn't exist and no shape provided")
        if (header != None):
            self.shard_size_0 = header.shard_size_0
            self.shard_size_1 = header.shard_size_1
            self.shape = header.shape
        else:
            self.shape = shape
            self.shard_size_0 = shard_size_0
            self.shard_size_1 = shard_size_1

        self.__transposed__ = transposed
        if (self.shard_size_0 == None): raise Exception("No shard_0_size provided")
        if (self.shard_size_1 == None):
            self.shard_size_1 = self.shape[1]
        self.symmetric = False
        if(reshard and self.__shard_matrix__(matrix)): raise Exception("Matrix sharding failed")

    def __get_matrix_shard_key__(self, start_0, end_0, start_1, end_1, replicate):
            if replicate == 0:
                rep = ""
            else:
                rep = str(replicate)

            return self.key_base + hash_string(str(start_0) + str(end_0) + str(self.shard_size_0) + str(start_1) + str(end_1) + str(self.shard_size_1) + rep)


    def __read_header__(self):
        client = boto3.client('s3')
        try:
            key = self.key_base + "header"
            header = cloudpickle.loads(client.get_object(Bucket=self.bucket, Key=key)['Body'].read())
        except:
            header = None
        return header


    def __write_header__(self):
        client = boto3.client('s3')
        key = self.key_base + "header"
        client.put_object(Key=key, Bucket = self.bucket, Body=cloudpickle.dumps(self))
        return 0

    def __shard_idx_to_key__(self, shard_0, shard_1, replicate=0):
        N = self.shape[0]
        D = self.shape[0]
        start_0 = shard_0*self.shard_size_0
        start_1 = shard_1*self.shard_size_1
        end_0 = min(start_0+self.shard_size_0, N)
        end_1 = min(start_1+self.shard_size_1, D)
        key = self.__get_matrix_shard_key__(start_0, end_0, start_1, end_1, replicate)
        return key

    def __s3_key_to_byte_io__(self, key):
        client = boto3.client('s3')
        return io.BytesIO(client.get_object(Bucket=self.bucket, Key=key)['Body'].read())

    def __save_matrix_to_s3__(self, X, out_key):
        client = boto3.client('s3')
        outb = io.BytesIO()
        np.save(outb, X)
        response = client.put_object(Key=out_key, Bucket=self.bucket, Body=outb.getvalue())
        return response

    def __shard_matrix__(self, X):
        if (X is None): return 0
        print("Sharding matrix..... of shape {0}".format(X.shape))
        N = self.shape[0]
        D = self.shape[1]
        client = boto3.client('s3')
        def err_cb(exception):
            raise exception
        pool = ThreadPool(self.n_jobs)
        try:
            results = []
            for i in range(0, N, self.shard_size_0):
                start_0 = i
                end_0 = min(i+self.shard_size_0, N)
                for j in range(0, D, self.shard_size_1):
                    start_1 = j
                    end_1 = min(j+self.shard_size_1, D)
                    results.append(pool.apply_async(func=self.__shard_matrix_part__, args=(client, X, start_0, end_0, start_1, end_1), error_callback=err_cb))

            pool.close()
            pool.join()
        except:
            pool.terminate()
            raise
        return 0

    def __shard_matrix_part__(self, client, X, start_0, end_0, start_1, end_1):
        N = self.shape[0]
        D = self.shape[1]
        end_0 = min(end_0, N)
        end_1 = min(end_1, D)
        X_block = X[start_0:end_0, start_1:end_1]
        outb = io.BytesIO()
        np.save(outb, X_block)
        for i in range(self.replication_factor):
            key = self.__get_matrix_shard_key__(start_0, end_0, start_1, end_1, i)
            r = client.put_object(Key=key, Bucket=self.bucket, Body=outb.getvalue())
        return 0

    def get_blocks(self, blocks_0, block_1):
        client = boto3.client('s3')
        blocks = [self.get_block(block_0, block_1, client) for block_0 in blocks_0]
        return 0
        return np.vstack(blocks)

    def get_blocks_mmap(self, blocks_0, block_1, mmap_loc, mmap_shape, dtype='float32'):
        X = np.memmap(mmap_loc, dtype=dtype, mode='r+', shape=mmap_shape)
        for block_0 in blocks_0:
            block = self.get_block(block_0, block_1)
            sidx = block_0*self.shard_size_0
            eidx = min((block_0 + 1)*self.shard_size_0, self.shape[0])
            X[sidx:eidx, :] = block
        X.flush()
        return (mmap_loc, mmap_shape, dtype)


    def get_block(self, block_0, block_1, client=None):
        if (client == None):
            client = boto3.client('s3')
        else:
            client = client

        if (self.__transposed__):
            block_0, block_1 = block_1, block_0
        s = time.time()
        r = np.random.choice(self.replication_factor, 1)[0]
        key = self.__shard_idx_to_key__(block_0, block_1, r)
        print(key)
        bio = self.__s3_key_to_byte_io__(key)
        e = time.time()
        s = time.time()
        X_block = np.load(bio)
        e = time.time()

        if (self.__transposed__):
            X_block = X_block.T

        return X_block

    def put_block(self, block_0, block_1, block):
        if (block.shape != (self.shard_size_0, self.shard_size_1)):
            raise Exception("Incompatible block size")

        for i in range(self.replication_factor):
            key = self.__get_matrix_shard_key__(start_0, end_0, start_1, end_1, i)
            r = client.put_object(Key=key, Bucket=self.bucket, Body=outb.getvalue())

        key = self.__shard_idx_to_key__(block_0, block_1)
        print(key)
        return self.__save_matrix_to_s3__(block, key)




    @property
    def T(self):
        return ShardedMatrix(self.key, self.shard_size_0, self.shard_size_1, self.bucket, self.prefix, transposed = True)


    def dumps(self):
        return cloudpickle.dumps(self)


class ShardedSymmetricMatrix(ShardedMatrix):
    def __init__(self, hash_key, data=None, shape=None, shard_size_0=4096, shard_size_1=None, bucket=None, prefix='pywren.linalg/', transposed = False, diag_offset=0.0):
        ShardedMatrix.__init__(self, hash_key, data, shape, shard_size_0, shard_size_1, bucket, prefix, transposed)
        self.symmetric = True
        self.diag_offset = diag_offset

    def get_block(self, block_0, block_1, client=None):
        # For symmetric matrices it suffices to only read from lower triangular
        s = time.time()
        if (client == None):
            client = boto3.client('s3')

        flipped = False
        if (block_1 > block_0):
            flipped = True
            block_0, block_1 = block_1, block_0

        r = np.random.choice(self.replication_factor, 1)[0]
        key = self.__shard_idx_to_key__(block_0, block_1, r)
        e = time.time()
        bio = self.__s3_key_to_byte_io__(key)
        s = time.time()
        X_block = np.load(bio)
        e = time.time()

        if (flipped):
            X_block = X_block.T

        if (block_0 == block_1 and X_block.shape[0] == X_block.shape[1]):
            diag = np.diag_indices(X_block.shape[0])
            X_block[diag] += self.diag_offset

        return X_block

    def put_block(self, block_0, block_1, block):

        if (block_1 > block_0):
            block_0, block_1 = block_1, block_0
            block = block.T

        start_0 = block_0*self.shard_size_0
        end_0 = min(block_0*self.shard_size_0 + self.shard_size_0, self.shape[0])
        shape_0 = end_0 - start_0

        start_1 = block_1*self.shard_size_1
        end_1 = min(block_1*self.shard_size_1 + self.shard_size_1, self.shape[1])
        shape_1 = end_1 - start_1



        if (block.shape != (shape_0, shape_1)):
            raise Exception("Incompatible block size: {0} vs {1}".format(block.shape, (shape_0,shape_1)))

        key = self.__shard_idx_to_key__(block_0, block_1)
        print(key)

        return self.__save_matrix_to_s3__(block, key)




def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def computeDistanceMatrix(XTest, XTrain):
    XTrain = XTrain.reshape(XTrain.shape[0], -1)
    XTest = XTest.reshape(XTest.shape[0], -1)
    XTrain_norms = (np.linalg.norm(XTrain, axis=1) ** 2)[:, np.newaxis]
    XTest_norms = (np.linalg.norm(XTest, axis=1) ** 2)[:, np.newaxis]
    K = XTest.dot(XTrain.T)
    K *= -2
    K += XTrain_norms.T
    K += XTest_norms
    return K

def computeRBFGramMatrix(XTest, XTrain, gamma=1):
    gamma = -1.0 * gamma
    return np.exp(gamma*computeDistanceMatrix(XTest, XTrain))


def compute_kernel_block(block_pair, K, X, gamma):
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
    K_block = computeRBFGramMatrix(block1, block2, gamma=gamma)
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


def block_matrix_multiply(block_0, block_1, K, x):
    print("MULTIPLYING BLOCKS {0}, {1}".format(block_0, block_1))
    K_block = K.get_block(block_0, block_1)
    x_block = x.get_block(block_1, 0)
    out = K_block.dot(x_block)
    del K_block
    del x_block
    return out


def blocks_matrix_multiply(blocks, K, x, out_size=None):
    block_x, blocks_y = blocks
    if (out_size == None):
        K_sample_block = K.get_block(block_x, blocks_y[0])
        x_sample_block = x.get_block(blocks_y[0], 0)
        dim_0 = K_sample_block.shape[0]
        dim_1 = x_sample_block.shape[1]

    y = np.zeros((dim_0, dim_1), 'float32')
    for block_y in blocks_y:
        print(bl)
        y += block_matrix_multiply(block_x, block_y, K, x)

    return MapResult(hash_numpy_array(y), "vaishaalpywrenlinalg",  y)

def compute_rbf_kernel_blockwise(block_pairs, X, gamma):
    times = np.zeros(4)
    kernel_key = make_rbf_hash(X.key, gamma)
    K = ShardedSymmetricMatrix(kernel_key, shape=(X.shape[0], X.shape[0]), bucket=X.bucket,
                      shard_size_0=X.shard_size_0, shard_size_1=X.shard_size_0, prefix=X.prefix)
    for bp in block_pairs:
        time = compute_kernel_block(bp, K, X, gamma)
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

def sum_rows(row):
    results = []
    row_sum = row[0].result().read()

    for i in row:
        row_sum += i.result().read()
    return row_sum

def pywren_matrix_vector_multiply(pwex, K, x, col_chunk_size=10, row_chunk_size=3000):
    num_column_blocks = int(math.ceil(K.shape[0]/float(K.shard_size_0)))
    num_row_blocks = int(math.ceil(K.shape[1]/float(K.shard_size_1)))
    all_chunks, num_col_chunks = matrix_chunk(num_column_blocks, num_row_blocks, col_chunk_size)
    print(num_col_chunks)
    print("NUM CHUNKS",len(all_chunks))
    chunked_blocks = list(chunk(all_chunks, row_chunk_size))
    out_vector = []
    all_futures = []
    for i,c in enumerate(chunked_blocks):
        start_time = time.time()
        futures = pwex.map(lambda chunk: blocks_matrix_multiply(chunk, K, x), c)
        all_futures.extend(futures)
        print("map submitted now")
        end_time = time.time()
        print("submit took {0} seconds".format(end_time - start_time))
        start_time = time.time()
        pywren.wait(futures)
        end_time = time.time()
        print("map took {0} seconds".format(end_time - start_time))
    chunked_results = list(chunk(all_futures, num_col_chunks))
    start_time = time.time()
    reduced_row_futures = pwex.map(sum_rows, chunked_results)
    reduced_rows = []
    pywren.wait(reduced_row_futures)
    for i, x in enumerate(reduced_row_futures):
        print("Reducing row {0}".format(i))
        reduced_rows.append(x.result())
    end_time = time.time()
    print("Reduce Time", end_time - start_time)
    return np.vstack(reduced_rows)

def generate_chunked_block_pairs(num_blocks, inner_chunk_size=10, outer_chunk_size=1000):
    all_pairs = list(itertools.product(range(num_blocks), range(num_blocks)))
    sorted_pairs = map(lambda x: tuple(sorted(x)), all_pairs)
    dedup_sorted_pairs = list(set(sorted_pairs))
    return list(chunk(list(chunk(dedup_sorted_pairs, inner_chunk_size)), outer_chunk_size))


def pcg_pywren(A,b,pwex, prc=lambda x: x, max_iter=100, tol=1e-3, col_chunk_size=25, row_chunk_size=2500):
    i = 0
    # starting residual is b
    r = b
    d = prc(r)
    delta_new = np.linalg.norm(r.T.dot(d))
    delta_0 = delta_new
    print("Delta 0 is {0}".format(delta_0))
    x = np.zeros((A.shape[0], b.shape[1]), 'float32')
    print(x.shape)
    while (True):
        if (i >= max_iter):
            break

        if (delta_new < tol*delta_0):
            break
        # Expensive
        print("Matrix multiply")
        d_sharded = ShardedMatrix(d, shard_size_0=A.shard_size_0, reshard=True, bucket="imagenet-raw")
        q = pywren_matrix_vector_multiply(pwex, A, d_sharded, col_chunk_size=col_chunk_size, row_chunk_size=row_chunk_size)
        a = delta_new/np.linalg.norm(d.T.dot(q))
        print(a)
        x = x + a*d
        r = r - a*q
        print("Iter {0}, NORM IS {1}".format(i,np.linalg.norm(r)))
        s = prc(r)
        delta_old = delta_new
        delta_new = np.linalg.norm(r.T.dot(s))
        beta = delta_new/delta_old
        d = s + beta * d
        i = i + 1

    return x

def make_smart_precondition(X_train, lambdav, gamma, n_components):
    '''Make a preconditioner kernel ridge regression by using nystroem
       features  '''
    print("Computing Nystroem svd")
    nystroem = Nystroem(gamma=gamma, n_components=n_components)
    nystroem.fit(X_train)
    print("Computing Nystroem features")
    X_train_lift = nystroem.transform(X_train)
    print("Computing ZTZ")
    ztz = X_train_lift.T.dot(X_train_lift)
    ztz_reg = ztz + lambdav * np.eye(ztz.shape[0]).astype('float32')
    print("Computing Cholesky")
    L = np.linalg.cholesky(ztz_reg)
    U = scipy.linalg.solve(L, X_train_lift.T)
    print(U.shape)
    def prc(x):
        return (1.0/lambdav)*(x - U.T.dot(U.dot(x)))
    return prc, U


def block_kernel_solve(K, y, epochs=1, max_iter=313, block_size=4096, num_blocks=313, lambdav=0.1, verbose=True, prc=lambda x: x):
        '''Solve (K + \lambdaI)x = y
            in a block-wise fashion
        '''

        # compute some constants
        x = np.zeros(y.shape)
        i = 0
        for e in range(epochs):
                for b in range(int(num_blocks)):
                        if (i > max_iter):
                            return x
                        print("Downloading Block (this is slow)")
                        # pick a subset of the kernel matrix (note K can be mmap-ed)
                        K_block = fast_kernel_column_block_get(K,b, num_blocks=num_blocks)
                        b_start =  b*block_size
                        b_end = min((b+1)*block_size, K.shape[1])

                        print("Applying preconditioner...")
                        K_block = prc(K_block)
                        y_block = prc(y[b_start:b_end, :])

                        # This is a matrix vector multiply very efficient can be parallelized
                        # (even if K is mmaped)

                        # calculate
                        R = np.zeros(y_block.shape)
                        print(R.shape)

                        for b2 in range(int(num_blocks)):
                            if b2 == b: continue
                            s =  b2*block_size
                            e = min((b2+1)*block_size, K.shape[1])
                            Kbb2 = K_block[s:e]
                            R += Kbb2.T.dot(x[s:e, :])



                        Kbb = K_block[b_start:b_end, :]
                        print(Kbb.shape)
                        # Add term to regularizer
                        idxes = np.diag_indices(block_size)
                        Kbb[idxes] += lambdav
                        print("solving system {0}".format(b))
                        print("Residual {0}".format(np.linalg.norm(y_block - R)))
                        x_block = scipy.linalg.solve(Kbb, y_block - R)
                        # update model
                        x[b_start:b_end] = x_block
                        i += 1
        return x

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

def compute_linear_kernel_pywren(pwex, X_train_sharded, tasks_per_job, num_jobs, exists=False):
    ''' If exists is true then add to existing kernel matrix '''
    num_blocks = int(math.ceil(X_train_sharded.shape[0]/float(X_train_sharded.shard_size_0)))
    chunked_blocks = generate_chunked_block_pairs(num_blocks, tasks_per_job, num_jobs)
    i = 0
    for c in chunked_blocks:
        print(i)
        t = time.time()
        kernel_futures = pwex.map(lambda x: linalg.compute_linear_kernel_blockwise(x, X_train_sharded, exists=exists), c)
        e = time.time()
        print("SUBMIT",e - t)
        t = time.time()
        pywren.wait(kernel_futures)
        e = time.time()
        print("MAP", e - t)
        print("Chunk {0} done".format(i))
        i += 1
    K_sharded = compute_linear_kernel_blockwise([(0,0)], X_train_sharded)[1]
    return K_sharded


def compute_kernel_pywren(pwex, X_train_sharded, tasks_per_job, num_jobs, gamma):
    num_blocks = int(math.ceil(X_train_sharded.shape[0]/float(X_train_sharded.shard_size_0)))
    chunked_blocks = generate_chunked_block_pairs(num_blocks, tasks_per_job, num_jobs)
    print(len(chunked_blocks))
    for i,c in enumerate(chunked_blocks):
        kernel_futures = pwex.map(lambda x: linalg.compute_rbf_kernel_blockwise(x, X_train_sharded, gamma), c)
        print("job {0} submitted!".format(i))
        pywren.wait(kernel_futures)
        print("Chunk {0} done!".format(i))
        print(kernel_futures[0].result().key)
    K_sharded = compute_rbf_kernel_blockwise([(0,0)], X_train_sharded, gamma)[1]
    return K_sharded

@jit(nopython=True)
def __calculate_res(R, num_blocks, b, K_block, x, block_size, n):
    for b2 in range(int(num_blocks)):
        if b2 == b: continue
        s =  b2*block_size
        e = min((b2+1)*block_size, n)
        if (np.all(x[s:e, :] == 0)): continue
        Kbb2 = K_block[s:e]
        a = np.dot(Kbb2.T, x[s:e, :])
        R += a
    return R

def block_kernel_solve(K, y, epochs=1, max_iter=313, block_size=4096, num_blocks=313, lambdav=0.1, verbose=True, prc=lambda x: x, workers=22):
        '''Solve (K + \lambdaI)x = y
            in a block-wise fashion
        '''
        labels = np.argmax(y, axis=1)
        with fs.ProcessPoolExecutor(max_workers=workers) as executor:
            mmap_locs = ["/dev/shm/block0", "/dev/shm/block1"]
            mmap_loc = mmap_locs[0]
            # compute some constants
            x = np.zeros(y.shape, 'float32')
            i = 0


            K_block_future = fast_kernel_column_block_async(K_sharded, 0, mmap_loc=mmap_loc, executor=executor, workers=workers)

            for e in range(epochs):
                    y_hat = np.zeros(y.shape)
                    for b in range(int(num_blocks)):
                            iter_start = time.time()
                            if (i > max_iter):
                                return x

                            mmap_loc = mmap_locs[i%2 == 0]
                            print("Grabbing this block from oven")
                            s = time.time()
                            K_block = load_mmap(*K_block_future.result())
                            e = time.time()
                            print("Block spent {0} seconds in oven".format(e - s))
                            print("Putting next Block in oven")
                            s = time.time()
                            K_block_future = fast_kernel_column_block_async(K_sharded, (b+1)%num_blocks, mmap_loc=mmap_loc, executor=executor, workers=workers)
                            # pick a subset of the kernel matrix (note K can be mmap-ed)
                            e = time.time()
                            print("Took {0} seconds to put block in oven".format(e - s))
                            b_start =  b*block_size
                            b_end = min((b+1)*block_size, K.shape[1])
                            y_block = y[b_start:b_end, :]
                            start = time.time()
                            R = np.zeros((b_end - b_start, y.shape[1]), 'float32')
                            __calculate_res(R, num_blocks, b, K_block, x, block_size, K.shape[0])
                            end = time.time()
                            print("Residual time {0}".format(end - start))
                            start = time.time()
                            Kbb = K_block[b_start:b_end, :].astype('float64')
                            print(Kbb.shape)
                            # Add term to regularizer
                            idxes = np.diag_indices(Kbb.shape[0])
                            try:
                                Kbb[idxes] += lambdav
                                x_block = scipy.linalg.solve(Kbb, y_block - R, sym_pos=True)
                                Kbb[idxes] -= lambdav
                                # update model
                                x_block = x_block.astype('float32')
                                x[b_start:b_end] = x_block
                                t = time.time()
                                print("Residual is {0}".format(np.linalg.norm(y_block - R)))
                                y_hat += K_block.dot(x_block)
                                acc = metrics.accuracy_score(np.argmax(y_hat, axis=1), labels)
                                print("Iteration {0}, Training Accuracy {1}".format(i, acc))
                                e = time.time()
                                print("Calculating accuracy took {0} secs ".format(e -s))
                            except LinAlgError as e:
                                print("Singular matrix in block {0} with reg {1}".format(b, lambdav))
                            iter_end = time.time()
                            print("Iteration {0} took {1} seconds".format(i, iter_end - iter_start))
                            i += 1
            return x

def load_mmap(mmap_loc, mmap_shape, mmap_dtype):
    return np.memmap(mmap_loc, dtype=mmap_dtype, mode='r+', shape=mmap_shape)

def fast_kernel_column_block_async(K, col_block, executor=None, workers=22, mmap_loc="/dev/shm/block0", wait=False):
    if (executor == None):
        executor = fs.ProcessPoolExecutor(max_workers=workers)
    num_blocks = 313
    if ((col_block+1)*K.shard_size_1 > K.shape[1]):
        block_width = (K.shape[1] - col_block*K.shard_size_1)
    else:
        block_width = (K.shard_size_1)

    mmap_shape = (K.shape[0],  block_width)
    s = time.time()
    if (not os.path.isfile(mmap_loc)):
        np.memmap(mmap_loc, dtype='float32', mode='w+', shape=mmap_shape)
    e = time.time()
    futures = []
    chunk_size = int(np.ceil(num_blocks/workers))
    chunks = linalg.chunk(list(range(num_blocks)), chunk_size)
    for c in chunks:
        futures.append(executor.submit(K.get_blocks_mmap, c, col_block, mmap_loc, mmap_shape))
    return futures[0]


def fast_kernel_column_block_get(K, col_block, workers=21):
    num_blocks = 313
    with fs.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        chunk_size = int(np.ceil(num_blocks/workers))
        chunks = list(linalg.chunk(list(range(num_blocks)), chunk_size))
        print(len(chunks))
        for c in chunks:
            futures.append(executor.submit(K.get_blocks, c, col_block))
        fs.wait(futures)
        return np.vstack(list(map(lambda x: x.result(), futures)))



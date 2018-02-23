import os
import boto3
import numpy as np
import pywren
import linalg
from sklearn.datasets import fetch_mldata
import math


def test_kernel_matrix_multiply():
    os.system("mkdir -p /tmp/mnist/mldata")
    np.random.seed(0)
    s3 = boto3.resource('s3')
    gamma = 1e-3
    pwex = pywren.default_executor()
    out = s3.meta.client.download_file('imagenet-raw', 'mnist-original.mat', '/tmp/mnist/mldata/mnist-original.mat')
    mnist = fetch_mldata('MNIST original', data_home="/tmp/mnist")
    X_train = mnist.data[:5000].astype('float32')/255.0
    print("resharding matrix")
    X_train_sharded = linalg.ShardedMatrix(None, data=X_train, shard_size_0=4096, bucket="imagenet-raw", reshard=True)
    w = np.random.randn(X_train_sharded.shape[0], 10)
    w_sharded = linalg.ShardedMatrix(hash_key="w_full", data=w, shard_size_0=4096, bucket="imagenet-raw", n_jobs=8, reshard=True)
    print("forming kernel matrix sharded")
    num_blocks = int(math.ceil(X_train_sharded.shape[0]/float(X_train_sharded.shard_size_0)))
    tasks_per_job = 5
    num_jobs = 2500
    chunked_blocks = linalg.generate_chunked_block_pairs(num_blocks, tasks_per_job, num_jobs)
    K_sharded = pywren.get_all_results(pwex.map(lambda x: linalg.compute_rbf_kernel_blockwise(x, X_train_sharded, gamma), chunked_blocks[0]))[0][1]
    Kw_pywren = linalg.pywren_matrix_vector_multiply(pwex, K_sharded, w_sharded, col_chunk_size=1, row_chunk_size=2750)
    print("forming kernel matrix local")
    K = linalg.computeRBFGramMatrix(X_train, X_train, gamma=gamma)
    Kw = K.dot(w)
    print(Kw)
    print(Kw_pywren)
    tot_far = np.sum(~np.isclose(Kw_pywren, Kw))
    max_discrep = np.max(np.abs(Kw_pywren - Kw))

    frac_far = tot_far/float(X_train.shape[0])

    assert (frac_far <= 1e-3) and (max_discrep <= 1e-4)




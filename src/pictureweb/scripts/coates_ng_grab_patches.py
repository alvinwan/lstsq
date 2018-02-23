import sys
sys.path.insert(0, "..")
import multiprocessing as mp
from distributed.sharded_matrix import ShardedMatrix
import distributed.distributed as D
from utils import misc
import numpy as np
import os
from operator import mul
import concurrent.futures as fs
import time
from conv import multigpu
from conv import coates_ng_help
from conv import *




if __name__ == "__main__":
    J = 4
    SHARD_SIZE_1 = 18432
    SHARD_SIZE_0 = 4096
    NUM_GPUS = 16
    conv_args_template = \
    {
      "num_feature_batches":1,
      "data_batch_size": 8,
      "feature_batch_size": 1024,
      "pool_size":24,
      "pool_type":"avg",
      "pool_stride":12,
      "patch_size":19,
      "pad":0,
      "bias": 4.45,
      "conv_stride":4,
      "random_seed": 0,
      "preprocess_batch": coates_ng_help.normalize_images
    }


    X_train = D.get_row_block(X_train_sharded, X_train_sharded._block_idxs(0), dtype="uint8")
    #D.get_row_block(X_train_sharded, [0]).shape)
    np.random.seed(0)

    patches = coates_ng_help.grab_patches(X_train, tot_patches=1e7, patch_size=32)
    patches_flat = patches.reshape(patches.shape[0], -1)

    print("SHARDING MATRIX")

    patches_sharded = ShardedMatrix("imagenet_patches_raw_uint8", bucket="vaishaalpywrenlinalg", reshard=True, data=patches_flat, shard_size_0=65536*2, n_jobs=32)

    '''
    handler = multigpu.MultiGpuHandler(NUM_GPUS)
    handler.kill_all_gpu_processes()
    handler.start_and_wait_for_gpu_init()
    # TODO: Don't hard code this
    full_result_shape = (N, num_out_features)
    blocks_0 = X_train_sharded._block_idxs(0)
    chunked_idxs = misc.chunk_idxs(len(blocks_0), NUM_GPUS)
    gpu_results = []
    for gpu,(sidx,eidx) in zip(handler.gpus, chunked_idxs):
        result = gpu.submit_async(scatter_and_upload_blocks, X_train_sharded, X_train_lift_sharded, J, num_out_features, blocks_0[sidx:eidx])
        gpu_results.append(result)

    for gpu_result in gpu_results:
            gpu_result.result()
    '''














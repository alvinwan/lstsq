import sys
sys.path.insert(0, "..")
import multiprocessing as mp
from distributed.sharded_matrix import ShardedMatrix
from distributed import distributed
from utils import misc
import numpy as np
import os
from operator import mul
import concurrent.futures as fs
from conv import multigpu


def scatter_and_upload_blocks(X_train_sharded, X_train_lift_sharded, scat_j, num_out_features, blocks_0, workers=2):
    from scatwave.scattering import Scattering
    gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
    print("Forming scattering network on gpu {0}".format(gpu_id))
    scat = Scattering(M=256, N=256, J=4).cuda()
    futures = []
    for block_0 in blocks_0:
        X_lift_block = scatter_block(X_train_sharded, scat, num_out_features, block_0)
        upload_block(X_train_lift_sharded, X_lift_block, block_0)
    return 0

# create output
def scatter_block(X_train_sharded, scat, num_out_features, block_0):
        import torch
        gpu_id = os.environ["CUDA_VISIBLE_DEVICES"]
        print("Getting data for block {0}".format(block_0))
        X_train_batch = distributed.fast_kernel_column_blocks_get(X_train_sharded, X_train_sharded._block_idxs(1), "/dev/shm/train_{0}".format(gpu_id), dtype="uint8", row_blocks=[block_0])
        batch_size = X_train_batch.shape[0]

        # hard code image size
        X_train_batch = X_train_batch.reshape(batch_size, 3, 256, 256).astype('float32')
        batch_result_shape = (batch_size, num_out_features)
        X_out_batch = np.zeros(batch_result_shape)
        print("Scattering the image on gpu {0}".format(gpu_id))
        idxs = misc.chunk_idxs(batch_size, int(np.ceil(batch_size/512)))
        for sidx, eidx in idxs:
            chunk = X_train_batch[sidx:eidx]
            chunk_torch = torch.Tensor(chunk).cuda()
            result = scat(chunk_torch).cpu().numpy()
            X_out_batch[sidx:eidx] = result.reshape(result.shape[0], -1)
        return X_out_batch


def upload_block(X_train_lift_sharded, X_lift_batch, block_0):
    # indices of column blocks
    idxs_d = X_train_lift_sharded._blocks(1)
    blocks_d = X_train_lift_sharded._block_idxs(1)
    for (block_1, (sidx, eidx)) in zip(blocks_d, idxs_d):
        print("Uploading block ({0} {1}), indices {2} to {3}".format(block_0, block_1, sidx, eidx))
        X_train_lift_sharded.put_block(block_0, block_1, X_lift_batch[:, sidx:eidx])
    return 0

if __name__ == "__main__":
    J = 4
    SHARD_SIZE_1 = 18432
    SHARD_SIZE_0 = 4096
    NUM_GPUS = 16
    handler = multigpu.MultiGpuHandler(NUM_GPUS)
    handler.kill_all_gpu_processes()
    handler.start_and_wait_for_gpu_init()


    X_train_sharded = ShardedMatrix("imagenet_train_raw_uint8", bucket="vaishaalpywrenlinalg")
    X_test_sharded = ShardedMatrix("imagenet_test_raw_uint8", bucket="vaishaalpywrenlinalg")
    N = X_train_sharded.shape[0]

    # TODO: Don't hard code this
    num_out_features = 320256
    full_result_shape = (N, num_out_features)
    X_train_lift_sharded = ShardedMatrix("X_train_scattering_J_{0}".format(J), shape=full_result_shape, bucket="vaishaalpywrenlinalg", shard_size_1=SHARD_SIZE_1, shard_size_0=SHARD_SIZE_0)

    blocks_0 = X_train_sharded._block_idxs(0)
    chunked_idxs = misc.chunk_idxs(len(blocks_0), NUM_GPUS)
    gpu_results = []
    for gpu,(sidx,eidx) in zip(handler.gpus, chunked_idxs):
        result = gpu.submit_async(scatter_and_upload_blocks, X_train_sharded, X_train_lift_sharded, J, num_out_features, blocks_0[sidx:eidx])
        gpu_results.append(result)

    for gpu_result in gpu_results:
            gpu_result.result()














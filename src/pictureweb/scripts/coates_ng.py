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
from conv import multigpu
from conv import coates_ng_help
from conv import filter_gen
from conv._conv import _conv_tf, conv_compute_output_shape
import logging
import watchtower
import json
import boto3
import time


def conv_and_upload_blocks(X_sharded, X_lift_sharded, filters_mmap, conv_args, blocks_0, outfeatures, imshape=(3, 256, 256)):
    for block_0 in blocks_0:
        fg = filter_gen.make_empirical_filter_gen(filters_mmap, seed=conv_args['random_seed'])


        conv_args["filter_gen"] = fg
        conv_args_copy = conv_args.copy()
        gpuid = "gpu" + str(os.environ["CUDA_VISIBLE_DEVICES"])
        X_block = D.get_row_block(X_sharded, row_blocks=[block_0], dtype="uint8",mmap_loc="/dev/shm/{0}".format(gpuid))
        X_block = X_block.reshape((X_block.shape[0],) + imshape)
        conv_args_copy["data"] = X_block
        log_key = X_lift_sharded.key.replace("(", "__").replace(")", "__")
        cw_handler = watchtower.CloudWatchLogHandler("hyperbandfeaturize", create_log_group=False)
        logfmt = 'DataBlock{0}-%(gpu)s-%(levelname)s-%(message)s'.format(block_0)
        cw_handler.setFormatter(logging.Formatter(logfmt))
        conv_args_copy["log_handlers"] = [cw_handler]
        conv_args_copy["logfmt"] = logfmt
        conv_args_copy["outfeatures"] = outfeatures
        conv_args_copy["logkey"] = log_key

        X_lift_block, _ = _conv_tf(**conv_args_copy)
        X_lift_block = X_lift_block.reshape(X_lift_block.shape[0], -1)
        upload_block(X_lift_sharded, X_lift_block, block_0)
    return 0


def upload_block(X_lift_sharded, X_lift_batch, block_0):
    # indices of row blocks
    # indices of column blocks
    idxs_d = X_lift_sharded._blocks(1)
    blocks_d = X_lift_sharded._block_idxs(1)
    for (block_1, (sidx, eidx)) in zip(blocks_d, idxs_d):
        print("Uploading block ({0} {1}), indices {2} to {3} for matrix{4}".format(block_0, block_1, sidx, eidx, X_lift_sharded.key))
        X_lift_sharded.put_block(block_0, block_1, X_lift_batch[:, sidx:eidx])
    return 0

def conv_args_to_matrix_key(root_matrix, conv_args):
    num_filters = conv_args["num_feature_batches"]*conv_args["feature_batch_size"]
    pool_size = conv_args["pool_size"]
    pool_stride = conv_args["pool_stride"]
    patch_size = conv_args["patch_size"]
    patch_stride = conv_args["conv_stride"]
    bias = conv_args["bias"]
    random_seed = conv_args["random_seed"]
    root_matrix_key = root_matrix.key

    matrix_key = "coatesng_{0}_{1}_{2}_{3}_{4}_{5}_{6}({7})".format(num_filters, patch_size, patch_stride, pool_size, pool_stride, bias, random_seed, root_matrix_key)
    return matrix_key



if __name__ == "__main__":
    NUM_GPUS = 16
    conv_args_template = \
    {
      "num_feature_batches":4,
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

    conv_args = conv_args_template
    sqs = boto3.resource('sqs')
    in_queue = sqs.get_queue_by_name(QueueName='picturewebfeaturize')
    out_queue = sqs.get_queue_by_name(QueueName='picturewebsolve')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

    while(True):
        messages = in_queue.receive_messages(MaxNumberOfMessages=1)
        if (len(messages) == 0):
            print("No message...")
            time.sleep(30)
            continue
        message = messages[0]
        conv_args = conv_args_template.copy()
        in_queue = sqs.get_queue_by_name(QueueName='picturewebfeaturize')
        print(message.body)
        param = json.loads(message.body)
        rand_wait = np.random.choice(list(range(100)), 1)[0]
        try:
            conv_args["patch_size"] = int(param["PatchSize"])
            conv_args["conv_stride"] = int(param["PatchStride"])
            conv_args["pool_size"] = int(param["PoolSize"])
            conv_args["pool_stride"] = int(param["PoolStride"])
            conv_args["bias"] = float(param["Bias"])
            print("Featurizing {0}".format(conv_args))
            X_train_sharded = ShardedMatrix("imagenet_train_raw_uint8", bucket="vaishaalpywrenlinalg")

            X_test_sharded = ShardedMatrix("imagenet_test_raw_uint8", bucket="vaishaalpywrenlinalg")

            N = X_train_sharded.shape[0]
            sample = np.zeros((1, 3, 256, 256))
            mmap_out_train_shape = conv_compute_output_shape(data=sample, **conv_args)
            template = "___coatesng_{0}_{1}_{2}_{3}_{4}_{5}_{6}({7})__".format("num_filters", "patch_size", "patch_stride", "pool_size", "pool_stride", "bias", "random_seed", "root_matrix_key")
            mmap_out_train_shape = (N,) + mmap_out_train_shape[1:]

            out_features = mmap_out_train_shape[1]*mmap_out_train_shape[2]*mmap_out_train_shape[3]

            big_patches_sharded = ShardedMatrix("imagenet_patches_raw_uint8", bucket="vaishaalpywrenlinalg")


            big_patches = D.get_row_block(big_patches_sharded, big_patches_sharded._block_idxs(0)[:1] , dtype="uint8")
            print("Downloaded Patches")
            big_patches = big_patches.reshape(big_patches.shape[0], 3, 32, 32)

            patches = coates_ng_help.grab_patches(big_patches, tot_patches=5e4, patch_size=conv_args['patch_size'])

            filters = coates_ng_help.normalize_patches(patches)

            filters_mmap_data = np.memmap("/dev/shm/imagenet_patches_normalized", shape=patches.shape, dtype='float32', mode="w+")
            np.copyto(filters_mmap_data, filters)
            filters_mmap = multigpu.MmapArray(filters_mmap_data, mode="r+")

            train_key = conv_args_to_matrix_key(X_train_sharded, conv_args)
            test_key = conv_args_to_matrix_key(X_test_sharded, conv_args)

            X_train_lift_sharded = ShardedMatrix(train_key, bucket="picturewebhyperband", shape=(X_train_sharded.shape[0], out_features))
            X_test_lift_sharded = ShardedMatrix(test_key, bucket="picturewebhyperband", shape=(X_test_sharded.shape[0], out_features))
            print("output shape", X_train_lift_sharded.shape)

            handler = multigpu.MultiGpuHandler(NUM_GPUS)
            handler.start_and_wait_for_gpu_init()
            handler.kill_all_gpu_processes()

            blocks_0 = X_test_sharded._block_idxs(0)
            chunked_idxs = misc.chunk_idxs(len(blocks_0), NUM_GPUS)
            gpu_results = []
            for gpu,(sidx,eidx) in zip(handler.gpus, chunked_idxs):
                result = gpu.submit_async(conv_and_upload_blocks, X_test_sharded, X_test_lift_sharded, filters_mmap, conv_args, blocks_0[sidx:eidx], out_features)
                gpu_results.append(result)

            for gpu_result in gpu_results:
                    gpu_result.result()

            if (len(X_train_sharded.blocks_exist) == len(X_train_sharded._block_idxs(0))): continue
            blocks_0 = X_train_sharded._block_idxs(0)
            chunked_idxs = misc.chunk_idxs(len(blocks_0), NUM_GPUS)
            gpu_results = []
            for gpu,(sidx,eidx) in zip(handler.gpus, chunked_idxs):
                result = gpu.submit_async(conv_and_upload_blocks, X_train_sharded, X_train_lift_sharded, filters_mmap, conv_args, blocks_0[sidx:eidx], out_features)
                gpu_results.append(result)

            for gpu_result in gpu_results:
                    gpu_result.result()


            message.delete()
            out_msg = {"test":X_test_lift_sharded.key, "train":X_train_lift_sharded.key}
            print("OUT MESSAGE " + json.dumps(out_msg))
            out_queue.send_message(MessageBody=json.dumps(out_msg))
            time.sleep(20)
        except:
            raise















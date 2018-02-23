import sys
sys.path.insert(0, "..")
import argparse
import os
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

CONV_ARGS_TEMPLATE = \
{
"num_feature_batches":2,
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
RAW_DATA_DTYPE  = "uint8"



def conv_and_upload_blocks(X_sharded, X_lift_sharded, filters_mmap, conv_args, blocks_0, outfeatures, column_block, imshape=(3, 256, 256)):
    gpuid = "gpu" + str(os.environ["CUDA_VISIBLE_DEVICES"])
    print("Convolution on gpu {0}".format(gpuid))
    for block_0 in blocks_0:
        fg = filter_gen.make_empirical_filter_gen(filters_mmap, seed=conv_args['random_seed'] + column_block)
        conv_args["filter_gen"] = fg
        conv_args_copy = conv_args.copy()
        blocks_exist = set(X_lift_sharded.block_idxs_exist)
        if (block_0, column_block) in blocks_exist:
            continue

        X_block = D.get_row_block(X_sharded, row_blocks=[block_0], dtype="uint8",mmap_loc="/dev/shm/{0}".format(gpuid))
        X_block = X_block.reshape((X_block.shape[0],) + imshape)
        conv_args_copy["data"] = X_block
        log_key = X_lift_sharded.key.replace("(", "__").replace(")", "__")
        cw_handler = watchtower.CloudWatchLogHandler("hyperbandfeaturize", create_log_group=False)
        logfmt = 'DataBlock{0}-%(gpu)s-%(levelname)s-%(message)s'.format(block_0)
        cw_handler.setFormatter(logging.Formatter(logfmt))
        cw_handler.setLevel(logging.INFO)
        conv_args_copy["log_handlers"] = [cw_handler]
        conv_args_copy["logfmt"] = logfmt
        conv_args_copy["outfeatures"] = outfeatures
        conv_args_copy["logkey"] = log_key
        print("Generating output block {0}".format((block_0, column_block)))
        X_lift_block, _ = _conv_tf(**conv_args_copy)
        X_lift_block = X_lift_block.reshape(X_lift_block.shape[0], -1)
        upload_block(X_lift_sharded, X_lift_block, block_0, column_block)
    return 0


def upload_block(X_lift_sharded, X_lift_batch, block_0, block_1):
    # indices of row blocks
    # indices of column blocks
    sidx = block_1*X_lift_sharded.shard_size_1
    eidx = min(X_lift_sharded.shape[1], (block_1+1)*X_lift_sharded.shard_size_1)
    print("Uploading block ({0} {1}), indices {2} to {3} for matrix{4}".format(block_0, block_1, sidx, eidx, X_lift_sharded.key))
    X_lift_sharded.put_block(block_0, block_1, X_lift_batch)
    return 0

def reset_msg_visibility(msg, not_done, timeout):
    print("Starting message visibility resetter")
    while(not_done[0]):
        time.sleep(timeout/2)
        msg.change_visibility(VisibilityTimeout=timeout)

def parse_message(msg, conv_args):
        param = json.loads(msg.body)
        conv_args["patch_size"] = int(param["PatchSize"])
        conv_args["conv_stride"] = int(param["PatchStride"])
        conv_args["pool_size"] = int(param["PoolSize"])
        conv_args["pool_stride"] = int(param["PoolStride"])
        conv_args["bias"] = float(param["Bias"])
        conv_args["column_block"] = int(param.get("column_block", 0))
        conv_args["num_column_blocks"] = int(param.get("num_column_blocks", 1))
        return conv_args

def conv_args_to_matrix_key(root_matrix, conv_args):
    num_filters = conv_args["num_feature_batches"]*conv_args["feature_batch_size"]*conv_args["num_column_blocks"]
    pool_size = conv_args["pool_size"]
    pool_stride = conv_args["pool_stride"]
    patch_size = conv_args["patch_size"]
    patch_stride = conv_args["conv_stride"]
    bias = conv_args["bias"]
    random_seed = conv_args["random_seed"]
    root_matrix_key = root_matrix.key

    matrix_key = "coatesng_{0}_{1}_{2}_{3}_{4}_{5}_{6}({7})".format(num_filters, patch_size, patch_stride, pool_size, pool_stride, bias, random_seed, root_matrix_key)
    return matrix_key

def featurize(in_queue, out_queue,
             bucket, timeout,
             raw_train_data_key, raw_test_data_key,
             raw_data_bucket, raw_image_shape,
             raw_patch_data_key, raw_patch_shape, patch_sample, num_gpus):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
    executor = fs.ThreadPoolExecutor(1)
    sqs = boto3.resource('sqs')
    in_queue = sqs.get_queue_by_name(QueueName=in_queue)
    out_queue = sqs.get_queue_by_name(QueueName=out_queue)
    patches_sample = int(patch_sample)
    while(True):
        messages = in_queue.receive_messages(MaxNumberOfMessages=1)
        if (len(messages) == 0):
            print("No message...")
            time.sleep(timeout)
            continue
        message = messages[0]
        conv_args = CONV_ARGS_TEMPLATE.copy()
        not_done = [True]
        executor.submit(reset_msg_visibility, message, not_done, timeout)
        conv_args = parse_message(message, conv_args)
        print("Conv Args " + str(conv_args))
        column_block = conv_args['column_block']
        X_train_sharded = ShardedMatrix(raw_train_data_key, bucket=raw_data_bucket)
        X_test_sharded = ShardedMatrix(raw_test_data_key, bucket=raw_data_bucket)
        N = X_train_sharded.shape[0]
        sample = np.zeros((1,) + raw_image_shape)
        mmap_out_train_shape = conv_compute_output_shape(data=sample, **conv_args)
        mmap_out_train_shape = (N,) + mmap_out_train_shape[1:]
        out_features = mmap_out_train_shape[1]*mmap_out_train_shape[2]*mmap_out_train_shape[3]*conv_args['num_column_blocks']

        big_patches_sharded = ShardedMatrix(raw_patch_data_key, bucket=raw_data_bucket)
        big_patches = D.get_row_block(big_patches_sharded, big_patches_sharded._block_idxs(0)[:1], dtype=RAW_DATA_DTYPE)
        patches_shape = (big_patches.shape[0],) + raw_patch_shape
        big_patches = big_patches.reshape(patches_shape)
        seed = conv_args['random_seed']
        print("grabbing patches")

        # Note that the seed of the grab patches function (which is random)
        # is the random seed + the column block we are generating
        # this means every column block will be different
        # this is very important!!!
        patches = coates_ng_help.grab_patches(big_patches, tot_patches=patch_sample, patch_size=conv_args['patch_size'], seed=seed)

        print("normalizing patches")
        filters = coates_ng_help.normalize_patches(patches)

        filters_mmap_data = np.memmap("/dev/shm/patches", shape=patches.shape, dtype='float32', mode="w+")

        np.copyto(filters_mmap_data, filters)
        filters_mmap = multigpu.MmapArray(filters_mmap_data, mode="r+")

        train_key = conv_args_to_matrix_key(X_train_sharded, conv_args)
        test_key = conv_args_to_matrix_key(X_test_sharded, conv_args)

        print("Train key " + train_key)
        print("Test key " + test_key)
        column_shard_size = int(out_features/conv_args['num_column_blocks'])
        print("Column shard size is " + str(column_shard_size))
        X_train_lift_sharded = ShardedMatrix(train_key, bucket=bucket, shape=(X_train_sharded.shape[0], out_features), shard_size_1=column_shard_size)

        X_test_lift_sharded = ShardedMatrix(test_key, bucket=bucket, shape=(X_test_sharded.shape[0], out_features), shard_size_1=column_shard_size)

        print("Starting gpus")
        handler = multigpu.MultiGpuHandler(num_gpus)
        handler.start_and_wait_for_gpu_init()
        handler.kill_all_gpu_processes()

        out_test_blocks = list(set(list(map(lambda x: x[0], filter(lambda x: x[1] == column_block, X_test_lift_sharded.block_idxs_not_exist)))))
        out_train_blocks = list(set(list(map(lambda x: x[0], filter(lambda x: x[1] == column_block, X_train_lift_sharded.block_idxs_not_exist)))))

        print("out test blocks to generate " + str(len(out_test_blocks)))
        print("out train blocks to generate " + str(len(out_train_blocks)))


        # Generate test features
        chunked_idxs_test = misc.chunk(out_test_blocks, int(np.ceil(len(out_test_blocks)/num_gpus)))
        gpu_results = []
        for gpu, blocks_test in zip(handler.gpus, chunked_idxs_test):
            result = gpu.submit_async(conv_and_upload_blocks, X_test_sharded, X_test_lift_sharded, filters_mmap, conv_args, blocks_test, out_features, column_block)
            gpu_results.append(result)

        for gpu_result in gpu_results:
            gpu_result.result()

        chunked_idxs_train = misc.chunk(out_train_blocks, int(np.ceil(len(out_train_blocks)/num_gpus)))
        gpu_results = []
        for gpu, blocks_train in zip(handler.gpus, chunked_idxs_train):
            result = gpu.submit_async(conv_and_upload_blocks, X_train_sharded, X_train_lift_sharded, filters_mmap, conv_args, blocks_train, out_features, column_block)
            gpu_results.append(result)

        for gpu_result in gpu_results:
            gpu_result.result()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate coates-ng features from sqs queue')
    parser.add_argument('--in_queue', type=str, help="SQS queue with featurizations", default="picturewebfeaturize")
    parser.add_argument('--out_queue', type=str, help="SQS queue for solves", default="picturewebsolve")
    parser.add_argument('--raw_train_key', type=str, help="Train key for raw data", default="imagenet_train_raw_uint8")
    parser.add_argument('--raw_test_key', type=str, help="Test key for raw data", default="imagenet_test_raw_uint8")
    parser.add_argument('--raw_patch_data_key', type=str, help="Key for raw patch data", default="imagenet_patches_raw_uint8")
    parser.add_argument('--raw_data_bucket', type=str, help="Bucket for raw data", default="vaishaalpywrenlinalg")
    parser.add_argument('--bucket', type=str, help="S3 bucket where sharded matrices live", default="picturewebhyperband")
    parser.add_argument('--message_visibility_timeout', type=int, help="How long to keep SQS messages invisbile", default=120)
    parser.add_argument('--raw_image_shape', type=tuple, help="Shape of raw images", default=(3,256,256))
    parser.add_argument('--raw_patch_shape', type=tuple, help="Shape of raw patches", default=(3,32,32))
    parser.add_argument('--patches_sample', type=float, help="Number of raw patches to sample", default=1e7)
    parser.add_argument('--num_gpus', type=int, help="Number of raw patches to sample", default=16)
    parser.add_argument('--total_num_batches', type=int, help="Total number of feature batches", default=4)
    args = parser.parse_args()
    featurize(args.in_queue, args.out_queue, args.bucket, args.message_visibility_timeout, args.raw_train_key, args.raw_test_key, args.raw_data_bucket, args.raw_image_shape, args.raw_patch_data_key, args.raw_patch_shape, args.patches_sample, args.num_gpus)





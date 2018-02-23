import sys
import argparse
sys.path.insert(0, "..")
sys.path.insert(0, "../pictureweb")
from pictureweb.conv import _conv, multigpu, filter_gen
from pictureweb.conv import coates_ng_help
import pictureweb.distributed.distributed as D
from pictureweb.distributed.sharded_matrix import ShardedMatrix
import pywren.wrenconfig as wrenconfig
import os
import numpy as np
import watchtower
import logging
import pywren
import boto3
import os


def conv_and_upload_blocks(X_input, X_output, big_patches_sharded, conv_args, row_blocks_in,  col_block_out, col_block_size, imshape=(3, 256, 256),overwrite=False):
    import os
    os.system("sudo nvidia-smi -pm 1")
    boto3.setup_default_session(region_name='us-west-2')
    if (os.environ.get("CUDA_VISIBILE_DEVICES") == None):
        gpuid = "cpu"
    else:
        gpuid = "gpu" + str(os.environ.get("CUDA_VISIBLE_DEVICES"))

    patch_path = "/dev/shm/patches_{0}".format(conv_args["patch_size"])
    if (not os.path.isfile(patch_path)):
        # we want to replace this with code that downloads all patches from s3, then samples UAR from those patches
        # with added functionality to sample smaller patches from the larger ones:
        # find the shareded matrix within specified bucket
        # make it into an np array
        big_patches = D.get_local_matrix(big_patches_sharded)
        # TODO: reshape - hard coded as 32X32 now
        big_patches = big_patches.reshape(big_patches.shape[0], 3, 32, 32)
        # sample 'tot_patches' patches at the desired resolution 'patch_size' -> filter bank 
        # may want to increase tot_patches=5e4 to something like 1e6
        patches = coates_ng_help.grab_patches(big_patches, tot_patches=5e4, patch_size=conv_args['patch_size'], seed=0)
        print("normalizing patches")
        filters = coates_ng_help.normalize_patches(patches)
        filters_mmap_data = np.memmap(patch_path, shape=patches.shape, dtype='float32', mode="w+")
        np.copyto(filters_mmap_data, filters)
    else:
        filters_mmap_data = np.memmap(patch_path, dtype='float32', mode="r+")
        ps = conv_args['patch_size']
        patch_shape = (-1, 3, ps, ps)
        filters_mmap_data = filters_mmap_data.reshape(patch_shape)
    filters = multigpu.MmapArray(filters_mmap_data, mode="r+")
    print("Convolution on {0}".format(gpuid))
    for row_block in row_blocks_in:
        fg = filter_gen.make_empirical_filter_gen(filters, seed=conv_args['random_seed'] + col_block_out, upsample=1)
        conv_args["filter_gen"] = fg
        conv_args_copy = conv_args.copy()
        blocks_exist = set(X_output.block_idxs_exist)
        if (((row_block, col_block_out) in blocks_exist) and (not overwrite)):
            continue
        # Download row block
        X_block = D.get_row_block(X_input, row_blocks=[row_block], dtype="uint8",mmap_loc="/dev/shm/{0}".format(gpuid))
        X_block = X_block.reshape((X_block.shape[0],) + imshape)
        conv_args_copy["data"] = X_block

        log_key = X_output.key.replace("(", "__").replace(")", "__")
        cw_handler = watchtower.CloudWatchLogHandler("remotesensingfeaturize")

        logfmt = 'DataBlock{0}-%(gpu)s-%(levelname)s-%(message)s'.format(row_block)
        cw_handler.setFormatter(logging.Formatter(logfmt))
        cw_handler.setLevel(logging.INFO)
        conv_args_copy["log_handlers"] = [cw_handler]
        conv_args_copy["logfmt"] = logfmt
        conv_args_copy["outfeatures"] = col_block_size
        conv_args_copy["logkey"] = log_key
        print("Generating output block {0}".format((row_block, col_block_out)))
        X_lift_block, _ = _conv._conv_tf(**conv_args_copy)
        X_lift_block = X_lift_block.reshape(X_lift_block.shape[0], -1)
        upload_block(X_output, X_lift_block, row_block, col_block_out)
    return 0

def upload_block(X_output_sharded, X_output_batch, row_block, column_block):
    sidx = column_block*X_output_sharded.shard_size_1
    eidx = min(X_output_sharded.shape[1], (column_block+1)*X_output_sharded.shard_size_1)
    print("Uploading block ({0} {1}), indices {2} to {3} for matrix {4}".format(row_block, column_block, sidx, eidx, X_output_sharded.key))
    X_output_sharded.put_block(row_block, column_block, X_output_batch)
    return 0

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a matrix of raw pixel values to a feature matrix')
    # arguments for reading/writing to the correct s3 files
    parser.add_argument('in_matrix_name', type=str, help="should be of form sample_strategy_years")
    parser.add_argument('--patches_name', type=str, help="name of patches sharded matrix", default=None)
    parser.add_argument('--out_matrix_name', type=str, help="should be of form featurized(sample_strategy_years)", default=None)
    parser.add_argument('--in_bucket', type=str, help="S3 bucket prefix where the pixel matrix lives", default="remotesensing2")
    parser.add_argument('--out_bucket', type=str, help="S3 bucket prefix where the feature matrix should go", default="remotesensing2")
    parser.add_argument('--in_prefix', type=str, help="S3 bucket prefix", default="pywren.linalg/")
    parser.add_argument('--out_prefix', type=str, help="S3 bucket prefix", default="pywren.linalg/featurized/")
    # arguments for featurization
    parser.add_argument('--num_feature_batches',type=int, help="number of features = 2*num_feature_batches*feature_batch_size",default=4)
    parser.add_argument('--feature_batch_size',type=int, help="number of features = 2*num_feature_batches*feature_batch_size",default=512)
    parser.add_argument('--patch_size',type=int, help="random patches will be of shape patch_size*patch_size",default=24)
    parser.add_argument('--pool_type',type=str, help="how we perform the pool operation",default="avg")
    parser.add_argument('--conv_stride',type=int, help="stride length for convolutions",default=1)
    parser.add_argument('--data_batch_size',type=int, help="data batch size",default=16)
    parser.add_argument('--subsample',type=int, help="limit data size",default=None)
    parser.add_argument('--bias',type=float, help="bias",default=1.0)
    args = parser.parse_args()

    conv_args = \
    {
        "num_feature_batches": args.num_feature_batches,
        "data_batch_size": args.data_batch_size,
        "feature_batch_size": args.feature_batch_size,
        "pool_size": 256-args.patch_size+1,
        "pool_stride": 256-args.patch_size+1,
        "pool_type": args.pool_type,
        "patch_size": args.patch_size,
        "pad":0,
        "bias": args.bias,
        "conv_stride": args.conv_stride,
        "random_seed": 0,
        "num_column_blocks": 1,
        "preprocess_batch": coates_ng_help.normalize_images
    }

    patch_path = "/dev/shm/patches_{0}".format(conv_args["patch_size"])
    print(patch_path)
    RAW_DATA_DTYPE  = "uint8"
    X_input_sharded = ShardedMatrix(args.in_matrix_name, bucket=args.in_bucket)

    if (args.out_matrix_name == None):
        num_filters = args.num_feature_batches * args.feature_batch_size
        out_matrix_name = conv_args_to_matrix_key(X_input_sharded, conv_args)
    else:
        out_matrix_name = args.out_matrix_name

    if (args.patches_name == None):
        patches_name = "patches({0})".format(args.in_matrix_name)
    else:
        patches_name = args.patches_name

    out_shape_0 = X_input_sharded.shape[0]
    out_shape_1 = conv_args["num_feature_batches"]*conv_args["feature_batch_size"]*2
    big_patches_sharded = ShardedMatrix("az_1e6_60000_patches",bucket='remotesensingesther')
    #big_patches_sharded = ShardedMatrix(args.patches_name, bucket=args.in_bucket)
    X_output_sharded = ShardedMatrix(out_matrix_name, prefix=args.out_prefix, bucket=args.out_bucket, shape=(out_shape_0, out_shape_1),shard_size_0=X_input_sharded.shard_size_0)
    print("matrix of shape {0} lives at {1}/{2}{3}".format(X_output_sharded.shape,X_output_sharded.bucket,X_output_sharded.prefix, X_output_sharded.key))
    keys = X_output_sharded.block_idxs_not_exist
    #print(X_output_sharded._block_idxs(axis=0))
    print("there are {0} keys".format(len(keys)))
    config = wrenconfig.default()
    config['runtime']['s3_bucket'] = "remotesensing2"
    config['runtime']['s3_key'] = "pywren.runtime/pywren_runtime-3.6-remotesensing_featurize.tar.gz"
    pwex = pywren.standalone_executor(config=config, job_max_runtime=3600)
    # call the futures
    print("Submitting pywren job to download") 
    if (args.subsample == None):
        subsample = X_input_sharded.shape[0]
    else:
        subsample = args.subsample
    futures = pwex.map(lambda x: conv_and_upload_blocks(X_input_sharded, X_output_sharded, big_patches_sharded, conv_args, row_blocks_in=[x[0]], col_block_out=x[1], col_block_size=out_shape_1), keys[:subsample])
    print("Waiting for pywren job to complete")
    pywren.wait(futures)
    success = 0
    fail = 0
    for f in futures:
        try:
            f.result()
            success += 1
        except:
            fail += 1

    print("{0} chunks uploaded sucessfully, {1} chunks failed".format(success, fail))
    print("result from futures[0]: ")    
    print(futures[0].result())
















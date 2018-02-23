import sys
sys.path.insert(0, "..")
import pictureweb.distributed as distributed
from pictureweb.distributed import kernel
from pictureweb.distributed import bcd
from pictureweb.distributed.sharded_matrix import ShardedMatrix, ShardedSymmetricMatrix
from oauth2client.service_account import ServiceAccountCredentials
import argparse
import pywren
import time
import boto3
import io
import numpy as np
from pictureweb.opt import ls
import os
import gspread
import concurrent.futures as fs
import subprocess
from subprocess import Popen
import os

'''
How to run:
python solve.py "rbf(XXT(X_block_scrambled_3_3_pool_12_12_patch_stride_1_1_blocks_no_normalize_04.22.2017, 3)" "scrambled_train_labels.npy"
'''

def evaluate_augmented(preds, augment_idxs, unaugmented_size, augmented_size=10):
    sort_idxs = np.argsort(augment_idxs)
    sort_preds = preds[sort_idxs]
    out_preds = None
    for i in range(augmented_size):
        idxs = np.where(((np.arange(sort_preds.shape[0]) - i) % augmented_size) == 0)
        print(idxs[0])
        if (out_preds == None):
            out_preds = sort_preds[idxs]
        else:
            out_preds += sort_preds[idxs]
    out_preds /= augmented_size
    return out_preds


def evaluate_test(model_path, test_key, test_labels_key, bucket, exists, augmented, augmented_idxs, augment_size, num_test_blocks):
    print(model_path)
    model = np.load(model_path)
    print(model)
    client = boto3.client('s3')
    print(test_labels_key)
    resp = client.get_object(Key=test_labels_key, Bucket=bucket)
    bio = io.BytesIO(resp["Body"].read())
    y_test = np.load(bio)
    print(y_test.shape)
    print(test_key)
    K_test = ShardedMatrix(test_key, bucket=bucket)
    print(K_test.shape)
    N_test = K_test.shape[1]
    if (num_test_blocks == None):
        num_test_blocks = len(K_test._block_idxs(1))
    test_size = min(num_test_blocks*K_test.shard_sizes[1], N_test)
    print("TEST SIZE ", test_size)
    #if (augmented and test_size != N_test):
        #raise Exception("Augmented evaluation only works with whole test set")
    print("exists ", exists)
    if (not exists):
        print("K_test.shape", K_test.shape)
        print("y_test", y_test.shape)
        futures = bcd.fast_kernel_column_block_async(K_test, col_blocks=K_test._block_idxs(1), mmap_loc="/dev/shm/K_test_block", dtype="float64")
        print("Downloading test block")
        fs.wait(futures)
    K_test_block = bcd.load_mmap("/dev/shm/K_test_block", (K_test.shape[0], test_size), "float64")
    model = model.astype('float64')
    print("K_Test BLOCK SHAPE ", K_test_block.shape)
    y_test_pred = K_test_block.T.dot(model)
    if (augmented):
        y_test_pred_augmented = y_test_pred
        print(augmented_idxs)
        bio = io.BytesIO(client.get_object(Bucket=bucket, Key=augmented_idxs)['Body'].read())
        test_augment_idxs = np.load(bio)
        print("y_test_pred_", y_test_pred_augmented.shape)
        print('test_augment_idxs', test_augment_idxs.shape)
        print('K_test', K_test.shape[1])
        y_test_pred = evaluate_augmented(y_test_pred_augmented, test_augment_idxs, K_test.shape[1], augmented_size=augment_size)

    test_top1 = ls.top_k_accuracy(y_test[:test_size], y_test_pred, k=1)
    test_top5 = ls.top_k_accuracy(y_test[:test_size], y_test_pred, k=5)
    print("Top 1 test {0}, Top 5 test {1}".format(test_top1, test_top5))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve ')
    parser.add_argument('model', type=str, help="path to model (on disk)", default=None)
    parser.add_argument('test_key', type=str, help="S3 Key to sharded test matrix", default=None)
    parser.add_argument('test_labels_key', type=str, help="S3 Key to sharded test matrix", default=None)
    parser.add_argument('--bucket', type=str, help="S3 bucket where sharded matrices live", default="picturewebsolve")
    parser.add_argument('--exists', action='store_const', const=True, default=False)
    parser.add_argument('--augmented', action='store_const', const=True, default=False)
    parser.add_argument('--augmented_test_idxs', type=str, help="augment test idxs", default=None)
    parser.add_argument('--augmented_size', type=int, help="augment size", default=10)
    parser.add_argument('--num_test_blocks', type=int, help="how many blocks to test", default=None)
    args = parser.parse_args()
    evaluate_test(args.model, args.test_key, args.test_labels_key, args.bucket, args.exists, args.augmented, args.augmented_test_idxs, args.augmented_size, args.num_test_blocks)






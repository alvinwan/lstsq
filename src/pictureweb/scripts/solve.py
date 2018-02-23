import sys
sys.path.insert(0, "..")
import pictureweb.distributed as distributed
from pictureweb.distributed import kernel
from pictureweb.distributed import bcd
from pictureweb.distributed.distributed import fast_kernel_column_block_async
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


def solve_fn(train_key, train_labels_key, test_key, test_labels_key, bucket, lambdav, blocks_per_iter, epochs, num_classes, eval_interval, start_block, start_epoch, warm_start, prev_yhat, num_test_blocks, sheet):
    client = boto3.client('s3')
    resp = client.get_object(Key=train_labels_key, Bucket=bucket)
    bio = io.BytesIO(resp["Body"].read())
    y_train = np.load(bio).astype('int')
    y_train_enc = np.eye(num_classes)[y_train]
    N_train = y_train.shape[0]
    K_train = ShardedSymmetricMatrix(train_key, bucket=bucket)
    if (test_key != None and test_labels_key != None):
        resp = client.get_object(Key=test_labels_key, Bucket=bucket)
        bio = io.BytesIO(resp["Body"].read())
        y_test = np.load(bio)
        N_test = y_test.shape[0]

        K_test = ShardedMatrix(test_key, bucket=bucket, shape=(N_train, N_test), shard_size_0=4096, shard_size_1=4096)
        if (num_test_blocks == None):
            num_test_blocks = len(K_test._block_idxs(1))
        test_size = min(num_test_blocks*K_test.shard_sizes[1], N_test)
        print("Test Size", test_size)
    else:
        K_test = None
        y_test = None

    def eval_fn(model, y_hat, y, lambdav, block, epoch, iter_time):
        print("Evaluating and saving result to s3")
        # save shit to s3
        # update gspread?
        print(model.dtype)

        if (K_test != None and y_test != None):
            if ((block == 0 and epoch == 0) or (not os.path.exists("/dev/shm/K_test_block"))):

                futures = fast_kernel_column_block_async(K_test, col_blocks=K_test._block_idxs(1)[:num_test_blocks], mmap_loc="/dev/shm/K_test_block", dtype="float64")
                print("Downloading test block")
                fs.wait(futures)

            K_test_block = bcd.load_mmap("/dev/shm/K_test_block", (K_test.shape[0], test_size), "float64")
            model = model.astype('float64')
            print("K_Test BLOCK SHAPE ", K_test_block.shape)
            y_test_pred = K_test_block.T.dot(model)
            print(y_test_pred.shape)
            test_top1 = ls.top_k_accuracy(y_test[:test_size], y_test_pred, k=1)
            test_top5 = ls.top_k_accuracy(y_test[:test_size], y_test_pred, k=5)
            print("Top 1 test {0}, Top 5 test {1}".format(test_top1, test_top5))
        else:
            test_top1 = "NA"
            test_top5 = "NA"

        print("Calculate objective value")
        tr_wTKw = np.sum(model * y_hat)
        w_2 = np.linalg.norm(model)
        tr_yw = np.sum(model * y)
        obj = 0.5*tr_wTKw + 0.5*lambdav*((w_2)**2) - tr_yw
        print("Objective value {0}".format(obj))


        print("Calculating top 5 and top 1 accuracies")
        train_top1 = ls.top_k_accuracy(y_train, y_hat, k=1)
        train_top5 = ls.top_k_accuracy(y_train, y_hat, k=5)
        matrix_name = K_train.key
        client = boto3.client('s3')

        print("Updating sheet")
        # update google sheet
        scope = ['https://spreadsheets.google.com/feeds']
        credentials = ServiceAccountCredentials.from_json_keyfile_name('/tmp/gspread.json', scope)
        gc = gspread.authorize(credentials)

        # Hard coded
        bcd_sheet = gc.open("Imagenet DeathMarch 2017").worksheet(sheet)
        bcd_sheet.append_row([matrix_name, lambdav, epoch, block, blocks_per_iter, obj, train_top1, test_top1, train_top5, test_top5, iter_time])


        print("Serializing and uploading models")
        os.system("rm -rf /tmp/models/*")

        key_base = "{0}/lambdav_{7}_trt1_{1}_trt5_{2}_tt1_{3}_tt5_{4}_epoch_{5}_block_{6}".format(matrix_name, train_top1, train_top5, test_top1, test_top5, epoch, block, lambdav)
        model_key = "models/" + key_base + ".model"
        yhat_key = "models/" + key_base + ".y_train_hat"
        os.system("mkdir -p \"/tmp/models/{0}\"".format(K_train.key))
        np.save("/tmp/"+ model_key, model)
        np.save("/tmp/"+ yhat_key,  y_hat)

        cmd1 = "aws s3 sync /tmp/models s3://{0}/models"
        FNULL = open(os.devnull, 'w')
        p = Popen(cmd1.split(" "),  stdout=FNULL, stderr=subprocess.STDOUT)
        return 0

    if (prev_yhat != None):
        prev_yhat = np.load(prev_yhat)

    if (warm_start != None):
        warm_start = np.load(warm_start)

    bcd.block_kernel_solve(K_train, y_train_enc, epochs=epochs, lambdav=lambdav, blocks_per_iter=blocks_per_iter, eval_fn=eval_fn, eval_interval=eval_interval, start_block=start_block, start_epoch=start_epoch, y_hat=prev_yhat, warm_start=warm_start, num_blocks=len(K_train._blocks(0)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve ')
    parser.add_argument('train_key', type=str, help="S3 Key to sharded train matrix")
    parser.add_argument('train_labels_key', type=str, help="S3 Key to sharded train matrix")
    parser.add_argument('--test_key', type=str, help="S3 Key to sharded test matrix", default=None)
    parser.add_argument('--test_labels_key', type=str, help="S3 Key to sharded test matrix", default=None)
    parser.add_argument('--bucket', type=str, help="S3 bucket where sharded matrices live", default="vaishaalpywrenlinalg")
    parser.add_argument('--lambdav', type=float, help="regularization value", default=1e-5)
    parser.add_argument('--blocks_per_iter', type=int, help="regularization value", default=1)
    parser.add_argument('--epochs', type=int, help="regularization value", default=1)
    parser.add_argument('--start_epoch', type=int, help="start epoch", default=0)
    parser.add_argument('--start_block', type=int, help="start block", default=0)
    parser.add_argument('--num_classes', type=str, help="regularization value", default=1000)
    parser.add_argument('--warm_start', type=str, help="path to warm start numpy array", default=None)
    parser.add_argument('--prev_yhat', type=str, help="path to a previous y_hat", default=None)
    parser.add_argument('--eval_interval', type=int, help="how often do you want to evaluate and update s3 with results", default=10)
    parser.add_argument('--num_test_blocks', type=int, help="how many blocks to test", default=None)
    parser.add_argument('--sheet', type=str, help="which sheet to update", default="bcd_auto")
    args = parser.parse_args()
    solve_fn(args.train_key, args.train_labels_key, args.test_key, args.test_labels_key, args.bucket, args.lambdav, args.blocks_per_iter, args.epochs, args.num_classes, args.eval_interval, args.start_block, args.start_epoch, args.warm_start, args.prev_yhat, args.num_test_blocks, args.sheet)







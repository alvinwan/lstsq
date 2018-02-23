import boto3
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import sys
sys.path.insert(0, "..")
from distributed import sharded_matrix
import distributed.distributed as D 
from loaders.imagenet_load import orient
from conv import coates_ng_help
from importlib import reload
import io
import scipy.linalg
from sklearn import metrics
from opt import ls
import numpy as np
import os
import concurrent.futures as fs
import time
import logging
import watchtower
from scipy.linalg import LinAlgError

def get_feature_parameters(sharded_matrix):
    split_key = X_train_sharded.key.split("_")[1:]
    fp = {}
    logger.info(split_key)
    num_filters = int(split_key[0])
    fp["num_filters"] = num_filters
    patch_size = int(split_key[1])
    fp["patch_size"] = patch_size
    patch_stride = int(split_key[2])
    fp["patch_stride"] = patch_stride
    pool_size = int(split_key[3])
    fp["pool_size"] = pool_size
    pool_stride = int(split_key[4])
    fp["pool_stride"] = pool_stride
    bias = float(split_key[5])
    fp["bias"] = bias
    fp["random_seed"] = int(split_key[6].split("(")[0])
    fp["num_features"] = sharded_matrix.shape[1]
    return fp
    
def reset_msg_visibility(msg, not_done, logger):
    logger.debug("Starting message visbility resetter..")
    while(not_done[0]):
        time.sleep(60)
        logger.debug("RESETTING MESSAGE VISIBILITY.. for message {0}".format(msg.body))
        msg.change_visibility(VisibilityTimeout=120)

executor = fs.ThreadPoolExecutor(1)
print("HELLOOO")

while(True):
    print("Starting solve")
    client = boto3.client('s3')
    resp = client.get_object(Key="scrambled_train_labels.npy", Bucket="vaishaalpywrenlinalg")
    bio = io.BytesIO(resp["Body"].read())
    y_train = np.load(bio).astype('int')
    y_train_enc = np.eye(1000)[y_train]

    client = boto3.client('s3')
    resp = client.get_object(Key="scrambled_test_labels.npy", Bucket="vaishaalpywrenlinalg")
    bio = io.BytesIO(resp["Body"].read())
    y_test = np.load(bio).astype('int')

# Get the queue
    sqs = boto3.resource('sqs')
    queue = sqs.get_queue_by_name(QueueName='picturewebsolve')

    solve_msg = queue.receive_messages(MaxNumberOfMessages=1)
    if (solve_msg == []):
        time.sleep(30)
        continue

    solve_matrix =json.loads(solve_msg[0].body)


    X_train_sharded = sharded_matrix.ShardedMatrix(solve_matrix["train"], bucket="picturewebhyperband")
    X_test_sharded = sharded_matrix.ShardedMatrix(solve_matrix["test"], bucket="picturewebhyperband")
    log_key = X_train_sharded.key.replace("(", "__").replace(")", "__")
    cw_handler = watchtower.CloudWatchLogHandler("hyperbandsolve")

    logger = logging.getLogger(log_key)
    logger.setLevel('INFO')
    print("NUM FEATURES {0}".format(X_train_sharded.shape[1]))
    print("Solving {0}".format(solve_matrix))
    consoleHandler = logging.StreamHandler()
    logger.addHandler(cw_handler)

    not_done = [True]
    executor.submit(reset_msg_visibility, solve_msg[0], not_done, logger)



    logger.info("Solving {0}".format(solve_matrix))
    logger.info("NUM FEATURES {0}".format(X_train_sharded.shape[1]))
    X_train = D.get_local_matrix(X_train_sharded, dtype="float64", mmap_loc="/dev/shm/X_train")
    X_test = D.get_local_matrix(X_test_sharded, dtype="float64", mmap_loc="/dev/shm/X_test")

    XTX = X_train.T.dot(X_train)
    XTy = X_train.T.dot(y_train_enc)


    diag_idxs = np.diag_indices(XTX.shape[0])

    reg = 1e-8

    try:
        XTX[diag_idxs] += reg
        model = scipy.linalg.solve(XTX, XTy, sym_pos=True)
        XTX[diag_idxs] -= reg
    except LinAlgError as e:
        logger.error("Singular Matrix {0}".format(X_train_sharded.key))
        not_done[0] = False
        os.system("rm -rf /dev/shm/*")
        continue



    y_hat_train = X_train.dot(model)
    y_hat_test = X_test.dot(model)

    top_1_train = ls.top_k_accuracy(y_train, y_hat_train, k=1)
    top_5_train = ls.top_k_accuracy(y_train, y_hat_train, k=5)

    top_1_test = ls.top_k_accuracy(y_test, y_hat_test, k=1)
    top_5_test = ls.top_k_accuracy(y_test, y_hat_test, k=5)

    results = {}
    results["train_top_1"] = top_1_train
    results["train_top_5"] = top_5_train
    results["test_top_1"] = top_1_test
    results["test_top_5"] = top_5_test
    logger.info("Solve Completed!")
    logger.info(results)
    print("Results {0}".format(results))


    solve_params = {"solver": "direct_linear", "regularization": reg}
    feature_params = get_feature_parameters(X_train_sharded)
    from utils import exputil
    reload(exputil)
    exputil.save_results(solve_params, feature_params, results, "hyperband", repo_path="/home/ubuntu/pictureweb")
    not_done[0] = False
    solve_msg[0].delete()
    os.system("rm -rf /dev/shm/*")

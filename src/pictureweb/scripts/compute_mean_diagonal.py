import sys
sys.path.insert(0, "..")
from pictureweb.distributed import kernel, matmul
from pictureweb.distributed.sharded_matrix import ShardedMatrix, ShardedSymmetricMatrix

import argparse
import pywren
import time
import numpy as np


def compute_mean_diagonal(matrix_key, bucket):
    X = ShardedSymmetricMatrix(matrix_key, bucket=bucket)
    diag_sum = 0
    tot = 0
    for i in X._block_idxs(0):
        xbb = np.diag(X.get_block(i,i))
        diag_sum += np.sum(xbb)
        tot += xbb.shape[0]
        print(diag_sum/tot)

    print("Mean Diagonal entry: ", diag_sum/X.shape[0])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Symmetric Sharded Matrix Diagonol')
    parser.add_argument('matrix_key', type=str, help="S3 Key to symmetric sharded matrix")
    parser.add_argument('--bucket', type=str, help="S3 bucket symmetric sharded matrix", default="picturewebsolve")
    args = parser.parse_args()
    compute_mean_diagonal(args.matrix_key, args.bucket)







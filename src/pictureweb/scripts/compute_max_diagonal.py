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
    max_diag = 0
    for i in X._block_idxs(0):
        max_diag = max(max_diag, np.max(np.diag(X.get_block(i,i))))
        print("{0}/{1}".format(i+1, len(X._block_idxs(0))), max_diag)

    print("Mean Diagonal entry: ", max_diag)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute Symmetric Sharded Matrix Diagonol')
    parser.add_argument('matrix_key', type=str, help="S3 Key to symmetric sharded matrix")
    parser.add_argument('--bucket', type=str, help="S3 bucket symmetric sharded matrix", default="picturewebsolve")
    args = parser.parse_args()
    compute_mean_diagonal(args.matrix_key, args.bucket)







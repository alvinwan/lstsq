import sys
sys.path.insert(0, "..")
import distributed.distributed as D
import distributed.sharded_matrix as SM
import argparse
import numpy as np

def sharded_matrix_diff(X_sharded_1, X_sharded_2):
    X1 = D.get_local_matrix(X_sharded_1)
    X2 = D.get_local_matrix(X_sharded_2)
    diff = np.abs(X1 - X2)
    return np.mean(diff), np.max(diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='check if two sharded matrices are equal (downloads them both!)')
    parser.add_argument('matrix_1', type=str, help="S3 Key to sharded matrix 1")
    parser.add_argument('matrix_2', type=str, help="S3 Key to sharded matrix 2")
    parser.add_argument('--bucket', type=str, help="S3 bucket", default="vaishaalpywrenlinalg")
    parser.add_argument('--symmetric', action='store_const', const=True, default=False)
    args = parser.parse_args()
    print(args.symmetric)

    if (args.symmetric):
        Matrix  = SM.ShardedSymmetricMatrix
    else:
        Matrix  = SM.ShardedMatrix

    X1 = Matrix(args.matrix_1, bucket=args.bucket)
    X2 = Matrix(args.matrix_2, bucket=args.bucket)
    avg_diff, max_diff = sharded_matrix_diff(X1, X2)
    print("Matrix have an average (L1) difference of {0}, and (max) difference of {1}".format(avg_diff, max_diff))





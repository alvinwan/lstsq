import sys
sys.path.insert(0, "..")
from pictureweb.distributed import kernel, matmul
from pictureweb.distributed.sharded_matrix import ShardedMatrix, ShardedSymmetricMatrix
import pywren.wrenconfig as wrenconfig
import argparse
import pywren
import time


def compute_kernel(train_key, test_key, linear_kernel_key, gamma, in_bucket, out_bucket, tasks_per_job, max_num_jobs, job_max_runtime, pywren_mode, kernel_type, overwrite, local, out_key, matmul_async):
    print(pywren_mode)
    start = time.time()
    X_train_sharded = ShardedMatrix(train_key, bucket=in_bucket)

    if (test_key == None):
        X_test_sharded = X_train_sharded
    else:
        X_test_sharded = ShardedMatrix(test_key, bucket=in_bucket)
    X_test_sharded.get_block(0,0)
    if (X_train_sharded.key == X_test_sharded.key):
        symmetric = True
        Matrix  = ShardedSymmetricMatrix
    else:
        symmetric = False
        Matrix  = ShardedMatrix

    if (pywren_mode == "lambda"):
        config = wrenconfig.default()
        config['runtime']['s3_bucket'] = "ericmjonas-public"
        config['runtime']['s3_key'] = "pywren.runtime/pywren_runtime-3.6-minimal.tar.gz"
        pwex = pywren.default_executor(job_max_runtime=job_max_runtime)
    elif (pywren_mode == "standalone"):
        config = wrenconfig.default()
        config['runtime']['s3_bucket'] = "pictureweb"
        config['runtime']['s3_key'] = "pywren.runtime/pywren_runtime-3.6-pictureweb.tar.gz"
        pwex = pywren.standalone_executor(job_max_runtime=job_max_runtime)
    else:
        raise Exception("Invalid pywren mode")

    if (linear_kernel_key == None):
        print("First computing linear kernel")
        start = time.time()
        print("Overwrite", overwrite)
        linear_kernel = matmul.compute_XYT_pywren(pwex, X=X_train_sharded, Y=X_test_sharded, tasks_per_job=tasks_per_job, num_jobs=max_num_jobs, out_bucket=out_bucket, overwrite=overwrite, local=local, out_key=out_key, pywren_mode=pywren_mode)
        end = time.time()
        print("Linear Kernel took {0} seconds to compute".format(end - start))
    else:
        linear_kernel = Matrix(linear_kernel_key, bucket=out_bucket, shape=(X_train_sharded.shape[0], X_test_sharded.shape[0]), shard_size_0=X_train_sharded.shard_size_0, shard_size_1=X_train_sharded.shard_size_0)

    linear_kernel.get_block(0,0)


    assert(linear_kernel.shape == (X_train_sharded.shape[0], X_test_sharded.shape[0]))
    if (kernel_type == "linear"):
        return linear_kernel
    elif (kernel_type == "rbf"):
        print("Computing RBF Kernel")
        RBF_kernel = kernel.compute_rbf_kernel_pywren(pwex, linear_kernel, X_test_sharded,\
                                                            X_train_sharded,\
                                                            gamma=gamma,\
                                                            tasks_per_job=tasks_per_job,\
                                                            num_features=X_test_sharded.shape[1],\
                                                            num_jobs=max_num_jobs, matmul_async=matmul_async)
        end = time.time()
        print("Done computing RBF Kernel, took {0} seconds".format(end - start))
        return RBF_kernel
    elif (kernel_type == "quadratic"):
        quadratic_kernel = kernel.compute_quadratic_kernel_pywren(pwex, linear_kernel, X_test_sharded,\
                                                            X_train_sharded,\
                                                            gamma=gamma,\
                                                            tasks_per_job=tasks_per_job,\
                                                            num_jobs=max_num_jobs)
        end = time.time()
        print("Done computing quadratic Kernel, took {0} seconds".format(end - start))
        return quadratic_kernel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute a kernel using pywren')
    parser.add_argument('train_key', type=str, help="S3 Key to sharded train matrix")
    parser.add_argument('--gamma', type=float, help="Inverse of kernel bandwidth parameter (1/sigma^{2} = gamma))", default=0.001)
    parser.add_argument('--test_key', type=str, help="S3 Key to sharded test matrix", default=None)
    parser.add_argument('--out_key', type=str, help="S3 Key to output sharded test matrix", default=None)
    parser.add_argument('--tasks_per_job', type=int, help="Number of blocks to be processed by one pywren job", default=4)
    parser.add_argument('--max_num_jobs', type=int, help="Maximum number of pywren jobs to go out at once", default=4500)
    parser.add_argument('--job_max_runtime', type=int, help="Maximum run time of a pywren job", default=600)
    parser.add_argument('--in_bucket', type=str, help="S3 bucket where input sharded matrices live", default="vaishaalpywrenlinalg")
    parser.add_argument('--out_bucket', type=str, help="S3 bucket where output sharded matrices go", default="vaishaalpywrenlinalg")
    parser.add_argument('--linear_kernel_key', type=str, help="S3 Key to sharded linear_kernel")
    parser.add_argument('--pywren_mode', type=str, help="Mode to run pywren in (lambda, standalone)", default="lambda")
    parser.add_argument('--type', type=str, default="linear")
    parser.add_argument('--overwrite', action='store_const', const=True, default=False, help="Overwrite the matrix if it exists" )
    parser.add_argument('--local', action='store_const', const=True, default=False, help="run map locally" )
    parser.add_argument('--matmul_async', action='store_const', const=True, default=False, help="async matmul" )
    args = parser.parse_args()
    compute_kernel(args.train_key, args.test_key, args.linear_kernel_key, args.gamma, args.in_bucket, args.out_bucket, args.tasks_per_job, args.max_num_jobs, args.job_max_runtime, args.pywren_mode, args.type, args.overwrite, args.local, args.out_key, args.matmul_async)









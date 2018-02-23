import sys
sys.path.insert(0, "../../")
import pictureweb.distributed.sharded_matrix as SM
import concurrent.futures as fs
import tornado.ioloop
import tornado.web
import time
import boto3
import os
import traceback

matrices = []
progress = []
def list_all_folders(bucket, prefix):
    client = boto3.client('s3')
    objects = client.list_objects(Bucket=bucket, Prefix=prefix, Delimiter="/")
    return list(map(lambda x: x['Prefix'], objects['CommonPrefixes']))

def list_sharded_matrices(folders, prefix, bucket, num_workers=32, executor=None):
    futures = []
    if (executor == None):
        executor = fs.ProcessPoolExecutor(num_workers)
    for folder in folders:
        key = os.path.split(folder[:-1])[1]
        prefix = os.path.split(folder[:-1])[0]
        prefix += "/"
        future = executor.submit(SM.ShardedMatrix, key, bucket=bucket, prefix=prefix)
        futures.append(future)
    fs.wait(futures)

    matrices = []
    for f in futures:
        try:
            matrices.append(f.result())
        except:
            pass
    return matrices

def get_feature_parameters(sharded_matrix):
    split_key = sharded_matrix.key.split("_")
    for i,elem in enumerate(split_key):
        try:
            int(elem)
            break
        except:
            raise
            pass

    split_key = sharded_matrix.key.split("_")[i:]
    fp = {}
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

def group_matrices_into_train_test(matrices):
    test_matrices = list(filter(lambda x: "test" in x.key, matrices))
    train_matrices = list(filter(lambda x: "train" in x.key, matrices))
    train_sorted = sorted(train_matrices, key=lambda x: x.key.replace("train", ""))
    test_sorted = sorted(test_matrices, key=lambda x: x.key.replace("test", ""))
    for test,train in zip(train_matrices, test_sorted):
        fp_test = get_feature_parameters(test)
        fp_train = get_feature_parameters(train)
        assert(fp_test == fp_train)

    return list(zip(train_sorted, test_sorted))


def grab_matrix_progress(matrix):
    all_blocks = matrix.blocks
    blocks_exist = matrix.blocks_exist
    return (len(blocks_exist),len(all_blocks))

def grab_all_matrices_progress(matrices, executor=None, num_workers=32):
    if (executor == None):
        executor = fs.ProcessPoolExecutor(num_workers)
    futures = []
    for m in matrices:
        future = executor.submit(grab_matrix_progress, m)
        futures.append(future)
    fs.wait(futures, return_when=fs.FIRST_EXCEPTION)
    return [f.result() for f in futures]

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        filter_str = self.get_argument("filter", default="", strip=True)
        full = bool(int(self.get_argument("full", default="0", strip=True)))
        self.write(get_progress(filter_str, full))


def make_app():
    return tornado.web.Application([
    (r"/", MainHandler),
    ])

def get_progress(filter_str, full):
    out = ""
    print(full)
    print(filter_str)
    print(matrices)
    print(progress)
    for m,p in zip(matrices, progress):
        out += "{0} ----- {1}/{2}<br>".format(m.key, p[0], p[1])
    return out

def load_matrix_progress():
    global matrices
    global progress
    while(True):
        try:
            print("Starting progress check")
            folders = list_all_folders("picturewebsolve", "pywren.linalg/")
            print("Got folders")
            matrices = list_sharded_matrices(folders, "pywren.linalg", "picturewebsolve")
            print("Got matrices")
            print(matrices)
            progress = grab_all_matrices_progress(matrices)
            print("Got progress")
            print(folders)
        except Exception as err:
            raise



if __name__ == "__main__":
    executor = fs.ThreadPoolExecutor(1)
    future = executor.submit(load_matrix_progress)

    print("starting server...")
    app = make_app()
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()



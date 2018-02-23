import numpy as np
from multigpu import GpuHandler, MultiGpuHandler
from filter_gen import make_gaussian_filter_gen
import conv

NUM_GPUS = 16

def simple_function_no_gpu(x):
    return np.exp(x)

def simple_function_gpu(x):
    from theano import function, config, shared, tensor
    x = shared(x)
    f = function([], tensor.exp(x))
    exp_x = f()
    used_gpu = np.any([isinstance(x.op, tensor.Elemwise) and
     	       ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()])
    return exp_x, used_gpu

def test_single_gpu_handler_simple():
    gpu_handler = GpuHandler(0)
    gpu_handler.start()
    gpu_handler.wait_for_gpu_init()
    assert(gpu_handler.submit(simple_function_no_gpu, 2) == np.exp(2))

def test_single_gpu_handler_gpu_func():
    gpu_handler = GpuHandler(0)
    gpu_handler.start()
    gpu_handler.wait_for_gpu_init()
    exp_x, used_gpu = gpu_handler.submit(simple_function_gpu, 2)
    assert(np.isclose(exp_x, np.exp(2)) and used_gpu)

def test_multi_gpu_handler_simple():
    gpu_handler = MultiGpuHandler(NUM_GPUS)
    gpu_handler.start_and_wait_for_gpu_init()
    for gpu in gpu_handler.gpus:
        assert(np.exp(2) == gpu.submit(simple_function_no_gpu, 2))


def test_multi_gpu_handler_gpu_func():
    gpu_handler = MultiGpuHandler(NUM_GPUS)
    gpu_handler.start_and_wait_for_gpu_init()
    for gpu in gpu_handler.gpus:
        exp_x, used_gpu = gpu.submit(simple_function_gpu, 2)
        assert(np.isclose(exp_x, np.exp(2)) and used_gpu)


def test_single_gpu_conv():
    gpu_handler = GpuHandler(0)
    gpu_handler.start()
    gpu_handler.wait_for_gpu_init()
    X = np.random.randn(32, 3, 256, 256).astype('float32')
    fg = make_gaussian_filter_gen(1.0, patch_size=11, seed=0)
    X = gpu_handler.submit(conv._conv, X, fg, 32, 1, 64, pool_size=4, pool_stride=4, pad=0, bias=1.0, patch_size=11, conv_stride=4)




def test_multiple_gpu_conv():
    gpu_handler = MultiGpuHandler(NUM_GPUS)
    gpu_handler.start_and_wait_for_gpu_init()
    X = np.random.randn(32, 3, 256, 256).astype('float32')
    fg = make_gaussian_filter_gen(1.0, patch_size=11, seed=0)
    futures = []
    start = time.time()
    for gpu in gpu_handlers.gpus:
        X_future = gpu.submit_async(conv._conv, X, fg, 32, 1, 64, pool_size=4, pool_stride=4, pad=0, bias=1.0, patch_size=11, conv_stride=4)
        futures.append(X_future)

    results = []
    for future in futures:
        results.append(future.result())
    end = time.time()
    tot_time = end - start
    print("Total time was {0}".format(tot_time))
    np.vstack(results)












import os
import multiprocessing as mp
import logging
import dill
import traceback
import sys
import numpy as np



def KILL_GPU_PROCESS(gpu_id):
    return os.system("kill -9 $(nvidia-smi | awk '$2==\"Processes:\" {p=1} p && $2 == " + str(gpu_id) +  "&& $3 > 0 {print $3}') > /dev/null 2>&1")

class GpuResult(object):
    ''' Simple async wrapper for gpu results '''
    def __init__(self, result_conn):
        self.result_conn = result_conn
        self._result = None
    def result(self):
        if (self._result != None):
            ret = self._result
        else:
            ret = self.result_conn.recv()
            self.result = ret
        if (isinstance(ret, MmapArray)):
            return ret.load()
        if isinstance(ret, Exception):
            raise ret
        else:
            return ret

class MmapArray():
    def __init__(self, mmaped, mode=None,idxs=None):
        self.loc = mmaped.filename
        self.dtype = mmaped.dtype
        self.shape = mmaped.shape
        self.mode = mmaped.mode
        self.idxs = idxs
        if (mode != None):
            self.mode = mode

    def load(self):
        X = np.memmap(self.loc, dtype=self.dtype, mode=self.mode, shape=self.shape)
        if self.idxs != None:
            return X[self.idxs[0]:self.idxs[1]]
        else:
            return X





class MultiGpuHandler(object):
    ''' Handler class for multiple gpus '''
    def __init__(self, num_gpus):
        self.gpus = [GpuHandler(i) for i in range(num_gpus)]

    def start_all(self):
        for gpu in self.gpus:
            gpu.start()

    def wait_for_all_gpu_init(self):
        for gpu in self.gpus:
            gpu.wait_for_gpu_init()

    def start_and_wait_for_gpu_init(self):
        self.start_all()
        self.wait_for_all_gpu_init()

    def kill_all_gpu_processes(self):
        for i in range(len(self.gpus)):
            KILL_GPU_PROCESS(i)

class GpuHandler(object):
    ''' Handler class for one gpu handling process
        this process will be bound to one gpu for its
        entire life
    '''
    def __init__(self, gpu_id):
        self.parent_pid = os.getpid()
        self.manager = mp.Manager()
        self.q_in = self.manager.Queue()
        conn_in, conn_out = mp.Pipe()
        self.conn_in = conn_in
        self.conn_out = conn_out
        self.p = mp.Process(target=self.handler)
        self.gpu_id = gpu_id


    def start(self):
        self.p.start()

    def wait_for_gpu_init(self):
        self.conn_out.recv()

    def __gpu_init(self):
        import os
        compile_dir ='base_compiledir=/tmp/.theano_{0}/'.format(self.gpu_id)
        os.environ['THEANO_FLAGS'] = '{0},device=gpu{1}'.format(compile_dir,self.gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        # GPU INIT HAPPENS HERE
        self.conn_in.send(1)

    def __pack_args(self, args):
        args = list(args)
        packed = []
        for i,a in enumerate(args):
            if (callable(a)):
                packed.append(i)
                args[i] = dill.dumps(a)
        args = tuple(args)
        return args, packed

    def __pack_kwargs(self, kwargs):
        packed = []
        for k, arg in kwargs.items():
            if (callable(arg)):
                packed.append(k)
                kwargs[k] = dill.dumps(arg)
        return kwargs, packed

    def __unpack_args(self, args, packed):
        args = list(args)
        for i in packed:
            args[i] = dill.loads(args[i])
        args = tuple(args)
        return args

    def __unpack_kwargs(self, kwargs, packed):
        for k in packed:
            kwargs[k] = dill.loads(kwargs[k])
        return kwargs

    def __submit(self, f, *args, **kwargs):
        conn1, conn2 = mp.Pipe()
        f_bytes = dill.dumps(f)
        args, packed = self.__pack_args(args)
        kwargs, packed_kwargs = self.__pack_kwargs(kwargs)
        self.q_in.put((conn2,packed,packed_kwargs,f_bytes,args,kwargs))
        return conn1

    def submit(self, f, *args, **kwargs):
        ret = self.__submit(f, *args, **kwargs).recv()
        if (isinstance(ret, MmapArray)):
            return ret.load()
        if isinstance(ret, Exception):
            raise ret
        else:
            return ret


    def submit_async(self, f, *args, **kwargs):
        return GpuResult(self.__submit(f, *args, **kwargs))

    def cancel():
        ''' Note this will invalidate all live futures '''
        self.p.terminate()

    def handler(self):
        if (os.getpid() == self.parent_pid):
            raise Exception("Handler function cannot be called from parent process")
        self.__gpu_init()
        while(True):
            conn2, packed_args, packed_kwargs, f_bytes, args, kwargs = self.q_in.get()
            args = self.__unpack_args(args, packed_args)
            kwargs = self.__unpack_kwargs(kwargs, packed_kwargs)
            f = dill.loads(f_bytes)
            try:
                ret = f(*args, **kwargs)
                conn2.send(ret)
            except:
                e = Exception("".join(traceback.format_exception(*sys.exc_info())))
                conn2.send(e)













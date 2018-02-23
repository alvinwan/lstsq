import logging
import numpy as np
import math
import datetime
import os
import watchtower

from importlib import reload
from . import multigpu

def conv_mmap(*args,
         **kwargs):
        X = kwargs["data"].load()
        X_out = kwargs["output"].load()
        if (kwargs.get("theano") != None):
             conv_fn = _conv
        else:
             conv_fn = _conv_tf
        X_out_local, filters = conv_fn(X,
                      filter_gen=kwargs["filter_gen"],
                      num_feature_batches=kwargs["num_feature_batches"],
                      feature_batch_size=kwargs["feature_batch_size"],
                      data_batch_size=kwargs["data_batch_size"],
                      pool_type=kwargs["pool_type"],
                      pool_size=kwargs["pool_size"],
                      pool_stride=kwargs["pool_stride"],
                      pad=kwargs["pad"],
                      bias=kwargs["bias"],
                      patch_size=kwargs["patch_size"],
                      conv_stride=kwargs["conv_stride"],
                      preprocess_batch=kwargs.get("preprocess_batch")
                      )
        if (os.environ.get("CUDA_VISIBILE_DEVICES") == None):
            device = "cpu"
        else:
            device = "gpu" + str(os.environ.get("CUDA_VISIBLE_DEVICES"))
        logger.info("IN CONV MMAP", extra={"gpu": device})
        np.copyto(X_out, X_out_local)
        X_out.flush()
        return kwargs["output"]

def conv_compute_output_shape(data,
         feature_batch_size,
         num_feature_batches,
         data_batch_size,
         pool_type='avg',
         pool_size=14,
         pool_stride=14,
         pad=0,
         bias=1.0,
         patch_size=6,
         conv_stride=1,
         *args,
         **kwargs):

         strideStart = pool_size/2.0
         # TODO: Is this correct?

         outX = int(math.ceil(((data.shape[2] - patch_size + 1)/conv_stride - pool_size)/float(pool_stride))) + 1
         outY = int(math.ceil(((data.shape[3] - patch_size + 1)/conv_stride - pool_size)/float(pool_stride))) + 1
         outFilters = 2*feature_batch_size*num_feature_batches
         return (data.shape[0], outFilters, outX, outY)

def _conv(data,
         filter_gen,
         feature_batch_size,
         num_feature_batches,
         data_batch_size,
         pool_type='avg',
         pool_size=14,
         pool_stride=14,
         pad=0,
         bias=1.0,
         patch_size=6,
         conv_stride=1,
         *args,
         **kwargs):
    '''
        Low level conv interface using custom pylearn2 (which exports AvgPool) and cuda convnet through Theano.
        Provides basic one layer convolution interface for Coates-Ng random network.

        Huge caveats: This code heavily depends on running exactly Theano 0.8.2 and having my custom pylearn2 fork
        (https://github.com/Vaishaal/pylearn2)
        hopefully these will alleviated in a future revision

        Architechture is: Convolution -> SymmetricRelu -> Pool

        TODO: Rip out Theano, replace with pycuda
        @param data - Batch Size x Channels x Rows x Columns array of data (float32)
        @param filter_gen - A function such that filter_gen(N) produces N, patch_size x patch_size x 3 filters (only 3 channel images supported for now)
        @param feature_batch_size - Number of features to produce in one convolution operation in GPU (this will be memory constrainted)
        @param data_batch_size -  Number of data points to featurize at once in GPU (this will be memory constrainted)
        @param num_feature_batches - Number of feature batches to generate total
        @param patch_size - Convolution patch size (must be same size as output of filter_gen)
        @param pool_type - Average or Max Pool
        @param pool_size - Size of pool
        @param pool_size - Pool Stride
        @param bias - Relu Bias
    '''
    # Theano has stupid initialization code run at import so this is also a hack
    import theano
    from theano import function, config, shared, sandbox
    import theano.tensor as T
    from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
    from pylearn2.sandbox.cuda_convnet.pool import MaxPool, AvgPool
    from theano.sandbox.cuda.basic_ops import gpu_contiguous

    device = "gpu" + os.environ["CUDA_VISIBLE_DEVICES"] 
    logger = logging.getLogger(__name__ + device)
    logger.setLevel('DEBUG')
    logger.handlers = []
    now = str(datetime.datetime.now()).replace(" ", "-")
    device = theano.config.device
    pid = os.getpid()
    logfmt = '%(gpu)s - %(asctime)s - %(name)s - %(levelname)s - %(message)s'

    fh = logging.FileHandler("/tmp/{0}_conv_{1}.log".format(device,now))
    fh.setFormatter(logging.Formatter(logfmt))
    logger.addHandler(fh)
    fh.flush()
    def LOGINFO(msg):
        logger.info(msg, extra={"gpu":device})

    def LOGWARN(msg):
        logger.warn(msg, extra={"gpu":device})

    def LOGDEBUG(msg):
        logger.debug(msg, extra={"gpu":device})

    strideStart = pool_size/2.0
    # TODO: Is this correct?


    outX = int(math.ceil(((data.shape[2] - patch_size + 1)/conv_stride - pool_size)/float(pool_stride))) + 1
    outY = int(math.ceil(((data.shape[3] - patch_size + 1)/conv_stride - pool_size)/float(pool_stride))) + 1
    outFilters = 2*feature_batch_size*num_feature_batches
    LOGINFO("Begining Convolution output size is {0} x {1} x {2}".format(outX, outY, outFilters))

    XFinal = np.zeros((data.shape[0], outFilters, outX, outY), 'float32')
    XBlock = None
    FTheano = None
    filters = []
    numImages = data.shape[0]
    # Convert to cuda-convnet order
    data = data.transpose(1,2,3,0)

    # POOL OP CREATION
    if (pool_type == 'avg'):
        pool_op = AvgPool(ds=pool_size, stride=pool_stride)
    elif (pool_type == 'max'):
        pool_op = MaxPool(ds=pool_size, stride=pool_stride)
    else:
        raise Exception('Unsupported pool type')

    if (conv_stride == 1):
        conv_op = FilterActs(pad=pad)
    else:
        conv_op = FilterActs(pad=pad, stride=conv_stride)

    for j in range(num_feature_batches):
        F = filter_gen(feature_batch_size)
        F = F.transpose(1,2,3,0)
        CHANNEL_AXIS = 0
        filters.append(F)

        #TODO: Lol what a hack
        if (FTheano == None):
            FTheano = shared(F.astype('float32'))
        else:
            FTheano.set_value(F.astype('float32'))

        start_filters = j*feature_batch_size
        end_filters = (j+1)*feature_batch_size

        start_filters *= 2
        end_filters *= 2

        for i in range(int(np.ceil(numImages/float(data_batch_size)))):
                start = i*data_batch_size
                end = min((i+1)*data_batch_size, numImages)

                LOG_STR = "FEATURE BATCH # {0}, DATA BATCH {1}, SIZE IS {2}".format((j), i, end - start)
                LOGINFO(LOG_STR)
                LOGINFO(F.shape)

                XBlock_cpu = data[:, :, :, start:end].astype('float32')
                LOGINFO(XBlock_cpu.shape)

                if (XBlock == None):
                    XBlock = shared(XBlock_cpu)
                else:
                    XBlock.set_value(XBlock_cpu)

                # CONV
                XBlock_conv_out = conv_op(XBlock, FTheano)

                # RELU
                XBlock0 = T.nnet.relu(XBlock_conv_out - bias, 0)
                XBlock1 = T.nnet.relu(-1.0 * XBlock_conv_out - bias, 0)

                XBlock0 = pool_op(XBlock0)
                XBlock1 = pool_op(XBlock1)
                XBlockOut = np.concatenate((XBlock0.eval(), XBlock1.eval()), axis=CHANNEL_AXIS)

                XBlockOut = XBlockOut.transpose(3,0,1,2)
                F = F.transpose(3,0,1,2)
                XFinal[start:end,start_filters:end_filters,:,:] = XBlockOut

    filters = np.concatenate(filters,axis=0)

    # Does this do anything?
    XBlock.set_value([[[[]]]])
    FTheano.set_value([[[[]]]])
    return (XFinal, filters)


def _conv_tf(data,
         filter_gen,
         feature_batch_size,
         num_feature_batches,
         data_batch_size,
         pool_type='avg',
         pool_size=14,
         pool_stride=14,
         pad=0,
         bias=1.0,
         patch_size=6,
         conv_stride=1,
         preprocess_batch=None,
         *args,
         **kwargs):
    '''
        Low level conv interface using custom tensorflow
        Architechture is: Convolution -> SymmetricRelu -> Pool

        @param data - Batch Size x Channels x Rows x Columns array of data (float32)
        @param filter_gen - A function such that filter_gen(N) produces N, patch_size x patch_size x 3 filters (only 3 channel images supported for now)
        @param feature_batch_size - Number of features to produce in one convolution operation in GPU (this will be memory constrainted)
        @param data_batch_size -  Number of data points to featurize at once in GPU (this will be memory constrainted)
        @param num_feature_batches - Number of feature batches to generate total
        @param patch_size - Convolution patch size (must be same size as output of filter_gen)
        @param pool_type - Average or Max Pool
        @param pool_size - Size of pool
        @param pool_size - Pool Stride
        @param bias - Relu Bias
    '''
    import tensorflow as tf

    if (os.environ.get("CUDA_VISIBILE_DEVICES") == None):
        device = "cpu"
    else:
        device = "gpu" + str(os.environ.get("CUDA_VISIBLE_DEVICES"))

    if (kwargs.get("logkey") == None):
        logger = logging.getLogger(__name__ + device)
    else:
        logger = logging.getLogger(kwargs.get("logkey"))
    logger.setLevel('DEBUG')
    logger.handlers = []
    now = str(datetime.datetime.now()).replace(" ", "-")

    if (os.environ.get("CUDA_VISIBILE_DEVICES") == None):
        gpuid = "cpu"
    else:
        gpuid = "gpu" + str(os.environ.get("CUDA_VISIBLE_DEVICES"))

    pid = os.getpid()
    if (kwargs.get("logfmt") == None):
        logfmt = '%(gpu)s-%(levelname)s-%(message)s'
    else:
        logfmt = kwargs.get("logfmt")

    fh = logging.FileHandler("/tmp/{0}_conv_{1}.log".format(device,now))
    fh.setFormatter(logging.Formatter(logfmt))

    if (kwargs.get("log_handlers") != None):
        for handler in kwargs.get("log_handlers"):
            logger.addHandler(handler)

    logger.addHandler(fh)
    fh.flush()
    def LOGINFO(msg):
        logger.info(msg, extra={"gpu":device})

    def LOGWARN(msg):
        logger.warn(msg, extra={"gpu":device})

    def LOGDEBUG(msg):
        logger.debug(msg, extra={"gpu":device})

    strideStart = pool_size/2.0
    # TODO: Is this correct?

    N, outFilters, outX, outY = conv_compute_output_shape(data, feature_batch_size, num_feature_batches, data_batch_size, pool_type, pool_size, pool_stride, pad, bias, patch_size, conv_stride, *args, **kwargs)
    LOGINFO("Begining Convolution output size is {0} x {1} x {2}".format(outX, outY, outFilters))

    XFinal = np.zeros((N, outFilters, outX, outY), 'float32')
    XBlock = None
    FTF = None
    filters = []
    numImages = data.shape[0]
    # Convert to tf order
    data = data.transpose(0,2,3,1)

    strides =  [1, conv_stride, conv_stride, 1]

    session_conf = tf.ConfigProto(log_device_placement=True)
    F_tf = tf.placeholder(tf.float32, shape=(patch_size, patch_size, 3, None))
    XBlock_tf = tf.placeholder(tf.float32, shape=(None, ) + data.shape[1:])
    session_conf.gpu_options.allow_growth = True

    conv_1 = tf.nn.conv2d(XBlock_tf, F_tf, strides=strides, padding='VALID')
    conv_2  = conv_1 * -1
    h_conv1 = tf.nn.relu(tf.subtract(conv_1, bias))
    h_conv_2 = tf.nn.relu(tf.subtract(conv_2, bias))

    pool_pos = tf.nn.avg_pool(h_conv1, ksize=[1, pool_size, pool_size, 1],
         strides=[1, pool_stride, pool_stride, 1], padding='VALID')

    pool_neg = tf.nn.avg_pool(h_conv_2, ksize=[1, pool_size, pool_size, 1],
         strides=[1, pool_stride, pool_stride, 1], padding='VALID')

    results = tf.concat([pool_pos, pool_neg], axis=3)
    try:
        with tf.Session(config=session_conf) as sess:
            for j in range(num_feature_batches):
                    F_cpu = filter_gen(feature_batch_size)
                    F_cpu = np.ascontiguousarray(F_cpu.transpose(2,3,1,0))
                    CHANNEL_AXIS = 0
                    filters.append(F_cpu)
                    start_filters = j*feature_batch_size
                    end_filters = (j+1)*feature_batch_size
                    start_filters *= 2
                    end_filters *= 2
                    for i in range(int(np.ceil(numImages/float(data_batch_size)))):
                            start = i*data_batch_size
                            end = min((i+1)*data_batch_size, numImages)
                            LOG_STR = "FEATURE BATCH # {0}, DATA BATCH {1}, SIZE IS {2}".format((j), i, end - start)
                            LOGDEBUG(LOG_STR)
                            XBlock_cpu = np.ascontiguousarray(data[start:end, :, :, :])

                            if (preprocess_batch != None):
                                LOGDEBUG("PREPROCESSING USING PREPROCESSING FUNCTION CORRECTLY: {0}".format(preprocess_batch))
                                XBlock_cpu = preprocess_batch(XBlock_cpu, feature_batch=j, mode="HWC")
                            XBlock_cpu = XBlock_cpu.astype('float32')
                            XBlockOut = sess.run(results, feed_dict={XBlock_tf: XBlock_cpu, F_tf: F_cpu})
                            XBlockOut = np.ascontiguousarray(XBlockOut.transpose(0,3,1,2))
                            LOGDEBUG("BEFORE WRITING TO XFINAL")
                            XFinal[start:end,start_filters:end_filters,:,:] = XBlockOut
                            LOGDEBUG(XBlockOut.shape)



    except:
        raise
    LOGINFO("Finished Convolution output size is {0} x {1} x {2}".format(outX, outY, outFilters))
    return (XFinal, [])




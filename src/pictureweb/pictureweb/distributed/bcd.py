from ..utils import linalg
from .distributed import fast_kernel_column_block_async, load_mmap
from .sharded_matrix import ShardedMatrix, ShardedSymmetricMatrix
from . import matmul
import numpy as np
import pywren
from ..utils import misc
import math
import concurrent.futures as fs
import time
import os
from numba import jit
from sklearn import metrics
from scipy.linalg import LinAlgError
from sklearn.datasets import fetch_mldata
import watchtower
import logging
import sys
import gc

try:
    import scipy.linalg
except:
    pass

@jit(nopython=True)
def __calculate_res(R, num_blocks, b, K_block, x, block_size, n):
    for b2 in range(int(num_blocks)):
        if b2 == b: continue
        s =  b2*block_size
        e = min((b2+1)*block_size, n)
        if (np.all(x[s:e, :] == 0)): continue
        Kbb2 = K_block[s:e]
        a = np.dot(Kbb2.T, x[s:e, :])
        R += a
    return R

def block_idxs_to_idxs(block_idxs, block_size, n):
    idxs = []
    for block in block_idxs:
        sidx = block_size*block
        eidx = min((block_size)*(block+1), n)
        idxs.extend(list(range(sidx, eidx)))
    return idxs

def block_kernel_solve(K, y, epochs=1,\
                       max_iter=313,\
                       block_size=4096,\
                       num_blocks=313,\
                       lambdav=0.1,\
                       blocks_per_iter=1,\
                       verbose=True,\
                       warm_start = None,\
                       y_hat = None,\
                       special_case = False,\
                       workers=22,\
                       dtype="float64",\
                       eval_fn=None,\
                       eval_interval=20,\
                       start_epoch=0,\
                       start_block=0,\
                       seed=0, \
                       no_shuffle=False, \
                       with_replace=False):


        '''Solve (K + \lambdaI)x = y
            in a block-wise fashion
        '''
        labels = np.argmax(y, axis=1)
        log_key = K.key.replace("(", "__").replace(")", "__") + "lambda_" +  str(lambdav)
        logger = logging.getLogger(log_key)
        logger.setLevel('DEBUG')
        cw_handler = watchtower.CloudWatchLogHandler("bcd")
        cw_handler.setLevel(logging.DEBUG)
        terminal_handler = logging.StreamHandler(sys.stdout)
        terminal_handler.setLevel(logging.DEBUG)
        logger.addHandler(cw_handler)
        logger.addHandler(terminal_handler)

        with fs.ProcessPoolExecutor(max_workers=workers) as executor:

            ex = fs.ThreadPoolExecutor(8)
            mmap_locs = ["/dev/shm/block0", "/dev/shm/block1"]

            # compute some constants
            if (warm_start == None):
                x = np.zeros(y.shape, dtype)
            else:
                x = warm_start

            if (y_hat is None):
                y_hat = np.zeros(y.shape)
            else:
                y_hat = y_hat.copy()

            i = 0
            num_samples = K.shape[0]
            shuffled_coords = list(range(num_samples))
            logger.info(("Num Epochs", epochs))
            logger.info(("Num blocks", num_blocks))
            logger.info(("BLOCKS PER ITER", blocks_per_iter))
            eval_futures = []

            all_blocks = np.array(K._block_idxs(0))
            num_blocks = len(all_blocks)
            np.random.seed(seed)
            iters = 0
            for epoch in range(epochs):
                for i in range(100): gc.collect()
                mmap_loc = mmap_locs[((start_block - 1) % 2) == 0]
                if (iters > max_iter): break
                # are you going to sample from the blocks with or without replacement
                if (not with_replace):
                    if (no_shuffle):
                        shuffled_blocks = all_blocks
                    else:
                        idxs = np.random.choice(num_blocks, num_blocks, replace=False)
                        shuffled_blocks = all_blocks[idxs]

                    if (num_blocks%blocks_per_iter != 0):
                        r = blocks_per_iter - num_blocks % blocks_per_iter
                        extra_idxs = list(np.random.choice(num_blocks, r*4, replace=False))
                        for b in list(shuffled_blocks[-blocks_per_iter:]):
                            if (b in extra_idxs):
                                extra_idxs.remove(b)

                        extra_idxs = extra_idxs[:r]
                        shuffled_blocks = np.hstack((shuffled_blocks, all_blocks[extra_idxs]))
                        assert(len(shuffled_blocks)%blocks_per_iter == 0)

                    chunked_shuffled_blocks = list(misc.chunk(shuffled_blocks, blocks_per_iter))
                else:
                    print("With replacement!")
                    chunked_shuffled_blocks = []
                    for i in range(int(np.ceil(len(all_blocks)/blocks_per_iter))):
                        idxs = np.random.choice(num_blocks, blocks_per_iter, replace=False)
                        shuffled_blocks = all_blocks[idxs]
                        chunked_shuffled_blocks.append(shuffled_blocks)

                # zip past first start_epoch epochs
                if (epoch < start_epoch): continue
                start_epoch = 0

                K_block_futures = fast_kernel_column_block_async(K, chunked_shuffled_blocks[start_block], mmap_loc=mmap_loc, executor=executor, workers=workers, dtype=dtype)
                logger.info(("Requesting:", chunked_shuffled_blocks[start_block]))
                logger.info(("Start block {0}".format(start_block)))
                for i,blocks in enumerate(chunked_shuffled_blocks):
                            # skip first start_block blocks
                            if (i < start_block): 
                                logger.info(("Skipping block {0}".format(i)))
                                continue
                            start_block = 0

                            # start timing
                            iter_start = time.time()

                            # switch between 2 buffers
                            mmap_loc = mmap_locs[i%2 == 0]

                            # convert block to bidxes
                            bidxes = block_idxs_to_idxs(blocks, K.shard_sizes[0], num_samples)

                            # load the block from memory
                            block_get_start = time.time()
                            fs.wait(K_block_futures)
                            [f.result() for f in K_block_futures]
                            K_block = load_mmap(*K_block_futures[0].result())
                            # delete the futures so we don't have a dumb bug
                            del K_block_futures
                            block_get_end = time.time()
                            print("Block get took {0} seconds".format(block_get_end - block_get_start))


                            solve_time_start = time.time()
                            # if this isn't the last block of the epoch, pre-fetch the next set of block
                            if (i < len(chunked_shuffled_blocks) - 1):
                                next_blocks = chunked_shuffled_blocks[i+1]
                                logger.info(("Requesting:", next_blocks))
                                K_block_futures = fast_kernel_column_block_async(K, next_blocks, mmap_loc=mmap_loc, executor=executor, workers=workers)





                            # compute residuals
                            y_block = y[bidxes, :]
                            logger.info(("Computing residual quantity"))

                            Kbb = K_block[bidxes, :]
                            bb_idxes = np.diag_indices(Kbb.shape[0])
                            Kbb[bb_idxes] += lambdav
                            K_block[bidxes, :] = Kbb
                            # this could be made faster cause x starts out sparse
                            R = K_block.T.dot(x)

                            Kbb = K_block[bidxes, :]
                            print(Kbb)
                            try:
                                res = y_block - R
                                if (np.linalg.norm(res) > 1e3):
                                    print("Residual too big {0}".format(np.linalg.norm(res)))
                                    raise Exception("Residual too big")
#shiv says make R - y_block
                                x_block = scipy.linalg.solve(Kbb, res, sym_pos=True)
                                Kbb[bb_idxes] -= lambdav
                                K_block[bidxes, :] = Kbb
                                # update model
#shiv says make this -=
                                x[bidxes] += x_block
                                t = time.time()
                                y_hat += K_block.dot(x_block)
                                acc = metrics.accuracy_score(np.argmax(y_hat, axis=1), labels)
                                logger.info(("Eval interval", eval_interval))
                                logger.info(("Epoch {0} Block {1} Lambda {3} Training Accuracy {2}".format(epoch, i, acc, lambdav)))
                                e = time.time()
                                iters += 1
                            except LinAlgError as e:
                                logger.info(("Singular Matrix at block {0}".format(i)))
                                acc = "NA"
                            iter_end = time.time()
                            logger.info(("Block {0} of epoch {1} took {2}".format(i, epoch, iter_end - iter_start)))
                            if ((i) % eval_interval == 0):
                                logger.info(("Evaluating"))
                                if (eval_fn != None):
                                    eval_futures.append(ex.submit(eval_fn, x, y_hat, y,  lambdav, block=i, epoch=epoch, iter_time=iter_end-iter_start))
                                    try:
                                        eval_futures[-1].result()
                                    except:
                                        pass
                            solve_time_end = time.time()
                            print("Residual + Solve time {0}".format(solve_time_end - solve_time_start))

        # all done!

        eval_futures.append(ex.submit(eval_fn, x, y_hat, y, lambdav, block=i, epoch=epoch, iter_time=iter_end-iter_start))
        eval_futures[-1].result()
        model_norm = np.linalg.norm(x)
        train_acc = metrics.accuracy_score(np.argmax(y_hat, axis=1), labels)
        np.save("/tmp/{3}_epochs_{0}_train_acc_{1}_norm_{2}.model".format(epochs, train_acc, model_norm, K.key), x)
        np.save("/tmp/{3}_epochs_{0}_train_acc_{1}_norm_{2}.yhat".format(epochs, train_acc, model_norm, K.key), y_hat)
        logger.info(("Waiting for evaluations to finish..."))
        fs.wait(eval_futures)

        return x, y_hat



def block_kernel_solve_local(K, y, block_size=4000, epochs=1, lambdav=0.1, verbose=True, prc=lambda x: x, warm_start=None):
        '''Solve (K + \lambdaI)x = y
            in a block-wise fashion
        '''

        # compute some constants
        num_samples = K.shape[0]
        num_blocks = math.ceil(num_samples/block_size)
        if (warm_start == None):
            x = np.zeros((K.shape[0], y.shape[1]))
        else:
            x = warm_start
        loss = 0
        labels = np.argmax(y, axis=1)

        y_hat = np.zeros(y.shape)
        y = prc(y)

        i = 0
        for e in range(epochs):
                np.random.seed(0)
                #shuffled_coords = list(np.random.choice(num_samples, num_samples, replace=False))
                shuffled_coords = list(range(num_samples))
                for b in range(int(num_blocks)):
                        # pick a block
                        block = shuffled_coords[b*block_size:min((b+1)*block_size, num_samples)]

                        # pick a subset of the kernel matrix (note K can be mmap-ed)
                        K_block = prc(K[:, block])
                        y_block = y[block, :]

                        # This is a matrix vector multiply very efficient can be parallelized
                        # (even if K is mmaped)

                        logger.info(("Solving blocks {0} to {1}".format(block[0], block[-1])))
                        # calculate
                        R = K_block.T.dot(x)
                        Kbb = K_block[block, :]

                        # Add term to regularizer
                        idxes = np.diag_indices(Kbb.shape[0])

                        Kbb[idxes] += lambdav
                        logger.info(("solving block {0}".format(b)))

                        x_block = scipy.linalg.solve(Kbb, y_block - R, sym_pos=True)
                        Kbb[idxes] -= lambdav
                        # update model
                        x_block_old = x[block]
                        x[block] += x_block
                        y_hat += K_block.dot(x_block)
                        acc = metrics.accuracy_score(np.argmax(y_hat, axis=1), labels)
                        logger.info(("Iteration {0}, Training Accuracy {1}".format(i, acc)))
                        i += 1

        return x

def block_kernel_solve_acc_local(K, y, block_size=4000, epochs=1, lambdav=0.1, mu=0.001, verbose=True, prc=lambda x: x, eval_fn=None, warm_start=None):
        '''Solve (K + \lambdaI)x = y
            in a block-wise fashion
        '''

        # compute some constants
        num_samples = K.shape[0]
        nu = 1.1*num_samples/block_size
        tau = np.sqrt(mu/nu)
        num_blocks = math.ceil(num_samples/block_size)
        if (warm_start == None):
            x = np.zeros((K.shape[0], y.shape[1]))
        else:
            x = warm_start
        loss = 0
        labels = np.argmax(y, axis=1)

        y_hat = np.zeros(y.shape)
        y = prc(y)

        i = 0
        dtype="float64"
        a_bcd_acc, z_bcd_acc = np.zeros(x.shape, dtype=dtype), np.zeros(x.shape, dtype=dtype)

        for e in range(epochs):
                np.random.seed(0)
                shuffled_coords = list(range(num_samples))
                for b in range(int(num_blocks)):
                        # pick a block
                        block = shuffled_coords[b*block_size:min((b+1)*block_size, num_samples)]

                        # pick a subset of the kernel matrix (note K can be mmap-ed)
                        K_block = prc(K[:, block])
                        y_block = y[block, :]

                        # This is a matrix vector multiply very efficient can be parallelized
                        # (even if K is mmaped)

                        # calculate
                        R = np.zeros(y_block.shape)
                        for b2 in range(int(num_blocks)):
                            if b2 == b: continue
                            block_b2 = shuffled_coords[b2*block_size:min((b2+1)*block_size, num_samples)]
                            R += K_block[block_b2, :].T.dot(x[block_b2, :])

                        Kbb = K_block[block, :]

                        # Add term to regularizer
                        idxes = np.diag_indices(Kbb.shape[0])

                        Kbb[idxes] += lambdav
                        logger.info(("solving block {0}".format(b)))

                        x_block = scipy.linalg.solve(Kbb, y_block - R, sym_pos=True)

                        x_bcd_acc_kp1 = (1/(1+tau))*a_bcd_acc + (tau/(1+tau))*z_bcd_acc
                        a_bcd_acc_kp1 = np.array(x_bcd_acc_kp1)

                        residual =  K_block.T.dot(x_bcd_acc_kp1) - y_block
                        H_k_nabla_f_x_kp1 = scipy.linalg.solve(Kbb, residual)
                        a_bcd_acc_kp1[block] -= H_k_nabla_f_x_kp1
                        z_bcd_acc_kp1 = (1-tau)*z_bcd_acc + tau*x_bcd_acc_kp1
                        z_bcd_acc_kp1[block] -= (tau/mu)*H_k_nabla_f_x_kp1

                        x_bcd_acc = x_bcd_acc_kp1
                        a_bcd_acc = a_bcd_acc_kp1
                        z_bcd_acc = z_bcd_acc_kp1

                        Kbb[idxes] -= lambdav
                        # update model
                        y_hat = K.dot(x_bcd_acc)
                        acc = metrics.accuracy_score(np.argmax(y_hat, axis=1), labels)
                        logger.info(("Iteration {0}, Training Accuracy {1}".format(i, acc)))
                        i += 1
        return x



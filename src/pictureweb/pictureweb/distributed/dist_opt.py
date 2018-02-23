from numpy as np
from sharded_matrix import ShardedMatrix
from distributed import fast_kernel_column_block_async

def pcg_pywren(A,b,pwex, prc=lambda x: x, max_iter=100, tol=1e-3, col_chunk_size=25, row_chunk_size=2500):
    i = 0
    # starting residual is b
    r = b
    d = prc(r)
    delta_new = np.linalg.norm(r.T.dot(d))
    delta_0 = delta_new
    print("Delta 0 is {0}".format(delta_0))
    x = np.zeros((A.shape[0], b.shape[1]), 'float32')
    print(x.shape)
    while (True):
        if (i >= max_iter):
            break

        if (delta_new < tol*delta_0):
            break
        # Expensive
        print("Matrix multiply")
        d_sharded = ShardedMatrix(d, shard_size_0=A.shard_size_0, reshard=True, bucket="imagenet-raw")
        q = pywren_matrix_vector_multiply(pwex, A, d_sharded, col_chunk_size=col_chunk_size, row_chunk_size=row_chunk_size)
        a = delta_new/np.linalg.norm(d.T.dot(q))
        print(a)
        x = x + a*d
        r = r - a*q
        print("Iter {0}, NORM IS {1}".format(i,np.linalg.norm(r)))
        s = prc(r)
        delta_old = delta_new
        delta_new = np.linalg.norm(r.T.dot(s))
        beta = delta_new/delta_old
        d = s + beta * d
        i = i + 1
    return x

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

def block_kernel_solve(K, y, epochs=1, max_iter=313, block_size=4096, num_blocks=313, lambdav=0.1, verbose=True, prc=lambda x: x, workers=22):
        '''Solve (K + \lambdaI)x = y
            in a block-wise fashion
        '''
        labels = np.argmax(y, axis=1)
        with fs.ProcessPoolExecutor(max_workers=workers) as executor:
            mmap_locs = ["/dev/shm/block0", "/dev/shm/block1"]
            mmap_loc = mmap_locs[0]
            # compute some constants
            x = np.zeros(y.shape, 'float32')
            i = 0


            K_block_future = fast_kernel_column_block_async(K_sharded, 0, mmap_loc=mmap_loc, executor=executor, workers=workers)

            for e in range(epochs):
                    y_hat = np.zeros(y.shape)
                    for b in range(int(num_blocks)):
                            iter_start = time.time()
                            if (i > max_iter):
                                return x

                            mmap_loc = mmap_locs[i%2 == 0]
                            print("Grabbing this block from oven")
                            s = time.time()
                            K_block = load_mmap(*K_block_future.result())
                            e = time.time()
                            print("Block spent {0} seconds in oven".format(e - s))
                            print("Putting next Block in oven")
                            s = time.time()
                            K_block_future = fast_kernel_column_block_async(K_sharded, (b+1)%num_blocks, mmap_loc=mmap_loc, executor=executor, workers=workers)
                            # pick a subset of the kernel matrix (note K can be mmap-ed)
                            e = time.time()
                            print("Took {0} seconds to put block in oven".format(e - s))
                            b_start =  b*block_size
                            b_end = min((b+1)*block_size, K.shape[1])
                            y_block = y[b_start:b_end, :]
                            start = time.time()
                            R = np.zeros((b_end - b_start, y.shape[1]), 'float32')
                            __calculate_res(R, num_blocks, b, K_block, x, block_size, K.shape[0])
                            end = time.time()
                            print("Residual time {0}".format(end - start))
                            start = time.time()
                            Kbb = K_block[b_start:b_end, :].astype('float64')
                            print(Kbb.shape)
                            # Add term to regularizer
                            idxes = np.diag_indices(Kbb.shape[0])
                            try:
                                Kbb[idxes] += lambdav
                                x_block = scipy.linalg.solve(Kbb, y_block - R, sym_pos=True)
                                Kbb[idxes] -= lambdav
                                # update model
                                x_block = x_block.astype('float32')
                                x[b_start:b_end] = x_block
                                t = time.time()
                                print("Residual is {0}".format(np.linalg.norm(y_block - R)))
                                y_hat += K_block.dot(x_block)
                                acc = metrics.accuracy_score(np.argmax(y_hat, axis=1), labels)
                                print("Iteration {0}, Training Accuracy {1}".format(i, acc))
                                e = time.time()
                                print("Calculating accuracy took {0} secs ".format(e -s))
                            except LinAlgError as e:
                                print("Singular matrix in block {0} with reg {1}".format(b, lambdav))
                            iter_end = time.time()
                            print("Iteration {0} took {1} seconds".format(i, iter_end - iter_start))
                            i += 1
            return x





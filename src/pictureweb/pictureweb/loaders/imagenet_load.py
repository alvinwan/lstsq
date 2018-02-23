import tarfile
import logging
import multiprocessing as mp
import concurrent.futures as fs
from imread import imread_from_blob
import numpy as np
import os

logger = logging.getLogger(__name__)

def class_to_tarball_name_train(classname):
    classname = classname.strip()
    return "/data/vaishaal/imagenet-misc/data/imagenet-train/{0}-scaled.tar".format(classname)

def class_to_tarball_name_val(classname):
    classname = classname.strip()
    return "/data/vaishaal/imagenet-misc/data/imagenet-validation-correct/val-{0}-scaled.tar".format(classname)

TRAIN_LOADER = class_to_tarball_name_train
VAL_LOADER = class_to_tarball_name_val

def is_image(tarinfo):
    return tarinfo.name.lower().endswith('jpg') or \
           tarinfo.name.lower().endswith('jpeg') or \
           tarinfo.name.lower().endswith('png')

def orient(im):
    return im.transpose(1,2,0)


def load_tar_into_idx(tar_path, sidx, eidx, mmap_loc, mmap_shape, preprocess):
    ''' Load tarball into self.mmap_loc[sidx:eidx,:,:,:] '''
    tarball = tarfile.TarFile(tar_path)
    members = tarball.getmembers()
    images = []
    for mem in members:
        if (not is_image(mem)): continue
        f = tarball.extractfile(mem)
        im_bytes = f.read()
        im = imread_from_blob(im_bytes, 'jpg')
        if (im.shape[2] != 3):
            # grey scale image
            im = np.concatenate((im,im,im), axis=2)
        if (preprocess != None):
            im = preprocess(im)
        im = im.transpose(2,0,1)
        im = im[np.newaxis, :, :, :]
        images.append(im)
    X_class = np.concatenate(images, axis=0)
    X = np.memmap(mmap_loc, dtype="uint8", mode="r+", shape=mmap_shape)
    X[sidx:eidx, :, :, :] = X_class
    tarball.close()
    X.flush()
    return 0


def count_images_in_tarball(loc):
    tarball = tarfile.TarFile(loc)
    count = 0
    for member in tarball.getmembers():
        count += is_image(member)
    tarball.close()
    return count


class ImagenetLoader(object):
    def __init__(self, num_classes, classes_path="./classes", tarball_func=class_to_tarball_name_train, mmap_loc="/tmp/imagenet", image_shape=(3,256,256), preprocess=None, n_procs=10):
        self.shape = image_shape
        self.classes = open(classes_path).readlines()[:num_classes]
        self.locs = list(map(tarball_func, self.classes))
        self.count = 0
        self.class_counts = [0]
        self.n_procs = n_procs
        self.preprocess = preprocess

        with mp.Pool(n_procs) as pool:
            count_asyncs = []
            for loc in self.locs:
                count_async = pool.apply_async(count_images_in_tarball, (loc,))
                count_asyncs.append(count_async)

            class_counts = (list(map(lambda x: x.get(), count_asyncs)))
            self.count = sum(class_counts)
            self.class_counts.extend(list(np.cumsum(class_counts)))

        self.class_idxs = list(zip(self.class_counts[:-1], self.class_counts[1:]))
        logger.info("There are {0} images to be extracted".format(self.count))
        self.mmap_loc = mmap_loc
        if (os.path.isfile(self.mmap_loc)):
            logger.warning("There exists a matrix at {0}".format(mmap_loc))


        self.X = np.memmap(mmap_loc, dtype="uint8", mode="w+", shape=(self.count,) + self.shape)
        self.Y = np.zeros(self.count)

        for i, (sidx, eidx) in enumerate(self.class_idxs):
            self.Y[sidx:eidx] = i

    def load_all(self):
        ''' Load imagenet into self.mmap_loc '''
        with mp.Pool(self.n_procs) as pool:
            futures = []
            i = 0
            for i,(sidx, eidx) in enumerate(self.class_idxs):
                future = pool.apply_async(load_tar_into_idx, (self.locs[i], sidx, eidx, self.mmap_loc, self.X.shape,self.preprocess))
                if (i % 100 == 0):
                    list(map(lambda x: x.get(), futures))
                    print("Loaded {0} classes into mem".format(i))
                futures.append(future)
            list(map(lambda x: x.get(), futures))
        return self.X




























import hashlib
import numpy as np

def hash_numpy_array(x):
    return hashlib.sha1(x.view(np.uint8)).hexdigest()

def hash_string(s):
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def make_rbf_hash(s, gamma):
    return hashlib.sha1("rbf_kernel({0}, {1})".format(s, gamma).encode('utf-8')).hexdigest()

def make_linear_hash(s):
    return hashlib.sha1("linear_kernel({0})".format(s).encode('utf-8')).hexdigest()



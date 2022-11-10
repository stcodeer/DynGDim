# import torch 
from tokenize import Double
import numpy as np
from scipy.sparse.linalg import eigs, expm_multiply, expm
from scipy.sparse import csr_matrix, csc_matrix
import cupy as cp
import cupyx as cpx
import random
import string
import time

s = []
for line in open("/home/tongsu/DynGDim/pyGOD_builtin_datasets/data"):
    s += line.split()
print(len(s))
data = s[0:168016]
indices = s[168016:336032]
indptr = s[336032:]
data = np.array([float(x) for x in data])
indices = np.array([int(x) for x in indices])
indptr = np.array([int(x) for x in indptr])

print(data)
print(indices)
print(indptr)

# scipy

a = csc_matrix((data, indices, indptr), shape=(10984, 10984))

b = np.random.rand(10984)

before = time.time()

print("begin scipy")

c = expm_multiply(a, b)

after = time.time()

print(c, after - before)

# cupy

data = cp.asarray(data)
indices = cp.asarray(indices)
indptr = cp.asarray(indptr)

a = cpx.scipy.sparse.csc_matrix((data, indices, indptr), shape=(10984, 10984))

b = cp.asarray(b)

before = time.time()

print("begin cupy")

c = (cpx.scipy.sparse.csc_matrix.expm1(a) + 1)

after = time.time()

print(c, after - before)


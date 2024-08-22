import numpy as np
from CloudKeL import encryption
def test_get_sparse_1():
    X = np.array([
        [0, 1, 0, 2],
        [4, 0, 3, 0],
        [4, 0, 0, 1],
        [0, 5, 7, 0]
    ], dtype=np.int64)

    target = [(0, 2), (3, 0), (2, 1), (1, 3)]
    sparse = encryption.get_sparse(X)
    assert target == sparse

def test_get_sparse_2():
    X = np.array([
        [0, 1, 0, 2, 0],
        [1, 0, 0, 1, 3],
        [0, 0, 0, 4, 0],
        [2, 1, 4, 0, 1],
        [0, 3, 0, 1, 0]
    ], dtype=np.int64)

    target = [(0, 2), (2, 0), (0, 4), (4, 0), (1, 2), (2, 1), (2, 4), (4, 2)]
    sparse = encryption.get_sparse(X)
    assert target == sparse

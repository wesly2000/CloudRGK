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


def test_matrix_confusion():
    def ind(i, j, n, shift):
        assert shift >= 0, "Only support non-negative shift."
        return i*n+j+shift

    X = np.array([
        [0, 1, 0, 2, 0],
        [1, 0, 0, 1, 3],
        [0, 0, 0, 4, 0],
        [2, 1, 4, 0, 1],
        [0, 3, 0, 1, 0]
    ], dtype=np.int64)

    confused_X = encryption.matrix_confusion(X, ind, shift=2)
    target = np.array([
        [0, 1, 4, 2, 6],
        [1, 0, 9, 1, 3],
        [12, 13, 0, 4, 16],
        [2, 1, 4, 0, 1],
        [22, 3, 24, 1, 0]
    ], dtype=np.int64)

    assert np.all(target == confused_X)
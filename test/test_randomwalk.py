import numpy as np
from CloudKeL import randomwalk


def test_delta_kernel_kronecker_product():
    X_1 = np.array([
        [0, 1, 2],
        [4, 0, 0],
        [3, 0, 0]
    ], dtype=np.int64)

    X_2 = np.array([
        [0, 1, 0, 4],
        [3, 0, 0, 0],
        [0, 2, 0, 5],
        [1, 1, 0, 0]
    ], dtype=np.int64)

    target = np.array([
        [1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    ], dtype=np.int64)

    X = randomwalk.delta_kernel_kronecker_product(X_1, X_2)
    assert np.all(X == target)

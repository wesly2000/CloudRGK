import numpy as np
import pytest

from CloudKeL import encryption, randomwalk

@pytest.fixture
def simple_matrices():
    X_1 = np.array([
        [0, 1, 0, 2],
        [4, 0, 3, 0],
        [4, 0, 0, 1],
        [0, 5, 7, 0]
    ], dtype=np.int64)

    X_2 = np.array([
        [0, 1, 0, 2, 0],
        [1, 0, 0, 1, 3],
        [0, 0, 0, 4, 0],
        [2, 1, 4, 0, 1],
        [0, 3, 0, 1, 0]
    ], dtype=np.int64)

    return X_1, X_2


def test_get_sparse_1(simple_matrices):
    X, _ = simple_matrices

    target = [(0, 2), (3, 0), (2, 1), (1, 3)]
    sparse = encryption.get_sparse(X)
    assert target == sparse


def test_get_edge_1(simple_matrices):
    X, _ = simple_matrices

    target = [(0, 1), (1, 0), (2, 0), (0, 3), (1, 2), (3, 1), (2, 3), (3, 2)]
    sparse = encryption.get_edge(X)
    assert target == sparse


def test_get_sparse_2(simple_matrices):
    _, X = simple_matrices

    target = [(0, 2), (2, 0), (0, 4), (4, 0), (1, 2), (2, 1), (2, 4), (4, 2)]
    sparse = encryption.get_sparse(X)
    assert target == sparse


def test_matrix_confusion(simple_matrices):
    def ind(i, j, n, shift):
        assert shift >= 0, "Only support non-negative shift."
        return i * n + j + shift

    _, X = simple_matrices

    confused_X = encryption.matrix_confusion(X, ind, shift=2)
    target = np.array([
        [0, 1, 4, 2, 6],
        [1, 0, 9, 1, 3],
        [12, 13, 0, 4, 16],
        [2, 1, 4, 0, 1],
        [22, 3, 24, 1, 0]
    ], dtype=np.int64)

    assert np.all(target == confused_X)

def test_xor_encryption(simple_matrices):
    _, X = simple_matrices

    X_xor_xor = encryption.xor_encryption(encryption.xor_encryption(X, 114514), 114514)

    assert np.all(X == X_xor_xor)

def test_matrix_correction():
    # This test covers a comprehensive case.
    # First, we encrypt the input matrice X_1, X_2 by matrix_confusion and xor;
    # Do delta Kronecker product;
    # Correct the result and validate.

    X_1 = np.array([
        [0, 1, 2],
        [4, 0, 0],
        [3, 0, 0]
    ], dtype=np.int64)
    sparse_1, edge_1 = encryption.get_sparse(X_1), encryption.get_edge(X_1)
    X_1_sparse_target = [(1, 2), (2, 1)]

    X_2 = np.array([
        [0, 1, 0, 4],
        [3, 0, 0, 0],
        [0, 2, 0, 5],
        [1, 1, 0, 0]
    ], dtype=np.int64)
    sparse_2, edge_2 = encryption.get_sparse(X_2), encryption.get_edge(X_2)
    X_2_sparse_target = [(0, 2), (2, 0), (1, 2), (1, 3), (3, 2)]
    assert (X_1_sparse_target == sparse_1) and (X_2_sparse_target == sparse_2)

    X = randomwalk.delta_kernel_kronecker_product(X_1, X_2)

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

    def ind(i, j, n, shift):
        assert shift >= 0, "Only support non-negative shift."
        return i * n + j + shift

    X_1_confused = encryption.matrix_confusion(X_1, ind, shift=2)
    X_1_encrypted = encryption.xor_encryption(X_1_confused, key=114514)
    X_2_confused = encryption.matrix_confusion(X_2, ind, shift=2)
    X_2_encrypted = encryption.xor_encryption(X_2_confused, key=114514)
    n_1, n_2 = X_1.shape[0], X_2.shape[0]
    X_encrypted = randomwalk.delta_kernel_kronecker_product(X_1_encrypted, X_2_encrypted)

    X_corrected = encryption.matrix_correction(X_encrypted, sparse_1, sparse_2, edge_1, edge_2, n_1, n_2)

    assert np.all(target == X_corrected)
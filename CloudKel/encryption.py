import numpy as np
from numpy.random import randint, rand, choice
from itertools import combinations
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from scipy.sparse import csc_matrix

from .constant import __ELEMENT_TYPE__, __INDEX_TYPE__, __RECOVER_STEP__

def matrix_confusion(X:np.ndarray, shift=1):
    '''
    Confuse the feature matrix to avoid 0-recognizing attack, 
    the confusing process is performed by shifting the amount @shift.
    '''
    dtype_check(X)
    N = X.shape[0]
    I = np.eye(N, dtype=X.dtype)
    return X + shift - shift*I

def get_sparse(X:np.ndarray):
    '''
    Get all the 0-entries' indices.
    '''
    dtype_check(X)
    n = X.shape[0]
    sparse = [(i, j) for (i, j) in combinations(range(n), 2) if X[i,j]==0]
    return sparse

def matrix_correction(W:np.ndarray, sparse_1, sparse_2, n_1, n_2):
    '''
    Correct the wrong entries due to the matrix confusion.
    '''
    for i in range(n_1):
        for (k, l) in sparse_2:
            W[i*n_2+k, i*n_2+l] = W[i*n_2+l, i*n_2+k] = 1

    for k in range(n_2):
        for (i, j) in sparse_1:
            W[i*n_2+k, j*n_2+k] = W[j*n_2+k, i*n_2+k] = 1

    return W

def key_generator():
    return randint(low=np.iinfo(__ELEMENT_TYPE__).min, high=np.iinfo(__ELEMENT_TYPE__).max, dtype=np.int64) # 64-bit key

def xor_encryption(X:np.ndarray, key:__ELEMENT_TYPE__):
    '''
    Encrypt X in element-wise manner, using XOR method. To avoid *known-plaintext
    attack, the diagonal elements will be set to 0, which means that we do not
    encrypt the diagonal.
    '''
    dtype_check(X)
    return X ^ (key*(np.ones_like(X, dtype=X.dtype)-np.eye(X.shape[0], dtype=X.dtype)))

def vertex_encryption(X:np.ndarray, key:__ELEMENT_TYPE__, shuffle=False):
    '''
    Encrypt graph vertex list in element-wise manner, using XOR method.
    '''
    dtype_check(X)
    if shuffle:
        np.random.shuffle(X)
    return X ^ (key*(np.ones_like(X, dtype=X.dtype)))

def tuple_list_encoder(index_list:list):
    '''
    Compress list of form [(i, j),...] to [i, j, ...], and convert it to
    byte array.
    '''
    compressed_list = []
    for (i, j) in index_list:
        compressed_list += [i, j]
    compressed_array = np.array(compressed_list, dtype=__INDEX_TYPE__)
    byte_stream = bytes(compressed_array)
    return byte_stream

def tuple_list_decoder(byte_stream:bytes, byte_order='little'):
    '''
    Convert byte array to list like [i, j, ...] and further convert it to
    [(i,j), ...]
    '''
    recover_list = []
    for i in range(0, len(byte_stream), 2*__RECOVER_STEP__):
        recover_list.append((int.from_bytes(byte_stream[i:i+__RECOVER_STEP__], byteorder=byte_order),
                             int.from_bytes(byte_stream[i+__RECOVER_STEP__:i+2*__RECOVER_STEP__], byteorder=byte_order)))
    return recover_list

def byte_stream_encryption(byte_stream:bytes, key:bytes):
    '''
    Encrypt byte array with @key using AES encryption algorithm, @key should
    be bytes type. 

    Returns:
        ciphertext: bytes,
        The cipher obtained from the plaintext.

        tag: bytes,
        The digest obtained from encryption, used for integrity.

        nonce: bytes,
        Used for decryption.
    '''
    if not isinstance(key, bytes):
        key = bytes(key)

    cipher = AES.new(key, AES.MODE_EAX)
    ciphertext, tag = cipher.encrypt_and_digest(bytes(byte_stream))
    return ciphertext, tag, cipher.nonce

def byte_stream_decryption(key, ciphertext, tag, nonce):
    cipher = AES.new(key, AES.MODE_EAX, nonce)
    plaintext = cipher.decrypt_and_verify(ciphertext, tag)
    return plaintext

def dtype_check(X:np.ndarray):
    '''
    Check if a matrix is of dtype np.int64
    '''
    assert X.dtype == __ELEMENT_TYPE__, 'Only accept arrays with dtype==np.int64'
    return None

def matrix_key_generation():
    return 3.97+0.03*rand()

def matrix_encryption(M:np.ndarray, key_0, key_1):
    '''
    Use chaotic system for matrix encryption.
    '''
    assert 3.97 < key_0 <= 4 and 3.97 < key_1 <= 4 ,'Only keys in range (3.97, 4] are acceptable.'
    U_0, U_1 = rand(), rand()
    N = M.shape[0]
    alpha, beta = np.zeros(N), np.zeros(N)
    shuffle_1, shuffile_2 = choice(a=N, size=N, replace=False), choice(a=N, size=N, replace=False)
    for _, (a_i, b_i) in enumerate(zip(shuffle_1, shuffile_2)):
        U_0 *= key_0*(1-U_0)
        alpha[a_i] = U_0
        U_1 *= key_1*(1-U_1)
        beta[b_i] = U_1
    A, B = csc_matrix(np.diag(alpha)), csc_matrix(np.diag(beta))
    return A@M@B, A, B

def matrix_decryption(M, A, B):
    return A@M@B
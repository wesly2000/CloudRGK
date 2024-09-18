'''
Random Walk Graph Kernel module.
'''

import numpy as np
from numpy.linalg import inv
from itertools import combinations, combinations_with_replacement, product
from scipy.sparse import csc_matrix
# from scipy.sparse.linalg import inv
from numpy.linalg import inv as dense_inv

from CloudKeL.preprocessing import Graph

from .constant import __ELEMENT_TYPE__
from .preprocessing import Graph
from .encryption import *

class CloudGraph(Graph):
    def __init__(self, ID, label, feature_matrix, sparse_info, edge_info, vertex_info) -> None:
        '''
        The graph object uploaded to the Cloud.
        
        Attributes:
            ID: int,
                The identifier of the graph.
            label: int,
                The class label of the graph.
            feature_matrix: np.ndarray,
                The encrypted feature matrix of the graph.
            N: int,
                The number of vertices.
            sparse_info: tuple,
                Indices of feature matrix 0-entry, which is encrypted. To recover
                it, one need key(possessed by the Owner), and (sparse, tag, nonce)
                which is sparse_info.
            edge_info: tuple,
                Indices of feature matrix non-0-entry, which is encrypted. This
                is used for recovering transition matrix. To recover it, one need
                key(possessed by the Owner), and (edge_index, tag, nonce) which is
                edge_info.
            vertex_info: np.ndarray(or any other iterable),
                Encrypted vertex label list.
        '''
        self.ID, self.label = ID, label
        self.feature_matrix = feature_matrix
        self.sparse_info = sparse_info
        self.edge_info = edge_info
        self.vertex_info = vertex_info
        self.N = feature_matrix.shape[0]

    def build_transition_matrix(self, key:bytes):
        recover_byte_stream = byte_stream_decryption(
                                key=key,
                                ciphertext=self.edge_info[0],
                                tag=self.edge_info[1],
                                nonce=self.edge_info[2]
            )
        edge_index = tuple_list_decoder(recover_byte_stream)
        # Build transition matrix
        # The previous build_transition_matrix should be modified such that
        # no prod_degree is returned.
        P = np.zeros(shape=(self.N, self.N))
        for (i, j) in edge_index:
            P[i,j] = 1
        P += P.T 
        degree_list = P.sum(axis=0)
        return P/degree_list
    
    def build_feature_matrix(self):
        raise NotImplementedError
    
    @property
    def vertex_num(self):
        return self.N 
    
    @property
    def edge_num(self):
        raise NotImplementedError
        

class RandomWalkKernel():
    '''
    The (vanilla)Random Walk Graph Kernel, which is mainly performed by Data Owner.
    '''
    def __init__(self) -> None:
        pass

    def vertex_similarity_measure(self, vertex_list_1, vertex_list_2):
        similarity = 0
        for v_1 in vertex_list_1:
            for v_2 in vertex_list_2:
                similarity += delta_kernel(v_1, v_2)

        return similarity

    def pair_similarity_measure(self, G_1:Graph, G_2:Graph):
        '''
        Compute similarity of G_1 and G_2 using random walk kernel
        '''
        # normalizer = (G_1.vertex_num*G_2.vertex_num)**2
        P_1, P_2 = G_1.build_transition_matrix(), G_2.build_transition_matrix()
        X_1, X_2 = G_1.build_feature_matrix(), G_2.build_feature_matrix()
        T = np.kron(P_1, P_2) * delta_kernel_kronecker_product(X_1, X_2)
        I = np.eye(T.shape[0])
        # sparse = csc_matrix(I-T)
        # similarity = inv(sparse).sum() + self.vertex_similarity_measure(G_1.vertex_list, G_2.vertex_list)
        # similarity = inv(sparse).sum()
        similarity = inv(I-T).sum()
        return similarity
    
    def perform(self, graphs):
        '''
        Wrapper function of @pair_similarity_measure, which measures 
        pair-wise graph similarity of a group of graphs.
        '''
        n = len(graphs)
        K = np.zeros((n, n))
        for (i,j) in combinations_with_replacement(range(len(graphs)), 2):
            K[i, j] = K[j, i] = self.pair_similarity_measure(graphs[i], graphs[j])

        return K
    
    def transform(self, train_graphs, test_graphs):
        ''''''
        m, n = len(test_graphs), len(train_graphs)
        K = np.zeros((m, n))
        for i, test_graph in enumerate(test_graphs):
            for j, train_graph in enumerate(train_graphs):
                K[i, j] = self.pair_similarity_measure(test_graph, train_graph)

        return K

class CloudRandomWalkKernel(RandomWalkKernel):
    def __init__(self) -> None:
        super().__init__()

    def vertex_similarity_measure(self, vertex_list_1, vertex_list_2):
        return super().vertex_similarity_measure(vertex_list_1, vertex_list_2)

    def similarity_matrix(self, G_1: CloudGraph, G_2: CloudGraph, sparse_key:bytes, edge_key:bytes):
        # Delta Kronecker Product is performed by the Cloud
        Confused_W = delta_kernel_kronecker_product(G_1.feature_matrix, G_2.feature_matrix)
        # Transition Matrix Kronecker Product is performed by the Owner
        P = np.kron(G_1.build_transition_matrix(key=edge_key), G_2.build_transition_matrix(key=edge_key))
        # W correction, by the Owner.
        sparse_1 = tuple_list_decoder(byte_stream_decryption(
                            key=sparse_key, 
                            ciphertext=G_1.sparse_info[0], 
                            tag=G_1.sparse_info[1],
                            nonce=G_1.sparse_info[2]
            ))
        sparse_2 = tuple_list_decoder(byte_stream_decryption(
                            key=sparse_key, 
                            ciphertext=G_2.sparse_info[0], 
                            tag=G_2.sparse_info[1],
                            nonce=G_2.sparse_info[2]
            ))
        W = matrix_correction(
                        W=Confused_W, 
                        sparse_1=sparse_1, 
                        sparse_2=sparse_2, 
                        n_1=G_1.N, 
                        n_2=G_2.N
            )

        return P * W
    def pair_similarity_measure(self, 
                                G_1: CloudGraph, 
                                G_2: CloudGraph, 
                                sparse_key:bytes, 
                                edge_key:bytes, 
                                matrix_key_0, 
                                matrix_key_1):
        T = self.similarity_matrix(G_1, G_2, sparse_key, edge_key)
        I = np.eye(T.shape[0])
        sparse = csc_matrix(I-T)
        sparse_enc, A, B = matrix_encryption(sparse, matrix_key_0, matrix_key_1)
        sparse_enc_inv = inv(sparse_enc)
        similarity = matrix_decryption(sparse_enc_inv, B, A).sum() + self.vertex_similarity_measure(G_1.vertex_info, G_2.vertex_info)
        return similarity
    
    def perform(self, graphs, sparse_key:bytes, edge_key:bytes, matrix_key_0, matrix_key_1):
        '''
        Wrapper function of @pair_similarity_measure, which measures 
        pair-wise graph similarity of a group of graphs.
        '''
        n = len(graphs)
        K = np.zeros((n, n))
        for (i,j) in combinations_with_replacement(range(len(graphs)), 2):
            K[i, j] = K[j, i] = self.pair_similarity_measure(
                                            G_1=graphs[i], 
                                            G_2=graphs[j], 
                                            sparse_key=sparse_key, 
                                            edge_key=edge_key,
                                            matrix_key_0=matrix_key_0,
                                            matrix_key_1=matrix_key_1
                                            )

        return K
    
    def transform(self, train_graphs, test_graphs, sparse_key:bytes, edge_key:bytes, matrix_key_0, matrix_key_1):
        ''''''
        m, n = len(test_graphs), len(train_graphs)
        K = np.zeros((m, n))
        for i, test_graph in enumerate(test_graphs):
            for j, train_graph in enumerate(train_graphs):
                K[i, j] = self.pair_similarity_measure(
                                                G_1=test_graph, 
                                                G_2=train_graph, 
                                                sparse_key=sparse_key, 
                                                edge_key=edge_key,
                                                matrix_key_0=matrix_key_0,
                                                matrix_key_1=matrix_key_1
                                                )

        return K

def gaussian_kernel(x_1, x_2, scale=.1):
    return 2*np.exp(-np.abs(x_1-x_2)**2/scale)

def gaussian_kernel_kronecker_product(X_1:np.ndarray, X_2:np.ndarray, sigma=.1):
    '''
    Kronecker product between Phi(X_1), Phi(X_2).
    '''
    n_1, n_2 = X_1.shape[0], X_2.shape[0]
    X_kron= np.zeros((n_1*n_2, n_1*n_2))
    for (i, j) in combinations_with_replacement(range(n_1),2):
        for (k, l) in combinations_with_replacement(range(n_2), 2):
            X_kron[i*n_2+k, j*n_2+l] = gaussian_kernel(X_1[i,j], X_2[k,l], sigma)
            X_kron[i*n_2+l, j*n_2+k] = gaussian_kernel(X_1[i,j], X_2[l,k], sigma)
            X_kron[j*n_2+k, i*n_2+l] = gaussian_kernel(X_1[j,i], X_2[k,l], sigma)
            X_kron[j*n_2+l, i*n_2+k] = gaussian_kernel(X_1[i,j], X_2[l,k], sigma)

    return X_kron

def delta_kernel(x_1, x_2):
    return int(x_1 == x_2)

def delta_kernel_kronecker_product(X_1:np.ndarray, X_2:np.ndarray):
    '''
    Kronecker product between Phi(X_1), Phi(X_2).
    '''
    # n_1, n_2 = X_1.shape[0], X_2.shape[0]
    # X_kron= np.zeros((n_1*n_2, n_1*n_2), dtype=int)
    # for (i, j) in combinations_with_replacement(range(n_1),2):
    #     x_i_j = X_1[i,j] # Avoid repetitive indexing.
    #     for (k, l) in combinations_with_replacement(range(n_2), 2):
    #         X_kron[i*n_2+k, j*n_2+l] = \
    #         X_kron[i*n_2+l, j*n_2+k] = \
    #         X_kron[j*n_2+k, i*n_2+l] = \
    #         X_kron[j*n_2+l, i*n_2+k] = \
    #         delta_kernel(x_i_j, X_2[k,l])
    #
    # return X_kron
    dtype_check(X_1)
    dtype_check(X_2)
    n_1, n_2 = X_1.shape[0], X_2.shape[0]
    X_kron = np.zeros((n_1 * n_2, n_1 * n_2), dtype=np.int64)
    for (i, j) in combinations_with_replacement(range(n_1), 2):
        x_i_j = X_1[i, j]
        x_j_i = X_1[j, i] # Avoid repetitive indexing.
        for (k, l) in combinations_with_replacement(range(n_2), 2):
            X_kron[i * n_2 + k, j * n_2 + l] = delta_kernel(x_i_j, X_2[k, l])
            X_kron[j * n_2 + k, i * n_2 + l] = delta_kernel(x_j_i, X_2[k, l])
            X_kron[i * n_2 + l, j * n_2 + k] = delta_kernel(x_i_j, X_2[l, k])
            X_kron[j * n_2 + l, i * n_2 + k] = delta_kernel(x_j_i, X_2[l, k])

    return X_kron

def graph_upload(G:Graph, ID:int, shift:int, matrix_key:np.int64, sparse_key:bytes, edge_key:bytes):
    feature_matrix = G.build_feature_matrix()
    sparse = get_sparse(feature_matrix)
    edge_index = [(i,j) for (i,j) in G.edge_list.keys()]
    # DES(AES)加密sparse序列(转换为字节流)

    encrypted_matrix = xor_encryption(
                            X=matrix_confusion(feature_matrix, shift),
                            key=matrix_key
    )

    encrypted_sparse, tag_sparse, nonce_sparse = byte_stream_encryption(
                            byte_stream=tuple_list_encoder(sparse),
                            key=sparse_key
    ) 
    encrypted_edge, tag_edge, nonce_edge = byte_stream_encryption(
                            byte_stream=tuple_list_encoder(edge_index),
                            key=edge_key
    )
    vertex_list = np.array(G.vertex_list, dtype=__ELEMENT_TYPE__)
    encrypted_vertex = vertex_encryption(vertex_list, key=matrix_key)
    graph = CloudGraph(
            ID=ID,
            label=G.label,
            feature_matrix=encrypted_matrix,
            sparse_info=(encrypted_sparse, tag_sparse, nonce_sparse),
            edge_info=(encrypted_edge, tag_edge, nonce_edge),
            vertex_info=vertex_list
    )
    return graph

# def feature_matrix_upload(X:np.ndarray, shift:np.int64, key:np.int64):
#     '''
#     The contributor encrypts the graph feature matrix and upload it
#     to the Cloud.
#     '''
#     shifted_matrix = matrix_confusion(X, shift)
#     encrypted_matrix = xor_encryption(shifted_matrix, key)
#     return encrypted_matrix

# def sparse_upload(sparse, key):
#     '''
#     Encrypt sparse(tuple list) of a graph, and upload it to the Cloud.
#     '''
#     encrypted_sparse, tag, nonce = byte_stream_encryption(
#                             byte_stream=tuple_list_encoder(sparse),
#                             key=key
#     )
#     return (encrypted_sparse, tag, nonce)
#
# def edge_upload(edge_index, key):
#     '''
#     Encrypt edge indices(tuple list) of a graph, and upload it to the Cloud.
#     '''
#     encrypted_edge, tag, nonce = byte_stream_encryption(
#                             byte_stream=tuple_list_encoder(edge_index),
#                             key=key
#     )
    
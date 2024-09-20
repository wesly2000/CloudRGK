import numpy as np
from CloudKeL import randomwalk
from CloudKeL import hash as k_hash
from CloudKeL.randomwalk import *
from CloudKeL.preprocessing import cox_data_processor


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


def test_pair_similarity_measure():
    graphs = cox_data_processor(
        adj_file="../SimpleData/Simple_A.txt",
        node_label_file="../SimpleData/Simple_node_labels.txt",
        edge_label_file="../SimpleData/Simple_edge_labels.txt",
        graph_label_file="../SimpleData/Simple_graph_labels.txt",
        graph_indicator_file="../SimpleData/Simple_graph_indicator.txt"
    )

    hashed_graphs = k_hash.cox_graph_hash(graphs)
    gk = RandomWalkKernel()
    local_sim = gk.pair_similarity_measure(hashed_graphs[0], hashed_graphs[1])

    def ind(i, j, n, shift):
        assert shift >= 0, "Only support non-negative shift."
        return i * n + j + shift

    shift = 1

    matrix_key = key_generator()
    sparse_key = edge_key = get_random_bytes(16)

    cloud_graphs = []
    for id, g in enumerate(hashed_graphs):
        cloud_graphs.append(randomwalk.graph_upload(
            G=g,
            ID=id,
            ind=ind,
            shift=shift,
            matrix_key=matrix_key,
            sparse_key=sparse_key,
            edge_key=edge_key
        ))

    key_0, key_1 = matrix_key_generation(), matrix_key_generation()
    crgk = CloudRandomWalkKernel()
    cloud_sim = crgk.pair_similarity_measure(
        cloud_graphs[0],
        cloud_graphs[1],
        sparse_key=sparse_key,
        edge_key=edge_key,
        matrix_key_0=key_0,
        matrix_key_1=key_1
    )

    assert local_sim == cloud_sim

from CloudKeL import preprocessing as pre
import numpy as np

def test_graph_str():
    graph = pre.Graph(
        label=1,
        vertex_list=[1, 0, 0, 1, 2, 2],
        edge_list={(1, 0): 0, (0, 1): 0, (0, 5): 0, (5, 0): 0, (1, 2): 1, (2, 1): 1, (1, 3): 0, (3, 1): 0, (2, 3): 2,
                   (3, 2): 2, (3, 5): 0, (5, 3): 0}
    )

    target = ("Graph data with {} vertices and {} edges:\n".format(6, 12) +
              "Label: {}\n".format(1) +
              "Vertex list: {},\n".format([1, 0, 0, 1, 2, 2]) +
              "Edge list: {}, \n".format(
                  {(1, 0): 0, (0, 1): 0, (0, 5): 0, (5, 0): 0, (1, 2): 1, (2, 1): 1, (1, 3): 0, (3, 1): 0, (2, 3): 2,
                   (3, 2): 2, (3, 5): 0, (5, 3): 0})
              )

    assert target == str(graph)


def test_cox_data_processor():
    graph_0 = pre.Graph(
        label=1,
        vertex_list=[1, 0, 0, 1, 2, 2],
        edge_list={(0, 1): 100, (1, 0): 100, (0, 5): 100, (5, 0): 100, (1, 2): 101, (2, 1): 101, (1, 3): 100, (3, 1): 100, (2, 3): 102,
                   (3, 2): 102, (3, 4): 102, (4, 3): 102, (3, 5): 100, (5, 3): 100}
    )
    graph_1 = pre.Graph(
        label=-1,
        vertex_list=[1, 0, 0, 2, 2, 1],
        edge_list={(0, 1): 102, (4, 0): 100, (0, 5): 100, (5, 0): 101, (1, 2): 100, (1, 3): 100, (2, 3): 101, (3, 4): 100}
    )

    target = [graph_0, graph_1]

    graphs = pre.cox_data_processor(
        adj_file="../SimpleData/Simple_A.txt",
        node_label_file="../SimpleData/Simple_node_labels.txt",
        edge_label_file="../SimpleData/Simple_edge_labels.txt",
        graph_label_file="../SimpleData/Simple_graph_labels.txt",
        graph_indicator_file="../SimpleData/Simple_graph_indicator.txt"
    )

    assert target == graphs


def test_build_feature_matrix_1():
    graph = pre.Graph(
        label=1,
        vertex_list=[1, 0, 0, 1, 2, 2],
        edge_list={(0, 1): 100, (1, 0): 100, (0, 5): 100, (5, 0): 100, (1, 2): 101, (2, 1): 101, (1, 3): 100, (3, 1): 100, (2, 3): 102,
                   (3, 2): 102, (3, 4): 102, (4, 3): 102, (3, 5): 100, (5, 3): 100}
    )

    target = np.array([
        [0, 100, 0, 0, 0, 100],
        [100, 0, 101, 100, 0, 0],
        [0, 101, 0, 102, 0, 0],
        [0, 100, 102, 0, 102, 100],
        [0, 0, 0, 102, 0, 0],
        [100, 0, 0, 100, 0, 0],
    ], dtype=np.int64)

    assert np.all(graph.build_feature_matrix() == target)

def test_build_feature_matrix_2():
    graph = pre.Graph(
        label=-1,
        vertex_list=[1, 0, 0, 2, 2, 1],
        edge_list={(0, 1): 102, (4, 0): 100, (0, 5): 100, (5, 0): 101, (1, 2): 100, (1, 3): 100, (2, 3): 101, (3, 4): 100}
    )

    target = np.array([
        [0, 102, 0, 0, 0, 100],
        [0, 0, 100, 100, 0, 0],
        [0, 0, 0, 101, 0, 0],
        [0, 0, 0, 0, 100, 0],
        [100, 0, 0, 0, 0, 0],
        [101, 0, 0, 0, 0, 0],
    ], dtype=np.int64)

    assert np.all(graph.build_feature_matrix() == target)
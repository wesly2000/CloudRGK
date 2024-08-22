from CloudKeL import preprocessing as pre

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
        edge_list={(0,1):0, (1,0):0, (0,5):0, (5,0):0, (1,2):1, (2,1):1, (1,3):0, (3,1):0, (2,3):2, (3,2):2, (3,4):2, (4,3):2, (3,5):0, (5,3):0}
    )
    graph_1 = pre.Graph(
        label=-1,
        vertex_list=[1, 0, 0, 2, 2, 1],
        edge_list={(0,1):2, (4,0):0, (0,5):0, (5,0):1, (1,2):0, (1,3):0, (2,3):1, (3,4):0}
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
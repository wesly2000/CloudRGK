from CloudKeL import hash as k_hash
from CloudKeL import preprocessing as pre

def test_cox_graph_hash():

    vertex_list_0 = [1, 0, 0, 1, 2, 2]
    edge_list_0 = {(0, 1): 100, (1, 0): 100, (0, 5): 100, (5, 0): 100, (1, 2): 101, (2, 1): 101, (1, 3): 100, (3, 1): 100,
                 (2, 3): 102, (3, 2): 102, (3, 4): 102, (4, 3): 102, (3, 5): 100, (5, 3): 100}

    hashed_edge_list_0 = {(i, j): hash((vertex_list_0[i], label, vertex_list_0[j])) for (i, j), label in edge_list_0.items()}

    hashed_graph_0 = pre.Graph(
        label=1,
        vertex_list=vertex_list_0,
        edge_list=hashed_edge_list_0
    )

    vertex_list_1 = [1, 0, 0, 2, 2, 1]
    edge_list_1 = {(0, 1): 102, (4, 0): 100, (0, 5): 100, (5, 0): 101, (1, 2): 100, (1, 3): 100, (2, 3): 101,
                 (3, 4): 100}

    hashed_edge_list_1 = {(i, j): hash((vertex_list_1[i], label, vertex_list_1[j])) for (i, j), label in edge_list_1.items()}

    hashed_graph_1 = pre.Graph(
        label=-1,
        vertex_list=vertex_list_1,
        edge_list=hashed_edge_list_1
    )

    target = [hashed_graph_0, hashed_graph_1]

    graphs = pre.cox_data_processor(
        adj_file="../SimpleData/Simple_A.txt",
        node_label_file="../SimpleData/Simple_node_labels.txt",
        edge_label_file="../SimpleData/Simple_edge_labels.txt",
        graph_label_file="../SimpleData/Simple_graph_labels.txt",
        graph_indicator_file="../SimpleData/Simple_graph_indicator.txt"
    )

    hashed_graphs = k_hash.cox_graph_hash(graphs)
    assert hashed_graphs == target
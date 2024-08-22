# This module provides some common interfaces to convert original 
# data format into preferring format.

import pandas as pd
import numpy as np
import warnings

from .constant import __ELEMENT_TYPE__

class EdgeData(object):
    def __init__(self, i, j, iLabel=None, jLabel=None, edgeLabel=None) -> None:
        self.i = i 
        self.j = j 
        self.iLabel = iLabel
        self.jLabel = jLabel
        self.edgeLabel = edgeLabel

class Graph(object):
    def __init__(self, label:int, vertex_list=[], edge_list={}, is_directed=False) -> None:
        '''
        Graph data, which is a collection of EdgeData.
        Attributes:
            edge_list: dict,
            The edge set of the graph, whose elements' type is EdgeData.

            N: int,
            The number of vertices in the graph.

            label: int,
            The class label of the graph.

            is_directed: boolean,
            The graph is digraph or not.
        '''
        self.label = label
        self.vertex_list = vertex_list
        self.edge_list = edge_list
        self.is_directed = is_directed

    @property
    def vertex_num(self):
        # The number of vertices in the graph.
        return len(self.vertex_list)
    
    @property
    def edge_num(self):
        # The number of edges in the graph.
        return len(self.edge_list)

    def __eq__(self, other):
        if not isinstance(other, Graph):
            return False
        return (self.label == other.label and
                self.vertex_list == other.vertex_list and
                self.edge_list == other.edge_list)

    def __str__(self):
        return (
                "Graph data with {} vertices and {} edges:\n".format(self.vertex_num, self.edge_num) +
                "Label: {}\n".format(self.label) +
                "Vertex list: {},\n".format(self.vertex_list) +
                "Edge list: {}, \n".format(self.edge_list)
                )

    def build_transition_matrix(self, integral=False):
        '''
        Compute the probability transition matrix of @g. When @integral is
        true, the matrix is represented by integers for encryption.

        Example:
            A = [[0,1,1,1]
                [1,0,1,0]
                [1,1,0,0]
                [1,0,0,0]]
            (Normal)P = [[0,1/3,1/3,1/3]
                        [1/2,0,1/2,0]
                        [1/2,1/2,0,0]
                        [1,0,0,0]]
            prod_d = 3*2*2*1 = 12
            (Integral)P = [[0,4,4,4]
                        [6,0,6,0]
                        [6,6,0,0]
                        [12,0,0,0]]
            When receiving integral P, the data owner recover normal P by
            first decrypting E(P), then dividing it by prod_d.
        '''
        P = np.zeros((self.vertex_num, self.vertex_num))
        for (i,j), _ in self.edge_list.items():
            P[i,j] = 1
        # If the graph is directed, the edge_list will cover all the edges.
        # When the graph is undirected, we consider only half of the edges (i->j, i<j),
        # and add P by its transpose for efficiency.
        # Since we do not allow self-loop, the diagonal elements will stay 0.
        # if not self.is_directed:
        #     P += P.T

        degree_list = P.sum(axis=0)
        if integral is False:
            return P/degree_list
        
        degree_prod = int(degree_list.prod())
        if degree_prod > 2**32-1:
            warnings.warn("Warning: degree_prod=={}, exceeding 2^32-1.".format(degree_prod))
        P *= (degree_prod//degree_list)
        return P
    
    def build_feature_matrix(self):
        n = self.vertex_num
        X = np.zeros((n, n), dtype=__ELEMENT_TYPE__)
        for (i, j), edgeLabel in self.edge_list.items():
            X[i,j] = edgeLabel
        # return X + X.T
        return X

def cox_data_processor(
        adj_file:str,
        node_label_file:str,
        edge_label_file:str,
        graph_label_file:str,
        graph_indicator_file:str
):
    '''
    A processor to process cox-dataset-like datasets. For example, we have two
    graph G_1, G_2 in class 0, 1, respectively. 
    G_1 = [(0,1,A,B,one), (0,2,A,B,one), (1,2,B,B,two)]
    G_2 = [(0,1,D,C,two)]
    All the corresponding files look like these:
    ---------------------
    adj_file
    ---------------------
    0,1
    1,0
    0,2
    2,0
    1,2
    2,1
    0,1
    1,0
    ---------------------
    ---------------------
    node_label_file.txt
    ---------------------
    A
    B
    B
    D
    C
    ---------------------
    ---------------------
    edge_label_file.txt
    ---------------------
    one
    one
    two
    two
    ---------------------
    ---------------------
    graph_label_file.txt
    ---------------------
    0
    1
    ---------------------
    '''
    adj_data = pd.read_csv(adj_file, names=['src', 'dst'])
    node_label_data = pd.read_csv(node_label_file, names=['node_label'])['node_label'].tolist()
    edge_label_data = pd.read_csv(edge_label_file, names=['edge_label'])['edge_label'].tolist()
    graph_label_data = pd.read_csv(graph_label_file, names=['graph_label'])['graph_label'].tolist()
    graph_indicator_data = pd.read_csv(graph_indicator_file, names=['graph_indicator'])['graph_indicator'].tolist()

    total_node_num = len(graph_indicator_data)
    total_edge_num = adj_data.shape[0]
    
    node_idx = 1
    edge_idx = 0
    graph_number = len(graph_label_data) # The number of graphs in the dataset.
    graphs = []
    for graph_idx in range(graph_number):

        graph = Graph(label=graph_label_data[graph_idx],
                          vertex_list=[],
                          edge_list={})

        # used to shift node number to origin node number
        # For example, in adj_data, there is an edge (97, 96) belonging to Graph 2,
        # and Graph 1 has 77 vertices. Then Vertex 97 is actually Vertex 20 in Graph 2.
        # Moreover, since vertices are numbered from 0 in practice, Vertex 20 is 
        # Vertex 19 in Graph 2. Therefore, we shift i,j by -node_idx.
        node_shift = node_idx 

        while graph_indicator_data[node_idx-1] == graph_idx+1:
            graph.vertex_list.append(node_label_data[node_idx-1])
            node_idx += 1
            if node_idx > total_node_num:
                break

        while adj_data.iloc[edge_idx]['src'] <= node_idx-1:
            i, j = adj_data.iloc[edge_idx]['src'], adj_data.iloc[edge_idx]['dst']
            if i == j:
                raise Exception('{}=={}, but self-loop is not allowed.'.format(
                    i,j
                ))
            # We need to shift i by -1 because i-th vertex in adj_data
            # is actually (i-1)-th vetex in node_label_data.
            edgeLabel = edge_label_data[edge_idx]
            graph.edge_list[(i-node_shift, j-node_shift)] = edgeLabel
            edge_idx += 1
            if edge_idx >= total_edge_num:
                break

        graphs.append(graph)

    return graphs

# In HashKeL, the acceptable data format is (i ,j , iLabel, jLabel, edgeLabel).
# The hash function H takes as input the data, and output (i, j, hashEdgeLabel)
# to construct the standard graph representation.
# NOTE: Currently, vertex/edge attribute vector will be ignored.

from .preprocessing import EdgeData, Graph
from .constant import __ELEMENT_TYPE__


class DataHash():
    def __init__(self, vertex_label=False, edge_label=True) -> None:
        self.vertex_label = vertex_label
        self.edge_label = edge_label

    def record_hash(self, data:EdgeData):
        '''
        Hash a single edge(record) of a graph into hashed edge.
        '''
        if type(data) is not EdgeData:
            raise TypeError("EdgeData data type is needed.")
        if self.vertex_label is True and self.edge_label is True:
            hashEdgeLabel = hash((data.iLabel, data.jLabel, data.edgeLabel))
            return (data.i, data.j, hashEdgeLabel)
        elif self.vertex_label is True and self.edge_label is False:
            hashEdgeLabel = hash((data.iLabel, data.jLabel))
            return (data.i, data.j, hashEdgeLabel)
        elif self.vertex_label is False and self.edge_label is True:
            return data
        else:
            raise Exception("Cannot handle graph without vertex and edge label.")
        
    def data_hash(self, data):
        '''
        Wrapper function of tuple_hash, handling a list of tuples.
        '''
        hashed_data = []
        for d in data:
            hashed_data.append(self.record_hash(d))
        return hashed_data

def cox_single_graph_hash(graph: Graph):
    '''
    Hash cox-data-like graph data into hashed graph.
    '''
    
    hash_edge = {}
    for (i,j), edgeLabel in graph.edge_list.items():
        hash_edge[(i,j)] = hash(
            (
                graph.vertex_list[i],
                graph.vertex_list[j],
                edgeLabel
            )
        )
    hash_graph = Graph(label=graph.label, vertex_list=graph.vertex_list, edge_list=hash_edge)
    return hash_graph

def cox_graph_hash(graphs):
    '''
    Wrapper function of cox_single_graph_hash, hashes a group of cox-data-like
    graphs into hashed graphs.
    '''
    hash_graphs = []
    for graph in graphs:
        hash_graphs.append(cox_single_graph_hash(graph=graph))
    return hash_graphs
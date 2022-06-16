import numpy as np
import networkx as nx


class DirectedGraph(object):
    def __init__(self, node_features=None, edge_features=None):
        self.graph = nx.DiGraph()
        if node_features is not None:
            self.update_nodes(node_features)
        if edge_features is not None:
            self.update_edges(edge_features)

    def update_nodes(self, node_features):
        self.graph.add_nodes_from(node_features.keys())
        for node in node_features:
            self.graph.nodes[node]['feature'] = node_features[node]

    def remove_nodes(self, nodes):
        self.graph.remove_nodes_from(nodes)

    def update_edges(self, edge_features):
        self.graph.add_edges_from(edge_features.keys())
        for edge in edge_features:
            assert len(edge) == 2
            self.graph[edge[0]][edge[1]]['feature'] = \
                edge_features[edge]

    def remove_edges(self, edges):
        self.graph.remove_edges_from(edges)

    def has_node(self, node):
        return self.graph.has_node(node)

    def has_edge(self, edge):
        assert len(edge) == 2
        return self.graph.has_edge(edge[0], edge[1])

    def nodes(self):
        return self.graph.nodes

    def edges(self):
        return self.graph.edges

    def number_of_nodes(self):
        return self.graph.number_of_nodes()

    def number_of_edges(self):
        return self.graph.number_of_edges()

    def get_node_features_tensor(self):
        node_features = []
        node_map = {}
        for (i, n) in enumerate(self.graph.nodes):
            feature = self.graph.nodes[n]['feature']
            if feature is not None:
                node_features.append(feature)
            node_map[i] = n

        return np.array(node_features), node_map

    def get_edge_features_tensor(self):
        edge_features = []
        edge_map = {}
        for (i, e) in enumerate(self.graph.edges):
            feature = self.graph[e[0]][e[1]]['feature']
            if feature is not None:
                edge_features.append(feature)
            edge_map[i] = e

        return np.array(edge_features), edge_map

    def convert_to_tensor(self):
        # node feature matrix, adjacency matrix, edge feature matrix,
        # node map (node index -> node object),
        # edge map (edge index -> edge object)
        node_features, node_map = self.get_node_features_tensor()
        edge_features, edge_map = self.get_edge_features_tensor()
        adj_mat = nx.adjacency_matrix(self.graph)
        return node_features, edge_features, adj_mat, node_map, edge_map

    def get_node_feature(self, node):
        return self.graph.nodes[node]['feature']

    def get_edge_feature(self, edge):
        return self.graph[edge[0]][edge[1]]['feature']

    def get_neighbors(self, node):
        list_neighbors = [n for n in self.graph.neighbors(node)]
        return list_neighbors

    def visualize(self):
        # TODO: use pydot
        pass



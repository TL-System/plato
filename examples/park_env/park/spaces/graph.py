import numpy as np
from park import core
from park.spaces.rng import np_random
from park.utils.directed_graph import DirectedGraph


class Graph(core.Space):
    """
    The element of this space is a DirectedGraph object.
    The node features and edge features need to be confined
    within their ranges.
    """
    def __init__(self, node_feature_space, edge_feature_space):
        self.node_feature_space = node_feature_space
        self.edge_feature_space = edge_feature_space
        core.Space.__init__(self, 'graph_node_edge', (), np.float32)

    def sample(self):
        # returns an empty graph
        return DirectedGraph()

    def contains(self, x):
        is_element = (type(x) == DirectedGraph)
        if is_element:
            # Note: this step can be slow
            node_features, _ = x.get_node_features_tensor()
            edge_features, _ = x.get_edge_features_tensor()

            is_element = self.node_feature_space.contains(node_features) and \
                         self.edge_feature_space.contains(edge_features)

        return is_element

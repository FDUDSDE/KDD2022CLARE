import networkx as nx
import numpy as np


def feature_augmentation(nodes, edges, num_node, normalize=True):
    r"""Node feature augmentation `[deg(u), min(deg(N)), max(deg(N)), mean(deg(N)), std(deg(N))]`"""
    g = nx.Graph(edges)
    g.add_nodes_from(nodes)

    node_degree = [g.degree[node] for node in range(num_node)]

    feat_matrix = np.zeros([num_node, 5], dtype=np.float32)
    # feature 1 - node degree
    feat_matrix[:, 0] = np.array(node_degree).squeeze()

    for node in range(num_node):
        if len(list(g.neighbors(node))) > 0:
            # other features
            neighbor_deg = feat_matrix[list(g.neighbors(node)), 0]
            feat_matrix[node, 1:] = neighbor_deg.min(), neighbor_deg.max(), neighbor_deg.mean(), neighbor_deg.std()

    if normalize:
        feat_matrix = (feat_matrix - feat_matrix.mean(0, keepdims=True)) / (feat_matrix.std(0, keepdims=True) + 1e-9)

    return feat_matrix, g

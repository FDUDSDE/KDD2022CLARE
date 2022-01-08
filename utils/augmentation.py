import networkx as nx
import numpy as np
from tqdm import tqdm
import torch


def feature_augmentation(nodes, edges, normalize=True):
    """Node feature augmentation `[deg(u), min(deg(N)), max(deg(N)), mean(deg(N)), std(deg(N))]` """
    g = nx.Graph(edges)
    g.add_nodes_from(nodes)

    num_node = len(nodes)

    node_degree = [g.degree[node] for node in range(num_node)]

    feat_matrix = np.zeros([num_node, 5], dtype=np.float32)
    feat_matrix[:, 0] = np.array(node_degree).squeeze()

    new_graph = nx.Graph()
    for node in tqdm(range(num_node), desc="Feature Computation"):
        if len(list(g.neighbors(node))) > 0:
            neighbor_deg = feat_matrix[list(g.neighbors(node)), 0]
            feat_matrix[node, 1:] = neighbor_deg.min(), neighbor_deg.max(), neighbor_deg.mean(), neighbor_deg.std()

    if normalize:
        feat_matrix = (feat_matrix - feat_matrix.mean(0, keepdims=True)) / (feat_matrix.std(0, keepdims=True) + 1e-9)

    for node in tqdm(range(num_node), desc="Feature Augmentation"):
        node_feat = feat_matrix[node, :].astype(np.float32)
        new_graph.add_node(node, node_feature=torch.from_numpy(node_feat))
    new_graph.add_edges_from(edges)
    return new_graph, feat_matrix

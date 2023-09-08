import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv, global_add_pool


class GNNEncoder(torch.nn.Module):
    r"""Graph Neural Networks for node/graph Encoding, Customized Settings include Dimension, Layer, and GNN-Type"""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=2, gnn_type="GAT"):
        super().__init__()

        if gnn_type == "GCN":
            conv = GCNConv
        elif gnn_type == "GAT":
            conv = GATConv
        elif gnn_type == "TransformerConv":
            conv = TransformerConv
        elif gnn_type == "SAGE":
            conv = SAGEConv
        else:
            raise KeyError("GNN_TYPE can only be GAT, GCN, SAGE, and TransformerConv")

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = torch.nn.LeakyReLU()

        self.pool = global_add_pool

        if n_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(n_layer))
        elif n_layer == 2:
            self.conv_layers = torch.nn.ModuleList([conv(input_dim, hidden_dim), conv(hidden_dim, output_dim)])
        else:
            layers = [conv(input_dim, hidden_dim)]
            for _ in range(n_layer - 2):
                layers.append(conv(hidden_dim, hidden_dim))
            layers.append(conv(hidden_dim, output_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, batch, return_node=False):
        for graph_conv in self.conv_layers[0:-1]:
            x = graph_conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)
        graph_emb = self.pool(node_emb, batch)

        # return both centric nodes' and subgraphs' embeddings
        return node_emb, graph_emb

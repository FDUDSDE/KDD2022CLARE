import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class GNNEncoder(nn.Module):
    def __init__(self, args):
        super(GNNEncoder, self).__init__()

        self.dropout = args.dropout
        self.n_layers = args.n_layers
        self.args = args

        input_dim, hidden_dim, output_dim = 5,  args.hidden_dim, args.output_dim
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, hidden_dim))
        conv_model = self.build_conv_model(args.conv_type)

        self.convs = nn.ModuleList()
        self.learnable_skip = nn.Parameter(torch.ones(self.n_layers, self.n_layers))

        for l in range(args.n_layers):
            hidden_input_dim = hidden_dim * (l + 1)
            self.convs.append(conv_model(hidden_input_dim, hidden_dim))

        post_input_dim = hidden_dim * (args.n_layers + 1)

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.conv_type = args.conv_type

    def build_conv_model(self, model_type):
        """Build GCN / GIN / SAGE """
        if model_type == "GCN":
            return pyg_nn.GCNConv
        elif model_type == "GIN":
            return lambda i, h: pyg_nn.GINConv(
                nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h))
            )
        elif model_type == "SAGE":
            return pyg_nn.SAGEConv

    def forward(self, data):
        data = data.to(self.args.device)
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        x = self.pre_mp(x)
        all_emb = x.unsqueeze(1)
        emb = x

        for i in range(len(self.convs)):
            skip_vals = self.learnable_skip[i, :i + 1].unsqueeze(0).unsqueeze(-1)
            curr_emb = all_emb * torch.sigmoid(skip_vals)
            curr_emb = curr_emb.view(x.size(0), -1)
            x = self.convs[i](curr_emb, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            emb = torch.cat((emb, x), 1)
            all_emb = torch.cat((all_emb, x.unsqueeze(1)), 1)
        emb = pyg_nn.global_add_pool(emb, batch)
        emb = self.post_mp(emb)
        return emb

import torch.nn as nn
import torch
from torch_geometric.nn import GINConv


# Embedding updater
class GCN(nn.Module):
    def __init__(self, input_channels=65, hidden_channels=64):
        super(GCN, self).__init__()
        # self.conv = GCNConv(input_channels, hidden_channels)
        self.conv = GINConv(
            nn.Sequential(nn.Linear(input_channels, hidden_channels), nn.ReLU(),
                          nn.Linear(hidden_channels, hidden_channels))
        )

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hid = torch.tanh(self.fc1(x))
        hid = torch.tanh(self.fc2(hid))
        return self.fc3(hid)

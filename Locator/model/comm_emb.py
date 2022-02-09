from .gnn import GNNEncoder
import torch
import torch.nn as nn


class CommunityOrderEmbedding(nn.Module):
    def __init__(self, args):
        super(CommunityOrderEmbedding, self).__init__()

        self.encoder = GNNEncoder(args)
        self.margin = args.margin
        self.device = args.device

    def forward(self, emb_as, emb_bs):
        return emb_as, emb_bs

    def predict(self, pred):
        emb_as, emb_bs = pred

        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=self.device), emb_bs - emb_as) ** 2, dim=1)
        return e

    def criterion(self, pred, labels):
        emb_as, emb_bs = pred
        e = torch.sum(torch.max(torch.zeros_like(emb_as, device=self.device), emb_bs - emb_as) ** 2, dim=1)
        margin = self.margin
        e[labels == 0] = torch.max(torch.tensor(0.0, device=self.device), margin - e)[labels == 0]
        return torch.sum(e)

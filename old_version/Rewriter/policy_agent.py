import torch.optim as optim
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical
from .policy_net import MLP, GCN
from .symbol import EXPAND, EXCLUDE, VIRTUAL_EXCLUDE_NODE, VIRTUAL_EXPAND_NODE


class RewritingAgent:
    def __init__(self, args):
        self.args = args
        self.exclude_net, self.expand_net, self.gcn = self.network_init()
        self.exclude_opt = optim.Adam(self.exclude_net.parameters(), lr=args.agent_lr)
        self.expand_opt = optim.Adam(self.expand_net.parameters(), lr=args.agent_lr)

    def network_init(self):
        exclude_net = MLP(65, 32, 1)
        expand_net = MLP(65, 32, 1)
        gcn = GCN()
        return exclude_net, expand_net, gcn

    def learn(self, log_prob, reward, cal_type=EXCLUDE):
        loss = (-log_prob * reward).sum()
        if cal_type == EXCLUDE:
            self.exclude_opt.zero_grad()
            loss.backward()
            self.exclude_opt.step()
        else:
            self.expand_opt.zero_grad()
            loss.backward()
            self.expand_opt.step()

    def choose_action(self, com_obj, cal_type=EXCLUDE):
        idx_list = []
        if cal_type == EXCLUDE:
            idx_list = sorted([idx for idx in com_obj.mapping.keys() if com_obj.mapping[idx] in com_obj.pred_com])
            if len(idx_list) <= 1:
                return None
        elif cal_type == EXPAND:
            expander = sorted([node for node in com_obj.nodes if node not in com_obj.pred_com])
            if len(expander) <= 1 or not com_obj.expand:
                return None
            idx_list = sorted([idx for idx in com_obj.mapping.keys() if com_obj.mapping[idx] in expander])

        tmp_mapping = {idx: com_obj.mapping[idx_val] for idx, idx_val in enumerate(idx_list)}
        feat = com_obj.feat_mat[idx_list, :]

        if cal_type == EXCLUDE:
            score = self.exclude_net(torch.FloatTensor(feat))
        else:
            score = self.expand_net(torch.FloatTensor(feat))
        probs = F.softmax(np.squeeze(score), dim=-1)

        action_dist = Categorical(probs)
        action = action_dist.sample()  # node idx
        log_prob = action_dist.log_prob(action)

        node = tmp_mapping[action.item()]

        if (cal_type == EXCLUDE and node == VIRTUAL_EXCLUDE_NODE) or \
                (cal_type == EXPAND and node == VIRTUAL_EXPAND_NODE):
            return None
        return {
            "log_prob": log_prob,
            "node": node
        }

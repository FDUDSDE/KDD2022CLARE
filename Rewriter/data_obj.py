import random
import networkx as nx
import numpy as np
import torch
from .symbol import VIRTUAL_EXCLUDE_NODE, VIRTUAL_EXPAND_NODE
from utils import generate_ego_net, generate_outer_boundary


class Community:
    def __init__(self, feat_mat, pred_com, true_com, nodes, subgraph, mapping, expand=True):
        """
        :param feat_mat: node feature matrix
        :param pred_com: init predicted community
        :param true_com: corresponding ground-truth community
        :param nodes: nodes set ( pred_com + outer_boundary )
        :param subgraph: `nx.Graph` object
        :param mapping:
        :param expand:
        """
        self.nodes = nodes
        self.feat_mat = feat_mat
        self.pred_com = pred_com
        self.true_com = true_com
        self.graph = subgraph
        self.mapping = mapping
        self.expand = expand

        # Virtual node for stopping exclusion
        self.nodes.append(VIRTUAL_EXCLUDE_NODE)
        self.pred_com.append(VIRTUAL_EXCLUDE_NODE)
        self.mapping[len(self.nodes)-1] = VIRTUAL_EXCLUDE_NODE

        # Virtual node for stopping expansion
        self.nodes.append(VIRTUAL_EXPAND_NODE)
        self.mapping[len(self.nodes)-1] = VIRTUAL_EXPAND_NODE

        # Virtual nodes embedding (all zero)
        self.feat_mat = np.vstack((self.feat_mat, np.zeros((2, 64))))

        # Augment node embedding with POSITION-FLAG
        position_flag = self.generate_position_flag()
        self.feat_mat = np.hstack((position_flag, self.feat_mat))

    def generate_position_flag(self):
        result = np.zeros((self.feat_mat.shape[0], 1))
        for idx, node in self.mapping.items():
            if node in self.pred_com:
                result[idx, 0] = 1
        return result

    def compute_cost(self, choice="f1"):
        """Compute the cost brought by current **Action**"""
        intersection = set(self.true_com) & set(self.pred_com)

        precision = len(intersection) / len(self.pred_com)
        recall = len(intersection) / len(self.true_com)
        f = 2 * precision * recall / (precision + recall + 1e-9)

        jaccard = len(intersection) / (len(self.true_com) + len(self.pred_com) - len(intersection))
        upper_base = 10
        if choice == "jaccard":
            return jaccard * upper_base
        elif choice == "f1":
            return f * upper_base
        elif choice == "hybrid":
            return jaccard * upper_base + f * upper_base
        elif choice == "precision":
            return precision * upper_base
        else:
            raise NotImplementedError

    def step(self, emb_updater):
        """Move into the next **State** with `emb_updater`"""
        edges = list(self.graph.subgraph(self.pred_com).edges())
        # print(edges)
        e_tensor = torch.zeros((2, len(edges)), dtype=int)
        revert_mapping = {node: idx for idx, node in self.mapping.items()}

        for i in range(len(edges)):
            u, v = edges[i]
            e_tensor[0][i], e_tensor[1][i] = revert_mapping[u], revert_mapping[v]

        # Update Node Embedding
        x_tensor = torch.FloatTensor(self.feat_mat)
        new_x = emb_updater(x_tensor, e_tensor)

        position_flag = self.generate_position_flag()
        new_feat_mat = np.hstack((position_flag, new_x.detach().numpy()))
        return new_feat_mat

    def apply_exclude(self, node, cost_choice):
        """Apply Exclude action and return corresponding Reward"""
        pre_cost = self.compute_cost(choice=cost_choice)
        if node in self.pred_com:
            self.pred_com.remove(node)
        return self.compute_cost(choice=cost_choice) - pre_cost

    def apply_expand(self, node, cost_choice):
        """Apply Expand action and return corresponding Reward"""
        pre_cost = self.compute_cost(choice=cost_choice)
        if node not in self.pred_com:
            self.pred_com.append(node)
        return self.compute_cost(choice=cost_choice) - pre_cost


class DataProcessor(object):
    def __init__(self, args, dataset, feat_mat, graph, train_comms, valid_comms):
        self.args = args
        self.dataset = dataset
        self.graph = graph
        self.feat_mat = feat_mat
        self.train_comms, self.valid_comms = train_comms, valid_comms

    def generate_data(self, batch_size=64, valid=False):
        comms = self.train_comms if not valid or len(self.valid_comms) == 0 else self.valid_comms

        train_set = []

        for _ in range(batch_size):
            # Step1, randomly choose a known community
            true_com = random.choice(comms)

            # Step2, choose a node by degree distribution
            subgraph = self.graph.subgraph(true_com)
            degree = nx.degree(subgraph)
            nodes = sorted(list(subgraph.nodes()))
            node2degree = [degree[node] for node in nodes]
            degree = np.array(node2degree)
            degree = degree / np.sum(degree, axis=0)
            root_node = np.random.choice(nodes, p=degree.ravel())

            # Step3, extract the root-node k-ego-net and outer boundary
            ego_net = generate_ego_net(self.graph, root_node, k=self.args.n_layers, max_size=self.args.comm_max_size, choice="neighbors")

            if valid:
                train_set.append(ego_net)
                continue

            outer_nodes = generate_outer_boundary(self.graph, ego_net, max_size=10)

            expand = False if len(outer_nodes) == 0 else True

            all_nodes = sorted(ego_net + outer_nodes)

            feat_mat = self.feat_mat[all_nodes, :]
            mapping = {idx: node for idx, node in enumerate(all_nodes)}

            com_obj = Community(feat_mat, ego_net, true_com, all_nodes, self.graph.subgraph(all_nodes), mapping,
                                expand=expand)

            train_set.append(com_obj)
        return train_set

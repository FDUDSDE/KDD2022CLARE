import random
from .gnn import GNNEncoder
import torch
import numpy as np
import torch.optim as optim
import time
from utils import prepare_locator_train_data, generate_ego_net
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, subgraph
import math


class CommMatching:
    def __init__(self, args, graph_data, train_communities, val_communities, device=torch.device("cuda:0")):
        print(f"Community Locator init ... ")
        self.args = args

        self.graph_data = graph_data
        self.seed_nodes = {node for com in train_communities + val_communities for node in com}

        self.train_comms = train_communities
        self.val_comms = val_communities

        self.num_node, input_dim = graph_data.x.size(0), graph_data.x.size(1)
        self.device = device

        self.gnn_encoder = GNNEncoder(input_dim, args.hidden_dim, args.output_dim, args.n_layers,
                                      gnn_type=args.gnn_type)
        self.gnn_encoder.to(device)
        print(self.gnn_encoder)
        print("Community Locator finish initialization!\n")

    def train(self, subg_max_size=20, num_hop=2):
        optimizer = optim.Adam(self.gnn_encoder.parameters(), lr=self.args.locator_lr, weight_decay=1e-5)

        print("Training Order Embedding ... ")
        for epoch in range(1, self.args.locator_epoch + 1):
            self.gnn_encoder.train()
            optimizer.zero_grad()

            st = time.time()

            node_list = random.sample(list(self.seed_nodes), self.args.locator_batch_size)
            batch_data, corrupt_batch = prepare_locator_train_data(node_list, self.graph_data, max_size=subg_max_size,
                                                                   num_hop=num_hop)

            batch_data = batch_data.to(self.device)
            corrupt_batch = corrupt_batch.to(self.device)

            _, summary = self.gnn_encoder(batch_data.x, batch_data.edge_index, batch_data.batch)
            _, corrupt_summary = self.gnn_encoder(corrupt_batch.x, corrupt_batch.edge_index, corrupt_batch.batch)

            # positive pair (corrupt_summary, summary) where corrupt one is a subgraph of one
            # negative pair (random_shuffled_summary, summary) where no subgraph / isomorphism exists
            shuf_index = torch.randperm(summary.size(0))
            random_summary = summary[shuf_index]

            emb_as = torch.cat((corrupt_summary, random_summary), dim=0)
            emb_bs = torch.cat((summary, summary), dim=0)

            labels = torch.tensor([1] * summary.size(0) + [0] * summary.size(0)).to(self.device)
            e = torch.sum(torch.max(torch.zeros_like(emb_as, device=self.device), emb_as - emb_bs) ** 2, dim=1)

            e[labels == 0] = torch.max(torch.tensor(0.0, device=self.device), self.args.margin - e)[labels == 0]
            loss = torch.mean(e)

            loss.backward()
            optimizer.step()
            print("***epoch: {:04d} | ORDER EMBEDDING train_loss: {:.5f} | cost time {:.3}s".format(epoch, loss,
                                                                                                    time.time() - st))

        print("Order Embedding Finish Training!\n")

    def generate_all_node_emb(self):
        self.gnn_encoder.eval()

        batch = Batch().from_data_list([self.graph_data])
        batch = batch.to(self.device)
        # return all nodes embedding
        node_emb, _ = self.gnn_encoder(batch.x, batch.edge_index, batch.batch, return_node=True)
        return node_emb

    def generate_target_community_emb(self, comms):
        batch = []
        self.gnn_encoder.eval()
        for community in comms:
            edge_index, _ = subgraph(community, self.graph_data.edge_index, relabel_nodes=True, num_nodes=self.num_node)
            node_x = self.graph_data.x[community]  # node features
            g_data = Data(x=node_x, edge_index=edge_index)
            batch.append(g_data)
        batch = Batch().from_data_list(batch).to(self.device)

        _, comms_emb = self.gnn_encoder(batch.x, batch.edge_index, batch.batch)
        return comms_emb

    def predict_community(self, nx_graph, comm_max_size=20, k=2):
        self.gnn_encoder.eval()

        # Step 1 generate embeddings
        query_emb = self.generate_target_community_emb(self.train_comms + self.val_comms)
        query_emb = query_emb.detach().cpu().numpy()

        ### extract each node's k-ego-net as candidates for further matching
        batch_size = 4096
        batch_len = math.ceil(self.num_node / batch_size)
        all_emb = np.zeros((self.num_node, self.args.output_dim))
        for i in range(batch_len):
            minn, maxx = i * batch_size, min((i + 1) * batch_size, self.num_node)

            batch, _ = prepare_locator_train_data(list(range(minn, maxx)), data=self.graph_data, max_size=20, num_hop=k)
            batch = batch.to(self.device)
            _, comms_emb = self.gnn_encoder(batch.x, batch.edge_index, batch.batch)
            comms_emb = comms_emb.detach().cpu().numpy()

            all_emb[minn:maxx, :] = comms_emb
            print(f"***Generate nodes embedding from idx {minn} to {maxx}")

        # Step 2 matching
        print(f"\nStart Matching ... ")
        pred_comms, seeds = [], []
        pred_size = self.args.num_pred
        single_pred_size = int(pred_size / query_emb.shape[0])
        for i in range(query_emb.shape[0]):
            q_emb = query_emb[i, :]
            distance = np.sqrt(np.sum(np.asarray(q_emb - all_emb) ** 2, axis=1))
            sort_dic = list(np.argsort(distance))
            # print(sort_dic[:5])

            if len(pred_comms) >= pred_size:
                break

            length = 0
            for node in sort_dic:
                if length >= single_pred_size:
                    break

                neighs = generate_ego_net(nx_graph, int(node), k=k, max_size=comm_max_size, choice="neighbors")

                if neighs not in pred_comms and len(pred_comms) < pred_size and node not in self.seed_nodes and node \
                        not in seeds:
                    seeds.append(node)
                    pred_comms.append(neighs)
                    length += 1
                    # print(f"[Generate] seed node {node}, community {neighs}")
        lengths = np.array([len(pred_com) for pred_com in pred_comms])
        print(f"[Generate] Pred size {len(pred_comms)}, Avg Length {np.mean(lengths):.04f}\n")

        return pred_comms

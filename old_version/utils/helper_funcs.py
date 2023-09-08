import random
import numpy as np
import scipy.stats as stats
import torch
from deepsnap.graph import Graph as DSGraph
from deepsnap.batch import Batch


def get_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def sample_neigh(graphs, size):
    """Sampling function during training"""
    ps = np.array([len(g) for g in graphs], dtype=np.float64)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))

    while True:
        idx = dist.rvs()
        graph = graphs[idx]

        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]

        if len(list(graph.neighbors(start_node))) > 0:
            frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        else:
            return graph, neigh

        visited = {start_node}
        while len(neigh) < size and len(frontier) > 0:
            new_node = random.choice(list(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)

            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]
        return graph, neigh


def generate_ego_net(graph, start_node, k=1, max_size=15, choice="subgraph"):
    """Generate **k** ego-net"""
    q = [start_node]
    visited = [start_node]

    iteration = 0
    while True:
        if iteration >= k:
            break
        length = len(q)
        if length == 0 or len(visited) >= max_size:
            break

        for i in range(length):
            # Queue pop
            u = q[0]
            q = q[1:]

            for v in list(graph.neighbors(u)):
                if v not in visited:
                    q.append(v)
                    visited.append(v)
                if len(visited) >= max_size:
                    break
            if len(visited) >= max_size:
                break
        iteration += 1
    visited = sorted(visited)
    # print(visited)
    return visited if choice == "neighbors" else graph.subgraph(visited)


def generate_outer_boundary(graph, com_nodes, max_size=20):
    """For a given graph and a community `com_nodes`, generate its outer-boundary"""
    outer_nodes = []
    for node in com_nodes:
        outer_nodes += list(graph.neighbors(node))
    outer_nodes = list(set(outer_nodes) - set(com_nodes))
    outer_nodes = sorted(outer_nodes)
    return outer_nodes if len(outer_nodes) <= max_size else outer_nodes[:max_size]


def batch2graphs(graphs, device=None):
    """Transform `List[nx.Graph]` into `DeepSnap.Batch` object"""
    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = batch.to(get_device(device=device))
    return batch


def generate_embedding(batch, model, device=None):
    batches = batch2graphs(batch, device=device)
    pred = model.encoder(batches)
    pred = pred.cpu().detach().numpy()
    return pred


def split_communities(comms, n_train, n_val=0):
    print(f"Split communities, # Train {n_train}, # Val {n_val}, # Test {len(comms) - n_train - n_val}")
    # train, val, test
    return comms[:n_train], comms[n_train:n_train + n_val], comms[n_train + n_val:]


def random_subgraphs(graph, pred_size=1000, max_size=12):
    nodes = list(graph.nodes())
    seeds = []
    while len(seeds) < pred_size:
        node = random.choice(nodes)
        if node not in seeds:
            seeds.append(node)
    pred_comms = [generate_ego_net(graph, seed, k=1, max_size=max_size, choice="neighbors") for seed in seeds]
    return pred_comms

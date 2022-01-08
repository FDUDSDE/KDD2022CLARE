import numpy as np
import networkx as nx


def load_comm_or_edges(name, file_type="cmty"):
    with open(f'{name}/{name}-1.90.{file_type}.txt') as file:
        contents = file.read().strip().split('\n')
        contents = [[int(i) for i in x.split()] for x in contents]
    return contents


def load_dataset(name):
    print(f"{name.upper()}")
    print("=" * 20)

    communities = load_comm_or_edges(name, file_type="cmty")
    edges = load_comm_or_edges(name, file_type="ungraph")

    # Community size analysis
    lengths = np.array([len(com) for com in communities])
    print("Comm size: avg community size {:.4f}, max {}, min{}".format(np.mean(lengths), np.max(lengths),
                                                                       np.min(lengths)))

    # Graph meta
    nodes = {node for e in edges for node in e}
    in_com_nodes = {node for com in communities for node in com}
    print("Graph ：nodes {}, edges {}, avg node degree {}".format(len(nodes), len(edges),
                                                                 round(len(edges) / len(nodes), 4)))
    print("Comm node：nodes in communities {}, percent {}".format(len(in_com_nodes),
                                                                 round(len(in_com_nodes) / len(nodes), 4)))

    graph = nx.Graph(edges)
    degree = nx.degree(graph)
    degree_all = np.array([degree[node] for node in sorted(nodes)])
    degree_comnode = np.array([degree[node] for node in sorted(list(in_com_nodes))])

    print("         nodes in communities avg degree {:.4f}, all nodes avg degree {:.4f}".format(np.mean(degree_comnode),
                                                                                                np.mean(degree_all)))

    cluster_coefficient = nx.clustering(graph)
    cluster_coefficient_all = np.array([cluster_coefficient[node] for node in sorted(nodes)])
    cluster_coefficient_comnode = np.array([cluster_coefficient[node] for node in sorted(in_com_nodes)])
    print("         nodes in communities avg cc {:.4f}, all nodes avg cc {:.4f}".format(
        np.mean(cluster_coefficient_comnode),
        np.mean(cluster_coefficient_all)))


for name in ["amazon", "dblp", "lj"]:
    load_dataset(name)

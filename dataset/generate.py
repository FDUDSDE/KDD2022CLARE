import random
import numpy as np
import seaborn as sns
import pathlib
from tqdm import tqdm

sns.set()
max_threshold = dict(lj=30, dblp=16, amazon=30)
seed = 0


def load_community(name):
    with open(f'raw/{name}_raw/com-{name}.top5000.cmty.txt') as fh:
        comms = fh.read().strip().split('\n')
        comms = [[int(i) for i in x.split()] for x in comms]

    lengths = np.array([len(com) for com in comms])
    print(lengths.shape, np.max(lengths), np.min(lengths))
    print(np.percentile(lengths, q=25), np.percentile(lengths, q=50), np.percentile(lengths, q=75),
          np.percentile(lengths, q=90))

    comms = [x for x in comms if len(x) <= max_threshold[name]]

    # Randomly choose 1000 communities
    rng = np.random.RandomState(seed)
    comms = rng.permutation(comms)
    comms = comms[:1000]

    lengths = [len(com) for com in comms]
    lengths = np.array(lengths)
    print(len(lengths), np.mean(lengths), np.max(lengths))

    return comms


def load_subgraph(comms, name):
    """Extract subgraphs (nodes and outer-boundary) from communities"""
    with open(f'raw/{name}_raw/com-{name}.ungraph.txt') as fh:
        edges = fh.read().strip().split('\n')

    nodes = {node for com in comms for node in com}
    edges = edges[4:]
    all_nodes = []
    for index in tqdm(range(len(edges)), desc="Extracting outer boundary"):
        contents = edges[index].split()
        if int(contents[0]) in nodes or int(contents[1]) in nodes:
            all_nodes.append(int(contents[0]))
            all_nodes.append(int(contents[1]))

    all_nodes = set(all_nodes)

    new_edges = []
    for index in tqdm(range(len(edges)), desc="Generating edges"):
        e = edges[index]
        contents = e.split()
        if int(contents[0]) in all_nodes and int(contents[1]) in all_nodes:
            new_edges.append([int(contents[0]), int(contents[1])])

    # unique
    new_edges = [[u, v] if u < v else [v, u] for u, v in new_edges if u != v]
    new_edges = list(set([tuple(t) for t in new_edges]))
    new_edges = [list(edge) for edge in new_edges]

    print("Finish! Now the generated new graph has nodes {}, edges {}, in-community nodes {}, percent{}".
          format(len(all_nodes), len(new_edges), len(nodes), round(len(nodes) / len(all_nodes), 4)))

    mapping = {u: i for i, u in enumerate(sorted(all_nodes))}
    edges = [[mapping[u], mapping[v]] for u, v in new_edges]
    edges = sorted(edges, key=lambda x: [x[0], x[1]])
    comms = [[mapping[node] for node in com] for com in comms]

    return edges, comms


def writ2file(edges, comms, name):
    root = pathlib.Path(name)
    root.mkdir(exist_ok=True, parents=True)
    with open(root / f'{name}-1.90.ungraph.txt', 'w') as fh:
        s = '\n'.join([f'{a}	{b}' for a, b in edges])
        fh.write(s)
    with open(root / f'{name}-1.90.cmty.txt', 'w') as fh:
        s = '\n'.join([' '.join([str(i) for i in x]) for x in comms])
        fh.write(s)


def create_hybrid_network(dataset1, dataset2, num_random_edges=5000):
    """Create hybrid network"""
    with open(f"{dataset1}/{dataset1}-1.90.ungraph.txt", 'r') as file:
        edges1 = file.read().strip().split('\n')
    edges1 = [[int(i) for i in e.split()] for e in edges1]
    nodes1 = {i for x in edges1 for i in x}
    offset = len(nodes1)

    with open(f"{dataset2}/{dataset2}-1.90.ungraph.txt", 'r') as file:
        edges2 = file.read().strip().split('\n')
    edges2 = [[int(i) + offset for i in e.split()] for e in edges2]
    nodes2 = {i for x in edges2 for i in x}

    # generate random edges
    random_edges = [[random.randint(0, offset), random.randint(offset, offset + len(nodes2))]
                    for _ in range(num_random_edges)]
    print(random_edges[:10])
    print(len(nodes1), len(edges1))
    print(len(nodes2), len(edges2))

    with open(f'{dataset1}/{dataset1}-1.90.cmty.txt') as file:
        communities = file.read().strip().split('\n')
        communities = [[int(i) for i in x.split()] for x in communities]

    edges = edges1 + random_edges + edges2

    writ2file(edges, communities, dataset1 + '_' + dataset2)
    return edges, communities


def generate_dataset():
    # Single
    for name in ["amazon", "dblp", "lj"]:
        comms = load_community(name)
        edges, comms = load_subgraph(comms, name)

        writ2file(edges, comms, name)
        print()

    # Hybrid
    create_hybrid_network("amazon", "dblp", num_random_edges=5000)
    create_hybrid_network("dblp", "amazon", num_random_edges=5000)
    create_hybrid_network("dblp", "lj", num_random_edges=10000)
    create_hybrid_network("lj", "dblp", num_random_edges=10000)


generate_dataset()

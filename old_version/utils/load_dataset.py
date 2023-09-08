def load(name):
    """Load snap dataset"""
    communities = open(f"./dataset/{name}/{name}-1.90.cmty.txt")
    edges = open(f"./dataset/{name}/{name}-1.90.ungraph.txt")

    communities = [[int(i) for i in x.split()] for x in communities]
    edges = [[int(i) for i in e.split()] for e in edges]
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]

    nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(nodes))}

    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communities = [[mapping[node] for node in com] for com in communities]

    print(f"[{name.upper()}], #Nodes {len(nodes)}, #Edges {len(edges)} #Communities {len(communities)}")

    return nodes, edges, communities

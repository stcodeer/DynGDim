import networkx as nx
import dgl

def build_networkx_from_torch_geometric(data):
    graph = nx.Graph()

    num_nodes = data.x.shape[0]

    graph.add_nodes_from(range(num_nodes))

    weight = dict()

    def get_hash(u, v):
        if u > v:
            u, v = v, u
        return u * num_nodes + v

    num_edges = data.edge_index.shape[1]

    for i in range(num_edges):
        u = data.edge_index[0][i].item()
        v = data.edge_index[1][i].item()
        if u == v:
            continue
        if get_hash(u, v) not in weight:
            graph.add_edge(u, v)
            weight[get_hash(u, v)] = 1 
        else:
            weight[get_hash(u, v)] = weight[get_hash(u, v)] + 1

    for u, v in graph.edges():
        graph[u][v]["weight"] = weight[get_hash(u, v)]

    print("Build Network Finished.")
    
    return graph

def build_networkx_from_dgl(data):
    return dgl.to_networkx(data)
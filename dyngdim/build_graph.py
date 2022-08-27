import networkx as nx
# import dgl
import torch_geometric
import torch
# from tqdm import trange


def get_hash(u, v, num_nodes):
    if u > v:
        u, v = v, u
    return u * num_nodes + v


def build_networkx_from_torch_geometric(data, directed=False):
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    num_edges = data.edge_index.shape[1]
    num_nodes = data.x.shape[0]

    graph.add_nodes_from(range(num_nodes))

    weight = dict()

    for i in range(num_edges):
        u = data.edge_index[0][i].item()
        v = data.edge_index[1][i].item()
        if u == v:
            continue
        h = get_hash(u, v, num_nodes)
        if get_hash(u, v, num_nodes) not in weight:
            graph.add_edge(u, v)
            weight[h] = 1
        else:
            weight[h] = weight[h] + 1

    for u, v in graph.edges():
        graph[u][v]["weight"] = weight[get_hash(u, v, num_nodes)]
    
    return graph


# def build_torch_geometric_from_networkx(data, y_structural, feature, num_nodes):
#     data_pyg = torch_geometric.utils.from_networkx(data)
    
#     data_pyg.train_mask = [0] * num_nodes
    
#     data_pyg.val_mask = [0] * num_nodes

#     data_pyg.test_mask = [1] * num_nodes

#     data_pyg.y = y_structural

#     data_pyg.x = feature
    
#     return data_pyg


# def build_networkx_from_dgl_graph(data):
#     return dgl.to_networkx(data)


# def build_networkx_from_dgl_hetero_graph(data):
#     graph = nx.Graph()
    
#     edge_x = []
#     edge_y = []
    
#     for etype in data.etypes:
#         edge_x += data.edges(etype=etype)[0].tolist()
#         edge_y += data.edges(etype=etype)[1].tolist()
        
#     num_edges = len(edge_x)
#     num_nodes = data.num_nodes(ntype=data.ntypes[0])
    
#     graph.add_nodes_from(range(num_nodes))
    
#     weight = dict()
    
#     for i in range(num_edges):
#         u = edge_x[i]
#         v = edge_y[i]
#         if u == v:
#             continue
#         h = get_hash(u, v, num_nodes)
#         if get_hash(u, v, num_nodes) not in weight:
#             graph.add_edge(u, v)
#             weight[h] = 1
#         else:
#             weight[h] = weight[h] + 1
    
            
#     for u, v in graph.edges():
#         graph[u][v]["weight"] = weight[get_hash(u, v, num_nodes)]
        
#     return graph


def build_torch_getometric_from_dgl_hetero_graph(data):
    edge_x = []
    edge_y = []
    
    for etype in data.etypes:
        edge_x += data.edges(etype=etype)[0].tolist()
        edge_y += data.edges(etype=etype)[1].tolist()
    
    feature = data.ndata['feature']

    label = data.ndata['label']
    
    graph = torch_geometric.data.Data(x=feature, y=label, edge_index=torch.tensor([edge_x, edge_y]))
    
    return graph


def graph_partitioning_torch_geometric(data, num_parts):
    
    cluster_data = torch_geometric.loader.ClusterData(data, num_parts=num_parts)
    
    train_loader = torch_geometric.loader.ClusterLoader(cluster_data, batch_size=1)
    
    datas = []
    
    for step, sub_data in enumerate(train_loader):
        datas.append(sub_data)
        
    return datas
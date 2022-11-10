import numpy as np

from dyngdim.anomaly_detection import *

from dyngdim.build_graph import *

from pygod.utils import load_data


times = np.logspace(0, 1.0, 20)

n_workers = 10


# datasets from https://github.com/pygod-team/data

datasets = ['weibo', 'reddit', 'inj_cora', 'inj_amazon', 'inj_flickr', 'gen_time', 'gen_100', 'gen_500', 'gen_1000', 'gen_5000', 'gen_10000']

dataset = "weibo"

graph_pyg = load_data(dataset)
print(graph_pyg)
# print(graph_pyg.is_directed())

print("Load Dataset Finished.")

num_nodes = len(graph_pyg.y)

graph_pygs = graph_partitioning_torch_geometric(graph_pyg, num_nodes // 1000)

print(graph_pygs)

num_total_edges = sum([graph_pyg_part.edge_index.shape[1] for graph_pyg_part in graph_pygs])

print("edges: " + str(graph_pyg.edge_index.shape[1]) + "  -->  " + str(num_total_edges))

y_structural = []
y_dyngdim = [[] for i in range(len(times))]
y_pygod = [[] for i in range(11)]
y_centrality = [[] for i in range(5)]
 
for graph_pyg_part in graph_pygs:
    if dataset == "weibo" or dataset == "reddit":
        y_structural_part = graph_pyg_part.y.int().tolist()
    else:
        y_structural_part = (graph_pyg_part.y >> 1 & 1).int().tolist()
        
    y_structural += y_structural_part
    
    graph_networkx = build_networkx_from_torch_geometric(graph_pyg_part)
    
    graph = dyngdim(graph_networkx, y_structural_part, times, dataset)
    
    graph.graph_anomaly_detection_dyngdim(n_workers=n_workers)
    
    for i in range(len(y_dyngdim)):
        y_dyngdim[i] += graph.local_dimensions[i].tolist()
    
    graph.graph_anomaly_detection_centrality()
    
    for i in range(len(y_centrality)):
        y_centrality[i] += graph.local_dimensions[i].tolist()
    
    graph.graph_anomaly_detection_pygod(graph_pyg_part)
    
    for i in range(len(y_pygod)):
        y_pygod[i] += graph.local_dimensions[i].tolist()

print("combaring all anomaly scores")

graph.plot_roc_auc_score(y_dyngdim, y_structural, "DynGDim")

graph.plot_roc_auc_score_times(display=True)

graph.plot_roc_auc_score(y_centrality, y_structural, "Centrality")

graph.plot_roc_auc_score(y_pygod, y_structural, "PyGOD")
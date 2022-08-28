import numpy as np

from dyngdim.anomaly_detection import *

from dyngdim.build_graph import *

from dgl.data.fraud import FraudDataset

import torch

times = np.logspace(0, 1.0, 20)

n_workers = 10


# datasets from https://github.com/dmlc/dgl/blob/master/python/dgl/data/fraud.py
#    The dataset includes two multi-relational graphs extracted from Yelp and Amazon
#    where nodes represent fraudulent reviews or fraudulent reviewers.

datasets = ['yelp', 'amazon']

dataset = 'amazon'

data = FraudDataset(dataset)

graph_dgl = data[0]

print(graph_dgl)

graph_pyg = build_torch_getometric_from_dgl_hetero_graph(graph_dgl)

print(graph_pyg)

num_nodes = len(graph_pyg.y)

graph_pygs = graph_partitioning_torch_geometric(graph_pyg, num_nodes // 1000)

print(graph_pygs)

y_structural = []
y_dyngdim = [[] for i in range(len(times))]
y_pygod = [[] for i in range(11)]
y_centrality = [[] for i in range(5)]
 
for graph_pyg_part in graph_pygs:
    y_structural += graph_pyg_part.y
    
    graph_networkx = build_networkx_from_torch_geometric(graph_pyg_part)
    
    graph = dyngdim(graph_networkx, graph_pyg_part.y, times, dataset)
    
    graph.graph_anomaly_detection_dyngdim(n_workers=n_workers)
    
    for i in range(len(y_dyngdim)):
        y_dyngdim[i] += graph.local_dimensions[i].tolist()
    
    graph.graph_anomaly_detection_centrality()
    
    for i in range(len(y_centrality)):
        y_centrality[i] += graph.local_dimensions[i].tolist()
    
    graph.graph_anomaly_detection_pygod(graph_pyg_part)
    
    for i in range(len(y_pygod)):
        y_pygod[i] += graph.local_dimensions[i].tolist()

    # graph.plot_local_dimensions_and_outliers(display=True)

    # graph.plot_network_structure(display=True)

print("combaring all anomaly scores")

graph.plot_roc_auc_score(y_dyngdim, y_structural, "DynGDim")

graph.plot_roc_auc_score_times(display=True)

graph.plot_roc_auc_score(y_centrality, y_structural, "Centrality")

graph.plot_roc_auc_score(y_pygod, y_structural, "PyGOD")

    

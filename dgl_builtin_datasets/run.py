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

graph_networkx = build_networkx_from_torch_geometric(graph_pyg)

graph = dyngdim(graph_networkx, graph_pyg.y, times, dataset)

# graph.graph_anomaly_detection_dyngdim(n_workers=n_workers)

# graph.graph_anomaly_detection_centrality()

graph.graph_anomaly_detection_pygod(graph_pyg)

# graph.plot_local_dimensions_and_outliers(display=True)

# graph.plot_network_structure(display=True)

    

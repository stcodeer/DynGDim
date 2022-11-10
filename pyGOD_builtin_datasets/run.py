
from pygod.utils import load_data
import torch

import numpy as np

from dyngdim.anomaly_detection import *

from dyngdim.build_graph import *


times = np.logspace(0, 1.0, 20)

n_workers = 1


# datasets from https://github.com/pygod-team/data

datasets = ['weibo', 'reddit', 'inj_cora', 'inj_amazon', 'inj_flickr', 'gen_time', 'gen_100', 'gen_500', 'gen_1000', 'gen_5000', 'gen_10000']

dataset = "reddit"

graph_pyg = load_data(dataset)
print(graph_pyg)
# print(graph_pyg.is_directed())

print("Load Dataset Finished.")

is_semi_supervised = False

# is_semi_supervised = hasattr(data, "train_mask")

# if is_semi_supervised:
#     print("train_mask: ", sum(data.train_mask))
#     print("val_mask: ", sum(data.val_mask))
#     print("test_mask: ", sum(data.test_mask))

y_structural = []

if dataset == "weibo" or dataset == "reddit":
    y_structural = graph_pyg.y.int().tolist()
else:
    y_structural = (graph_pyg.y >> 1 & 1).int().tolist()

graph_networkx = build_networkx_from_torch_geometric(graph_pyg)

graph = dyngdim(graph_networkx, y_structural, times, dataset, is_semi_supervised)

if is_semi_supervised:
    train_mask = [graph_pyg.train_mask[i] for i in range(graph.num_nodes)]
    test_mask = graph_pyg.test_mask
else:
    train_mask = []
    test_mask = []

graph.graph_anomaly_detection_dyngdim(train_mask, test_mask, n_workers)

graph.plot_roc_auc_score_times(display=True)

graph.graph_anomaly_detection_centrality()

graph.graph_anomaly_detection_pygod(graph_pyg)

# graph.plot_local_dimensions_and_outliers(display=True)

# graph.plot_network_structure(display=True)
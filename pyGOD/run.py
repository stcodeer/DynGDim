import numpy as np

import networkx as nx

from dyngdim.anomaly_detection import *

from dyngdim.build_graph import *

from pygod.utils import load_data


times = np.logspace(0, 1.0, 20)

n_workers = 5


# datasets from https://github.com/pygod-team/data

datasets = ['weibo', 'reddit', 'inj_cora', 'inj_amazon', 'inj_flickr', 'gen_time', 'gen_100', 'gen_500', 'gen_1000', 'gen_5000', 'gen_10000']

dataset = "inj_cora"

data = load_data(dataset)
print(type(data))
print(data)

print("Load Dataset Finished.")


# total_train_mask = sum(data.train_mask)
# print("train_mask: ", total_train_mask)
# total_val_mask = sum(data.val_mask)
# print("val_mask: ", total_val_mask)
# total_test_mask = sum(data.test_mask)
# print("test_mask: ", total_test_mask)


y_structural = []

if dataset == "weibo" or dataset == "reddit":
    y_structural = data.y.int().tolist()
else:
    y_structural = (data.y >> 1 & 1).int().tolist()

networkx = build_networkx_from_torch_geometric(data)

graph = dyngdim(networkx, y_structural, times, dataset)

# train_mask = [data.val_mask[i] or data.train_mask[i] for i in range(graph.num_nodes)]
# test_mask = data.test_mask
train_mask = 0
test_mask = 0

graph.graph_anomaly_detection(train_mask, test_mask, n_workers)

print("-----------------------scores-----------------------")

print("roc_auc_score:", graph.score)
print("roc_auc_score(argmax):", np.argmax(graph.score))
print("roc_auc_score(max):", max(graph.score))

# graph.plot_local_dimensions_and_outliers(display=True)

# graph.plot_network_structure(display=True)
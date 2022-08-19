import numpy as np

from dyngdim.anomaly_detection import *

from dyngdim.build_graph import *

from pygod.utils import load_data


times = np.logspace(0, 1.0, 2)

n_workers = 5


# datasets from https://github.com/pygod-team/data

datasets = ['weibo', 'reddit', 'inj_cora', 'inj_amazon', 'inj_flickr', 'gen_time', 'gen_100', 'gen_500', 'gen_1000', 'gen_5000', 'gen_10000']

dataset = "inj_cora"

data = load_data(dataset)
print(type(data))
print(data)

print("Load Dataset Finished.")

is_semi_supervised = hasattr(data, "train_mask")

if is_semi_supervised:
    print("train_mask: ", sum(data.train_mask))
    print("val_mask: ", sum(data.val_mask))
    print("test_mask: ", sum(data.test_mask))

y_structural = []

if dataset == "weibo" or dataset == "reddit":
    y_structural = data.y.int().tolist()
else:
    y_structural = (data.y >> 1 & 1).int().tolist()

networkx = build_networkx_from_torch_geometric(data)

graph = dyngdim(networkx, [y_structural[i] for i in range(len(networkx))], times, dataset, is_semi_supervised)

if is_semi_supervised:
    train_mask = [data.train_mask[i] for i in range(graph.num_nodes)]
    test_mask = data.test_mask
else:
    train_mask = []
    test_mask = []

graph.graph_anomaly_detection_dyngdim(train_mask, test_mask, n_workers)

graph.plot_roc_auc_score(display=True)

graph.graph_anomaly_detection_centrality()

graph.graph_anomaly_detection_pygod(data, GAAN)

# graph.plot_local_dimensions_and_outliers(display=True)

# graph.plot_network_structure(display=True)
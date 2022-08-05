import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys

import numpy as np

import torch

import networkx as nx

import matplotlib.pyplot as plt

from dyngdim import run_local_dimension
from dyngdim import compute_global_dimension
from dyngdim import run_single_source, get_initial_measure
from dyngdim import run_all_sources

from pygod.utils import load_data

from sklearn.metrics import roc_auc_score


# 0. data

# datasets from https://github.com/pygod-team/data

datasets = ['weibo', 'reddit', 'inj_cora', 'inj_amazon', 'inj_flickr', 'gen_time', 'gen_100', 'gen_500', 'gen_1000', 'gen_5000', 'gen_10000']

dataset = "weibo"

data = load_data(dataset)
print(data)

num_nodes = data.x.shape[0]
num_features = data.x.shape[1]
num_edges = data.edge_index.shape[1]

print("Load Dataset Finished.")


# 1. params in DynGDim

t_min = 0.0
t_max = 1.0
n_t = 20
times = np.logspace(t_min, t_max, n_t)

n_workers = 5

# 2. get mask

# print(data.train_mask)
# print(data.val_mask)
# print(data.test_mask)


# 3. get feature

# with open(dataset + "_feature.txt",'w') as fp:
#     for x in range(data.x.shape[0]):
#         for y in range(data.x.shape[1]):
#             if data.x[x][y] > 0:
#                 print(x, y, data.x[x][y], file = fp)


# 4. get outliers

# y_contestual = (data.y >> 0 & 1).tolist() # contextual outliers
y_structural = (data.y >> 1 & 1).tolist() # structural outliers

# torch.set_printoptions(threshold = np.inf)
# print(y_contestual)
# print(y_structural)


# 5. build networkx

graph = nx.Graph()

graph.add_nodes_from(range(num_nodes))

weight = dict()

def get_hash(u, v):
    if u > v:
        u, v = v, u
    return u * num_nodes + v

for i in range(num_edges):
    u = data.edge_index[0][i].item()
    v = data.edge_index[1][i].item()
    if get_hash(u, v) not in weight:
        graph.add_edge(u, v)
        weight[get_hash(u, v)] = 1 
    else:
        weight[get_hash(u, v)] = weight[get_hash(u, v)] + 1

for u, v in graph.edges():
    graph[u][v]["weight"] = weight[get_hash(u, v)]

print("Build Network Finished.")


# 6. delete 0-degree nodes

delete_nodes = []

for i in range(num_nodes):
    if graph.degree(i, weight = "weight") == 0:
        delete_nodes.append(i)

graph.remove_nodes_from(delete_nodes)

graph = nx.convert_node_labels_to_integers(graph)

num_nodes = len(graph.nodes())

num_edges = len(graph.edges())

for i in reversed(delete_nodes):
    del y_structural[i]
    # del y_contestual[i]

print("graph after deleting 0-degree nodes : ", graph)

print("Deleting 0-degree Nodes Finished.")


# 7. run DynGDim

relative_dimensions, peak_times = run_all_sources(graph, times, n_workers=n_workers)

local_dimensions = []

for time_horizon in times:
    relative_dimensions_reduced = relative_dimensions.copy()
    relative_dimensions_reduced[peak_times > time_horizon] = np.nan
    local_dimensions.append(np.nanmean(relative_dimensions_reduced, axis=1))

local_dimensions = np.array(local_dimensions)

global_dimension = compute_global_dimension(local_dimensions)

print("Calculation DynGDim Finished.")


# 8. calculate scores

best_time_index = -1
best_time_horizon = -1

for time_index, time_horizon in enumerate(times):
    num_nan = 0
    for index, local_dimension in enumerate(local_dimensions[time_index]):
        if np.isnan(local_dimension):
            local_dimensions[time_index][index] = 0
            num_nan = num_nan + 1
    if num_nan < num_nodes / 10 and time_horizon > 1 and best_time_index == -1:
        best_time_index = time_index
        best_time_horizon = time_horizon

score = []

# score_softmax = []

for time_index, time_horizon in enumerate(times):
    y_pred = local_dimensions[time_index]
    # y_pred_softmax = torch.softmax(torch.Tensor(y_pred), dim = 0)
    y_pred = y_pred - np.nanmin(y_pred)
    if np.nanmax(y_pred) < 1e-6:
        continue
    y_pred = y_pred / np.nanmax(y_pred)
    score.append(roc_auc_score(y_structural, y_pred))
    # score_softmax.append(roc_auc_score(y_structural, y_pred_softmax))


print("-----------------------scores-----------------------")

print("roc_auc_score:", score)
print("roc_auc_score(max):",max(score))

# print("roc_auc_score(softmax):")
# print(score_softmax)


# 9. plotting


# 9.1. plot scatter diagram

plt.figure()

x1 = []
y1 = []
x2 = []
y2 = []

for time_index, time_horizon in enumerate(times):
    for node in range(num_nodes):
        if y_structural[node]:
            x1.append(time_horizon + 0.1)
            y1.append(local_dimensions[time_index, node])
        else:
            x2.append(time_horizon)
            y2.append(local_dimensions[time_index, node])

plt.scatter(x2, y2, label="Normal Node", c = ["k"] * len(x2))
plt.scatter(x1, y1, label="Structural Outlier", c = ["r"] * len(x1))

plt.xlabel("Time Scale")
plt.ylabel("Local Dimension")

plt.legend(loc = "upper right")

plt.savefig("figs/" + dataset + "_scatter.pdf")

plt.show()

plt.close()


# 9.2. plot network diagram

if num_nodes >= 1000:
    sys.exit()

plt.figure()

vmin = np.nanmin(local_dimensions)
vmax = np.nanmax(local_dimensions)

pos = nx.spring_layout(graph)

node_size = local_dimensions[time_index, :] / np.max(local_dimensions[time_index, :]) * 20

cmap = plt.cm.coolwarm

node_order = np.argsort(node_size)

node_shape = ["o", "^"]

labels = ["Normal Node", "Structural Outlier"]

for node in node_order:
    nodes = nx.draw_networkx_nodes(
        graph,
        pos = pos,
        nodelist = [node],
        cmap = cmap,
        node_color = [local_dimensions[best_time_index, node]],
        vmin = vmin,
        vmax = vmax,
        node_shape = node_shape[y_structural[node]],
        label = labels[y_structural[node]]
    )
    labels[y_structural[node]] = None

plt.colorbar(nodes, label="Local Dimension")

weights = np.array([graph[i][j]["weight"] for i, j in graph.edges])
nx.draw_networkx_edges(graph, pos=pos, alpha=0.5, width=weights / np.max(weights))

plt.suptitle("Time Horizon {:.2e}".format(best_time_horizon), fontsize=14)

plt.legend(loc = "upper right")

plt.savefig("figs/" + dataset + "_network.pdf")

# plt.show()

plt.close()
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, LayerNorm, ReLU, Dropout
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer, GATConv, DenseGCNConv, GCNConv, GraphConv
from torch_geometric.data import Data, DataLoader

from sklearn.metrics import roc_auc_score
import scipy.sparse as sp

import warnings
warnings.filterwarnings("ignore")

import pandas as pd


df_features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
df_edges = pd.read_csv("data/elliptic_txs_edgelist.csv")
df_classes =  pd.read_csv("data/elliptic_txs_classes.csv")
df_classes['class'] = df_classes['class'].map({'unknown': 2, '1':1, '2':0})

# merging dataframes
df_merge = df_features.merge(df_classes, how='left', right_on="txId", left_on=0)
df_merge = df_merge.sort_values(0).reset_index(drop=True)
classified = df_merge.loc[df_merge['class'].loc[df_merge['class']!=2].index].drop('txId', axis=1)
unclassified = df_merge.loc[df_merge['class'].loc[df_merge['class']==2].index].drop('txId', axis=1)

# storing classified unclassified nodes seperatly for training and testing purpose
classified_edges = df_edges.loc[df_edges['txId1'].isin(classified[0]) & df_edges['txId2'].isin(classified[0])]
unclassifed_edges = df_edges.loc[df_edges['txId1'].isin(unclassified[0]) | df_edges['txId2'].isin(unclassified[0])]
del df_features, df_classes

# all nodes in data
nodes = df_merge[0].values
map_id = {j:i for i,j in enumerate(nodes)} # mapping nodes to indexes

edges = df_edges.copy()
edges.txId1 = edges.txId1.map(map_id)
edges.txId2 = edges.txId2.map(map_id)
edges = edges.astype(int)

edge_index = np.array(edges.values).T

# for undirected graph
# edge_index_ = np.array([edge_index[1,:], edge_index[0, :]])
# edge_index = np.concatenate((edge_index, edge_index_), axis=1)

# for directed graph
edge_index = torch.tensor(edge_index, dtype=torch.long).contiguous()
weights = torch.tensor([1]* edge_index.shape[1] , dtype=torch.double)

# maping txIds to corresponding indexes, to pass node features to the model
node_features = df_merge.drop(['txId'], axis=1).copy()
node_features[0] = node_features[0].map(map_id)
classified_idx = node_features['class'].loc[node_features['class']!=2].index
unclassified_idx = node_features['class'].loc[node_features['class']==2].index
# replace unkown class with 0, to avoid having 3 classes, this data/labels never used in training
node_features['class'] = node_features['class'].replace(2, 0) 

labels = node_features['class'].values
node_features = torch.tensor(np.array(node_features.drop([0, 'class', 1], axis=1).values, dtype=np.double), dtype=torch.double)

# converting data to PyGeometric graph data format
graph_pyg = Data(x=node_features, edge_index=edge_index, edge_attr=weights,
                               y=torch.tensor(labels, dtype=torch.double)) #, adj= torch.from_numpy(np.array(adj))

print(graph_pyg)
# print(graph_pyg.y.tolist())
print(graph_pyg.is_directed())

graph_pyg.x = graph_pyg.x.to(torch.float)
graph_pyg.edge_attr = graph_pyg.edge_attr.to(torch.float)


from dyngdim.anomaly_detection import *

from dyngdim.build_graph import *

times = np.logspace(0, 1.0, 20)

dataset = "elliptic"

graph_networkx = build_networkx_from_torch_geometric(graph_pyg)

graph = dyngdim(graph_networkx, graph_pyg.y, times, dataset)

# graph = dyngdim(y_structural=graph_pyg.y)

# graph.graph_anomaly_detection_dyngdim(n_workers=n_workers)

# graph.graph_anomaly_detection_centrality()

graph.graph_anomaly_detection_pygod(graph_pyg)
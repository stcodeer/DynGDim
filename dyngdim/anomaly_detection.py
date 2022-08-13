from zoneinfo import available_timezones
import numpy as np

import networkx as nx

from sklearn.metrics import roc_auc_score

from dyngdim.dyngdim import *

import matplotlib.pyplot as plt

import sys


class dyngdim:
    local_dimensions = []
    delete_nodes = []
    available_time_index = []
    score = []
 
 
    def __init__(self, graph, y_structural, times, dataset):
        self.graph = graph
        self.num_nodes = len(graph)
        self.y_structural = y_structural
        self.times = times
        self.dataset = dataset


    def delete_zero_degree_nodes(self):
        for i in range(self.num_nodes):
            if self.graph.degree(i, weight = "weight") == 0:
                self.delete_nodes.append(i)

        self.graph.remove_nodes_from(self.delete_nodes)
        
        for i in reversed(self.delete_nodes):
            del self.y_structural[i]
        
        print("graph after deleting 0-degree nodes : ", self.graph)
        
        self.graph = nx.convert_node_labels_to_integers(self.graph)
        
        self.num_nodes = len(self.graph)
            

    def get_local_dimensions(self, n_workers):
        relative_dimensions, peak_times = run_all_sources(self.graph, self.times, n_workers=n_workers)

        for time_horizon in self.times:
            relative_dimensions_reduced = relative_dimensions.copy()
            relative_dimensions_reduced[peak_times > time_horizon] = np.nan
            self.local_dimensions.append(np.nanmean(relative_dimensions_reduced, axis=1))

        self.local_dimensions = np.array(self.local_dimensions)

        print("Calculation DynGDim Finished.")


    def get_roc_auc_score(self, train_mask, test_mask):

        for time_index, time_horizon in enumerate(self.times):
            num_nan = 0
            for index, local_dimension in enumerate(self.local_dimensions[time_index]):
                if np.isnan(local_dimension):
                    self.local_dimensions[time_index][index] = 0
                    num_nan = num_nan + 1
        
        for time_index, time_horizon in enumerate(self.times):
            y_pred = self.local_dimensions[time_index]
            y_pred = y_pred - np.nanmin(y_pred)
            if np.nanmax(y_pred) < 1e-6:
                continue
            y_pred = y_pred / np.nanmax(y_pred)
            self.score.append(roc_auc_score(self.y_structural, y_pred))
            self.available_time_index.append(time_index)


    def plot_local_dimensions_and_outliers(self, display=False):
        plt.figure()

        x1 = []
        y1 = []
        x2 = []
        y2 = []

        for time_index, time_horizon in enumerate(self.times):
            for node in range(self.num_nodes):
                if self.y_structural[node]:
                    x1.append(time_horizon + 0.1)
                    y1.append(self.local_dimensions[time_index, node])
                else:
                    x2.append(time_horizon)
                    y2.append(self.local_dimensions[time_index, node])

        plt.scatter(x2, y2, label="Normal Node", c = ["k"] * len(x2))
        plt.scatter(x1, y1, label="Structural Outlier", c = ["r"] * len(x1))

        plt.xlabel("Time Scale")
        plt.ylabel("Local Dimension")

        plt.legend(loc = "upper right")

        plt.savefig("figs/" + self.dataset + "_scatter.pdf")

        if display:
            plt.show()

        plt.close()
        
        
    def plot_network_structure(self, nodes_bar=1000, display=False):
        
        if self.num_nodes >= nodes_bar:
            sys.exit()

        plt.figure()
        
        time_index = self.available_time_index[0]
        time_horizon = self.times[time_index]

        vmin = np.nanmin(self.local_dimensions)
        vmax = np.nanmax(self.local_dimensions)

        pos = nx.spring_layout(self.graph)

        node_size = self.local_dimensions[time_index, :] / np.max(self.local_dimensions[time_index, :]) * 20

        cmap = plt.cm.coolwarm

        node_order = np.argsort(node_size)

        node_shape = ["o", "^"]

        labels = ["Normal Node", "Structural Outlier"]

        for node in node_order:
            nodes = nx.draw_networkx_nodes(
                self.graph,
                pos = pos,
                nodelist = [node],
                cmap = cmap,
                node_color = [self.local_dimensions[time_index, node]],
                vmin = vmin,
                vmax = vmax,
                node_shape = node_shape[self.y_structural[node]],
                label = labels[self.y_structural[node]]
            )
            labels[self.y_structural[node]] = None

        plt.colorbar(nodes, label="Local Dimension")

        weights = np.array([self.graph[i][j]["weight"] for i, j in self.graph.edges])
        nx.draw_networkx_edges(self.graph, pos=pos, alpha=0.5, width=weights / np.max(weights))

        plt.suptitle("Time Horizon {:.2e}".format(time_horizon), fontsize=14)

        plt.legend(loc = "upper right")

        plt.savefig("figs/" + self.dataset + "_network.pdf")

        if display:
            plt.show()

        plt.close()
        
        
    def graph_anomaly_detection(self, train_mask, test_mask, n_workers):
        self.delete_zero_degree_nodes()
        
        self.get_local_dimensions(n_workers)
        
        self.get_roc_auc_score(train_mask, test_mask)
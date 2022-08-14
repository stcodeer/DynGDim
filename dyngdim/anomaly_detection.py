from zoneinfo import available_timezones
import numpy as np

import networkx as nx

from sklearn.metrics import roc_auc_score

from dyngdim.dyngdim import *

import matplotlib.pyplot as plt


class dyngdim:
    local_dimensions = []
    # delete_nodes = []
    available_time_index = []
    train_score = []
    test_score = []
 
 
    def __init__(self, graph, y_structural, times, dataset, is_semi_supervised):
        self.graph = graph
        self.num_nodes = len(graph)
        self.y_structural = y_structural
        self.times = times
        self.dataset = dataset
        self.is_semi_supervised = is_semi_supervised


    # def delete_zero_degree_nodes(self):
    #     for i in range(self.num_nodes):
    #         if self.graph.degree(i, weight = "weight") == 0:
    #             self.delete_nodes.append(i)

    #     self.graph.remove_nodes_from(self.delete_nodes)
        
    #     for i in reversed(self.delete_nodes):
    #         del self.y_structural[i]
        
    #     print("graph after deleting 0-degree nodes : ", self.graph)
        
    #     self.graph = nx.convert_node_labels_to_integers(self.graph)
        
    #     self.num_nodes = len(self.graph)
    
    
    def add_self_loops(self):
        for u in range(self.num_nodes):
            self.graph.add_edge(u, u)
            self.graph[u][u]["weight"] = 1
            

    def get_local_dimensions(self, n_workers):
        relative_dimensions, peak_times = run_all_sources(self.graph, self.times, n_workers=n_workers)

        for time_horizon in self.times:
            relative_dimensions_reduced = relative_dimensions.copy()
            relative_dimensions_reduced[peak_times > time_horizon] = np.nan
            self.local_dimensions.append(np.nanmean(relative_dimensions_reduced, axis=1))

        self.local_dimensions = np.array(self.local_dimensions)

        print("Calculation DynGDim Finished.")


    def get_roc_auc_score(self, train_mask, test_mask, display=True):

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
            
            if self.is_semi_supervised:
                y1 = [self.y_structural[i] for i in range(self.num_nodes) if train_mask[i]]
                y2 = [y_pred[i] for i in range(self.num_nodes) if train_mask[i]]
                self.train_score.append(roc_auc_score(y1, y2))
                
                y1 = [self.y_structural[i] for i in range(self.num_nodes) if test_mask[i]]
                y2 = [y_pred[i] for i in range(self.num_nodes) if test_mask[i]]
                self.test_score.append(roc_auc_score(y1, y2))
            else:
                self.test_score.append(roc_auc_score(self.y_structural, y_pred))
                
            
            self.available_time_index.append(time_index)
            
        if display:
            if self.is_semi_supervised:
                print("-----------------------train_scores-----------------------")

                print("roc_auc_score:", self.train_score)
                print("roc_auc_score(argmax):", np.argmax(self.train_score))
                print("roc_auc_score(max):", max(self.train_score))

                print("-----------------------test_scores-----------------------")

                print("roc_auc_score:", self.test_score)
                print("roc_auc_score(argmax):", np.argmax(self.test_score))
                print("roc_auc_score(max):", max(self.test_score))
            else:
                print("-----------------------scores-----------------------")

                print("roc_auc_score:", self.test_score)
                print("roc_auc_score(argmax):", np.argmax(self.test_score))
                print("roc_auc_score(max):", max(self.test_score))
                


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
        
        plt.suptitle("dataset: {dataset}", fontsize=14)

        plt.savefig("figs/" + self.dataset + "_local_dimensions_and_outliers.pdf")

        if display:
            plt.show()

        plt.close()
        
        
    def plot_network_structure(self, display=False):

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
        
        nx.draw_networkx_edges(self.graph, pos=pos, alpha=0.5, width = weights / np.max(weights))

        plt.suptitle("dataset: {dataset}", fontsize=14)

        plt.suptitle("Time Horizon {:.2e}".format(time_horizon), fontsize=14)

        plt.legend(loc = "upper right")

        plt.savefig("figs/" + self.dataset + "_network_structure.pdf")

        if display:
            plt.show()

        plt.close()
        
        
    def plot_roc_auc_score(self, display=False):
        num_scales = len(self.available_time_index)
        available_time_horizon = self.times[self.available_time_index]
        
        
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        if self.is_semi_supervised:
            axes.plot(available_time_horizon, self.train_score, label="Train Mask", linestyle='-', color='black', marker='.', linewidth=1.5)
            axes.plot(available_time_horizon, self.test_score, label="Test Mask", linestyle='-', color='red', marker='.', linewidth=1.5)
            plt.legend(loc = "upper right")
        else:
            axes.plot(available_time_horizon, self.test_score, linestyle='-', color='red', marker='.', linewidth=1.5)
        
        axes.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        axes.set_xticks([2, 4, 6, 8, 10])
        
        axes.grid(which='minor', c='lightgrey')
        
        axes.set_ylabel("ROC AUC Score")
        axes.set_xlabel("Time Horizon")
        
        plt.suptitle("dataset: {}".format(self.dataset), fontsize=14)
        
        plt.savefig("figs/" + self.dataset + "_roc_auc_score.pdf")
        
        if display:
            plt.show()
        
        plt.close()
        
        
    def graph_anomaly_detection(self, train_mask, test_mask, n_workers=1, display=True):
        # self.delete_zero_degree_nodes()
        
        self.add_self_loops()
        
        self.get_local_dimensions(n_workers)
        
        self.get_roc_auc_score(train_mask, test_mask)
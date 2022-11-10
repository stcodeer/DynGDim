import torch
import numpy as np

import networkx as nx

from sklearn.metrics import roc_auc_score

from dyngdim.dyngdim import *

from pygod.metrics import eval_roc_auc
from pygod.models import *

import matplotlib.pyplot as plt

import time

import torch


class dyngdim:
 
 
    def __init__(self, graph, y_structural, times, dataset, is_semi_supervised=False):
        self.graph = graph
        self.num_nodes = len(graph)
        self.y_structural = y_structural
        self.times = times
        self.dataset = dataset
        self.is_semi_supervised = is_semi_supervised
        self.local_dimensions = []
        # delete_nodes = []
        self.train_score = []
        self.test_score = []


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
            

    def get_local_dimensions(self, n_workers, directed=False):
        relative_dimensions, peak_times = run_all_sources(self.graph, self.times, n_workers=n_workers, directed=directed)
        
        self.local_dimensions = []
        
        for time_horizon in self.times:
            relative_dimensions_reduced = relative_dimensions.copy()
            relative_dimensions_reduced[peak_times > time_horizon] = np.nan
            self.local_dimensions.append(np.nanmean(relative_dimensions_reduced, axis=1))

        self.local_dimensions = np.array(self.local_dimensions)

        print("Calculation DynGDim Finished.")


    def get_roc_auc_score(self, train_mask=[], test_mask=[], display=True):
        self.train_score.clear()
        self.test_score.clear()
        
        for time_index, time_horizon in enumerate(self.times):
            num_nan = 0
            for index, local_dimension in enumerate(self.local_dimensions[time_index]):
                if np.isnan(local_dimension):
                    self.local_dimensions[time_index][index] = 0
                    num_nan = num_nan + 1
        
        for time_index, time_horizon in enumerate(self.times):
            y_pred = self.local_dimensions[time_index]
            
            if self.is_semi_supervised:
                y1 = [self.y_structural[i] for i in range(self.num_nodes) if train_mask[i]]
                y2 = [y_pred[i] for i in range(self.num_nodes) if train_mask[i]]
                self.train_score.append(roc_auc_score(y1, y2))
                
                y1 = [self.y_structural[i] for i in range(self.num_nodes) if test_mask[i]]
                y2 = [y_pred[i] for i in range(self.num_nodes) if test_mask[i]]
                self.test_score.append(roc_auc_score(y1, y2))
            else:
                self.test_score.append(roc_auc_score(self.y_structural, y_pred))
            
        if display:
            print("-----------------------DynGDim-----------------------")
            if self.is_semi_supervised:
                print("-----------------------train_scores-----------------------")

                print("roc_auc_score:", self.train_score)
                print("roc_auc_score(argmax):", np.argmax(self.train_score))
                print("roc_auc_score(max):", max(self.train_score))
                print("roc_auc_score(argmin):", np.argmin(self.train_score))
                print("roc_auc_score(min):", min(self.train_score))

                print("-----------------------test_scores-----------------------")

                print("roc_auc_score:", self.test_score)
                print("roc_auc_score(argmax):", np.argmax(self.test_score))
                print("roc_auc_score(max):", max(self.test_score))
                print("roc_auc_score(argmin):", np.argmin(self.test_score))
                print("roc_auc_score(min):", min(self.test_score))
            else:
                print("-----------------------scores-----------------------")

                print("roc_auc_score:", self.test_score)
                print("roc_auc_score(argmax):", np.argmax(self.test_score))
                print("roc_auc_score(max):", max(self.test_score))
                print("roc_auc_score(argmin):", np.argmin(self.test_score))
                print("roc_auc_score(min):", min(self.test_score))
                
            print("----------------------------------------------------")
                


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
        
        time_index = len(self.times) // 2
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
        
        
    def plot_roc_auc_score(self, y_preds, y_structural, name, display=True):
        self.test_score.clear()
        
        for time_index, time_horizon in enumerate(y_preds):
            y_pred = y_preds[time_index]
            self.test_score.append(roc_auc_score(y_structural, y_pred))
        
        if display:
            print("-----------------------" + name + "-----------------------")
            print("-----------------------scores-----------------------")

            print("roc_auc_score:", self.test_score)
            print("roc_auc_score(argmax):", np.argmax(self.test_score))
            print("roc_auc_score(max):", max(self.test_score))
            print("roc_auc_score(argmin):", np.argmin(self.test_score))
            print("roc_auc_score(min):", min(self.test_score))
            
            print("----------------------------------------------------")
        
        
    def plot_roc_auc_score_times(self, display=False):
        fig, axes = plt.subplots(1, 1, figsize=(8, 4))
        if self.is_semi_supervised:
            axes.plot(self.times, self.train_score, label="Train Mask", linestyle='-', color='black', marker='.', linewidth=1.5)
            axes.plot(self.times, self.test_score, label="Test Mask", linestyle='-', color='red', marker='.', linewidth=1.5)
            plt.legend(loc = "upper right")
        else:
            axes.plot(self.times, self.test_score, linestyle='-', color='red', marker='.', linewidth=1.5)
        
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
        
        
    def graph_anomaly_detection_dyngdim(self, train_mask=[], test_mask=[], n_workers=1, directed=False, display=True):
        # self.delete_zero_degree_nodes()
        
        self.add_self_loops()
        
        self.get_local_dimensions(n_workers, directed)
        
        self.get_roc_auc_score(train_mask, test_mask, display)
        
        
    def graph_anomaly_detection_centrality(self, display=True):
        self.local_dimensions = []
        
        def get_array_from_centrality(centrality):
            return np.array([centrality[i] for i in range(self.num_nodes)])
        
        # Degree Centrality
        deg_cen = get_array_from_centrality(nx.degree_centrality(self.graph))
        print("Calculating Degree Centrality Finished.")

        # Closeness Centrality (low speed)
        clo_cen = get_array_from_centrality(nx.closeness_centrality(self.graph))
        print("Calculating Closeness Centrality Finished.")

        # Eigenvector Centrality
        eig_cen = get_array_from_centrality(nx.eigenvector_centrality_numpy(self.graph))
        print("Calculating Eigenvector Centrality Finished.")

        # Katz Centrality
        kat_cen = get_array_from_centrality(nx.katz_centrality_numpy(self.graph))
        print("Calculating Katz Centrality Finished.")

        # Pagerank (low speed)
        pagerank = get_array_from_centrality(nx.pagerank_numpy(self.graph))
        print("Calculating Pagerank Finished.")

        centrality = np.array([deg_cen, clo_cen, eig_cen, kat_cen, pagerank])
        
        self.local_dimensions = centrality

        self.plot_roc_auc_score(centrality, self.y_structural, "Centrality", display)
            
            
    def graph_anomaly_detection_pygod(self, data, self_loop=True, delete_feature=False, delete_graph=False, display=True):
        self.local_dimensions = []

        # unsupervised models (low speed)
        models = [MLPAE, SCAN, Radar, ANOMALOUS, GCNAE, DOMINANT, DONE, AdONE, AnomalyDAE, GAAN, CONAD]
        
        # unsupervised models
        # models = [MLPAE, Radar, ANOMALOUS]
        
        num_nodes = data.y.shape[0]
        num_train_nodes =  num_nodes // 2
        num_test_nodes = num_nodes - num_train_nodes
        
        data.train_mask = torch.tensor([1] * num_train_nodes + [0] * num_test_nodes)
        data.test_mask = torch.tensor([0] * num_train_nodes + [1] * num_test_nodes)
        
        if self_loop:
            x = data.edge_index[0].tolist() + [i for i in range(num_nodes)]
            y = data.edge_index[1].tolist() + [i for i in range(num_nodes)]
            data.edge_index = torch.tensor([x, y])
            
        # deleting all features
        if delete_feature:
            data.x = data.x.zero_()

        # deleting graph structure
        if delete_graph:
            data.edge_index = torch.tensor([[i for i in range(num_nodes)], [i for i in range(num_nodes)]])

        for model_name in models:
            
            print("running " + model_name.__name__ + " model.")
            
            model = model_name()  # hyperparameters can be set here
            
            model.fit(data)  # data is a Pytorch Geometric data object

            outlier_scores = model.decision_function(data)
            
            # for i in range(len(outlier_scores)):
            #     if np.isnan(outlier_scores[i]):
            #         outlier_scores[i] = 0
                    
            self.local_dimensions.append(outlier_scores)
        
        self.plot_roc_auc_score(self.local_dimensions, self.y_structural, "PyGOD", display)
            
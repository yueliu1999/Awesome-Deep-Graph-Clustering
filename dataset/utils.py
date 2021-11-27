# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

import numpy as np
import torch


def numpy_to_torch(a, is_sparse=False):
    """
    numpy array to torch tensor
    :param a: the numpy array
    :param is_sparse: is sparse tensor or not
    :return: torch tensor
    """
    if is_sparse:
        a = torch.sparse.Tensor(a)
    else:
        a = torch.from_numpy(a)
    return a


def torch_to_numpy(t):
    """
    torch tensor to numpy array
    :param t: the torch tensor
    :return: numpy array
    """
    return t.numpy()


def load_graph_data(dataset_name, show_details=False):
    """
    load graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :return: the features, labels and adj
    """
    load_path = dataset_name+"/"+dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0]/2))
        print("category num:          ", max(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj


def load_data(dataset_name, show_details=False):
    """
    load non-graph data
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - category num
    - category distribution
    :return: the features and labels
    """
    load_path = dataset_name+"/"+dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("------details of dataset------")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("category num:   ", max(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label


def construct_graph(feat, k=5, metric="euclidean"):
    """
    construct the knn graph for a non-graph dataset
    :param feat: the input feature matrix
    :param k: hyper-parameter of knn
    :param metric: the metric of distance calculation
    - euclidean: euclidean distance
    - cosine: cosine distance
    - heat: heat kernel
    :return: the constructed graph
    """

    # euclidean distance, sqrt((x-y)^2)
    if metric == "euclidean" or metric == "heat":
        xy = np.matmul(feat, feat.transpose())
        xx = (feat * feat).sum(1).reshape(-1, 1)
        xx_yy = xx + xx.transpose()
        euclidean_distance = xx_yy - 2 * xy
        euclidean_distance[euclidean_distance < 1e-5] = 0
        distance_matrix = np.sqrt(euclidean_distance)

        # heat kernel, exp^{- euclidean^2/t}
        if metric == "heat":
            distance_matrix = - (distance_matrix ** 2) / 2
            distance_matrix = np.exp(distance_matrix)

    # cosine distance, 1 - cosine similarity
    if metric == "cosine":
        norm_feat = feat / np.sqrt(np.sum(feat ** 2, axis=1)).reshape(-1, 1)
        cosine_distance = 1 - np.matmul(norm_feat, norm_feat.transpose())
        cosine_distance[cosine_distance < 1e-5] = 0
        distance_matrix = cosine_distance

    # top k
    distance_matrix = numpy_to_torch(distance_matrix)
    top_k, index = torch.topk(distance_matrix, k)
    top_k_min = torch.min(top_k, dim=-1).values.unsqueeze(-1).repeat(1, distance_matrix.shape[-1])
    ones = torch.ones_like(distance_matrix)
    zeros = torch.zeros_like(distance_matrix)
    knn_graph = torch.where(torch.ge(distance_matrix, top_k_min), ones, zeros)

    return torch_to_numpy(knn_graph)


def norm_adj(adj, self_loop=True, symmetry=True):
    """
    normalize the adj matrix
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param symmetry: symmetry normalize or not
    :return: the normalized adj matrix
    """
    return None


if __name__ == '__main__':
    # graph dataset
    # graph_dataset = "dblp"
    # X, y, A = load_graph_data(graph_dataset, show_details=True)

    # non graph dataset
    non_graph_dataset = "hhar"
    X, y = load_data(non_graph_dataset, show_details=False)
    construct_graph(X, k=5)

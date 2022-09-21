# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

import torch
import random
import numpy as np
from .kmeans_gpu import kmeans
from sklearn.cluster import KMeans
from .evaluation import evaluation


def k_means(embedding, k, y_true, device="cpu"):
    """
    K-means algorithm
    :param embedding: embedding of clustering
    :param k: hyper-parameter in K-means
    :param y_true: ground truth
    :param device: device
    :returns acc, nmi, ari, f1, center:
    - acc
    - nmi
    - ari
    - f1
    - cluster centers
    """
    if device == "cpu":
        model = KMeans(n_clusters=k, n_init=20)
        cluster_id = model.fit_predict(embedding)
        center = model.cluster_centers_
    if device == "gpu":
        cluster_id, center = kmeans(X=torch.tensor(embedding), num_clusters=k, distance="euclidean", device="cuda")
        cluster_id = cluster_id.numpy()
    acc, nmi, ari, f1 = evaluation(y_true, cluster_id)
    return acc, nmi, ari, f1, center

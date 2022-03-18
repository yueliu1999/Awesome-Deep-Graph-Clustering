# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

import torch
import random
import numpy as np
from munkres import Munkres
from kmeans_gpu import kmeans
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score


def setup_seed(seed):
    """
    fix the random seed
    :param seed: the random seed
    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


def evaluation(y_true, y_pred):
    """
    evaluate the clustering performance
    :param y_true: ground truth
    :param y_pred: prediction
    :return:
    - accuracy
    - normalized mutual information
    - adjust rand index
    - f1 score
    """
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)

    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('error')
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')

    return acc, nmi, ari, f1


def k_means(embedding, k, y_true, device="cpu"):
    """
    K-means algorithm
    :param embedding: embedding of clustering
    :param k: hyper-parameter in K-means
    :param y_true: ground truth
    :param device: device
    :return:
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

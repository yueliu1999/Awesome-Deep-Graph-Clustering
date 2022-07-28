# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2022/7/28 14:07

import torch
import numpy as np


def diffusion_adj(adj, self_loop=True, mode="ppr", transport_rate=0.2):
    """
    graph diffusion
    :param adj: input adj matrix
    :param self_loop: if add the self loop or not
    :param mode: the mode of graph diffusion
    :param transport_rate: the transport rate
    - personalized page rank
    -
    :return diff_adj: the graph diffusion
    """
    # add the self_loop
    if self_loop:
        adj_tmp = adj + np.eye(adj.shape[0])
    else:
        adj_tmp = adj

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj_tmp.sum(0))
    d_inv = np.linalg.inv(d)
    sqrt_d_inv = np.sqrt(d_inv)

    # calculate norm adj
    norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj_tmp), sqrt_d_inv)

    # calculate graph diffusion
    if mode == "ppr":
        diff_adj = transport_rate * np.linalg.inv((np.eye(d.shape[0]) - (1 - transport_rate) * norm_adj))

    return diff_adj


def drop_edge(adj, drop_rate=0.2):
    """
    drop edges randomly
    :param adj: input adj matrix
    :param drop_rate: drop rate
    :return drop_adj: edge dropped adj matrix
    """

    return drop_adj


def add_edge(adj, add_rate=0.2):
    """
    add edges randomly
    :param adj: input adj matrix
    :param add_rate: drop rate
    :return add_adj: edge added adj matrix
    """

    return add_adj

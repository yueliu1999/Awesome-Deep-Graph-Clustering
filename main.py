# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

from utils import load_graph_data, load_data, construct_graph, normalize_adj
from clustering import setup_seed

if __name__ == '__main__':
    # fix the random seed
    setup_seed(0)

    # dataset_name = "dblp"
    # dataset_type = "graph"

    dataset_name = "hhar"
    dataset_type = "non_graph"

    if dataset_type == "graph":
        # graph dataset
        graph_dataset = "dblp"
        X, y, A = load_graph_data(graph_dataset, show_details=True)

    if dataset_type == "non_graph":
        # non graph dataset
        non_graph_dataset = "hhar"
        X, y = load_data(non_graph_dataset, show_details=True)
        A = construct_graph(X, k=5)

    # normalize the adj
    norm_A = normalize_adj(A, self_loop=True, symmetry=True)

# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

from visualization import t_sne, similarity_plot
from clustering import setup_seed
from utils import load_graph_data, load_data, construct_graph, normalize_adj, diffusion_adj

if __name__ == '__main__':
    # fix the random seed
    setup_seed(0)

    dataset_name = "dblp"
    dataset_type = "graph"

    # dataset_name = "hhar"
    # dataset_type = "non_graph"

    # load data
    if dataset_type == "graph":
        X, y, A = load_graph_data(dataset_name, show_details=True)

    if dataset_type == "non_graph":
        X, y = load_data(dataset_name, show_details=True)
        A = construct_graph(X, k=5)

    # normalize the adj
    norm_A = normalize_adj(A, self_loop=True, symmetry=True)

    # # graph diffusion
    # diff_A = diffusion_adj(A)

    # # t-SNE
    # t_sne(X, y)

    # similarity plot
    # similarity_plot(embedding=X, label=y, sample_num=1000, show_fig=True)

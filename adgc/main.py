# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

from clustering import setup_seed, k_means
from adgc.utils.visualization import t_sne, similarity_plot
from adgc.utils.augmentation import diffusion_adj, add_edge, drop_edge, mask_feat
from utils import load_graph_data, load_data, construct_graph, normalize_adj

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
    # norm_A = normalize_adj(A, self_loop=True, symmetry=True)

    # augmentations on graphs
    # 1. graph diffusion
    # diff_A = diffusion_adj(A)
    # 2. drop edges randomly
    # drop_A = drop_edge(A, 0.2)
    # 3. add edges randomly
    # add_A = add_edge(A, 0.2)
    # 4. mask feature randomly
    masked_X = mask_feat(X, 0.2)

    # # t-SNE
    # t_sne(X, y)

    # similarity plot
    # similarity_plot(embedding=X, label=y, sample_num=1000, show_fig=True)

    # clustering, k-means
    # acc, nmi, ari, f1, center = k_means(embedding=X, k=max(y), y_true=y, device="gpu")

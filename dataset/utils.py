# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:10
# @Function: load_data
import numpy as np


def load_data(dataset_name, show_details=False):
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
        print("------details of dataset------")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("edge num:       ", np.nonzero(adj)[0].shape)
        print("category num:   ", max(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj


if __name__ == '__main__':
    dataset = "acm"
    X, y, A = load_data(dataset, show_details=True)

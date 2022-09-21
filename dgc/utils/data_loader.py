# -*- coding: utf-8 -*-
# @Author  : Yue Liu
# @Email   : yueliu19990731@163.com
# @Time    : 2021/11/25 11:11

import os
import sys
import torch
import logging
import numpy as np


def load_graph_data(root_path=".", dataset_name="dblp", show_details=False):
    """
    load graph data
    :param root_path: the root path
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - adj shape
    - edge num
    - category num
    - category distribution
    :returns feat, label, adj: the features, labels and adj
    """
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=logging.INFO,
        stream=sys.stdout)
    root_path = root_path + "dataset/"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    dataset_path = root_path + dataset_name
    if not os.path.exists(dataset_path):
        # down load
        url = "https://drive.google.com/file/d/1_LesghFTQ02vKOBUfDP8fmDF1JP3MPrJ/view?usp=sharing"
        logging.info("Downloading " + dataset_name + " dataset from: " + url)
    else:
        logging.info("Loading " + dataset_name + " dataset from local")
    load_path = root_path + dataset_name + "/" + dataset_name
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
        print("edge num:   ", int(adj.sum() / 2))
        print("category num:          ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label, adj


def load_data(root_path="./", dataset_name="USPS", show_details=False):
    """
    load non-graph data
    :param root_path: the root path
    :param dataset_name: the name of the dataset
    :param show_details: if show the details of dataset
    - dataset name
    - features' shape
    - labels' shape
    - category num
    - category distribution
    :returns feat, label: the features and labels
    """
    root_path = root_path + "dataset/"
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    dataset_path = root_path + dataset_name
    if not os.path.exists(dataset_path):
        # down load
        pass
    load_path = root_path + dataset_name + "/" + dataset_name
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)

    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("------details of dataset------")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("category num:   ", max(label)-min(label)+1)
        print("category distribution: ")
        for i in range(max(label)+1):
            print("label", i, end=":")
            print(len(label[np.where(label == i)]))
        print("++++++++++++++++++++++++++++++")
    return feat, label

import torch
from torch.utils.data import random_split
import pickle
import numpy as np
import networkx as nx

import sys
sys.path.append("../")
from configs import gemini_config as config


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def read_cfg():
    all_function_dict = {}
    for a in config.arch:
        for v in config.version:
            for c in config.compiler:
                for o in config.optimizer:
                    filename = "_".join([v, a, c, o, "openssl"])
                    filepath = config.dir_name + filename+".cfg"
                    with open(filepath, "r") as f:
                        picklefile = pickle.load(StrToBytes(f))
                    for func in picklefile.raw_graph_list:
                        if len(func.g) < config.min_nodes_threshold:
                            continue
                        if all_function_dict.get(func.funcname) == None:
                            all_function_dict[func.funcname] = []
                        all_function_dict[func.funcname].append(func.g)
    new_all_function_dict = {k: v for k, v in all_function_dict.items() if len(v) >= 2}
    print(f'function num: {len(new_all_function_dict)}')
    return new_all_function_dict


def dataset_split(all_function_dict):
    all_func_list = list(all_function_dict.items())
    train_size = int(len(all_func_list) * 0.8)
    test_size = int(len(all_func_list) * 0.1)
    valid_size = len(all_func_list) - train_size - test_size
    train_func, test_func, valid_func = random_split(all_func_list, [train_size, test_size, valid_size])
    for type, dataset in zip(["train", "test", "valid"], [train_func, test_func, valid_func]):
        with open(config.dataset_dir + type, "wb") as f:
            pickle.dump(dict(dataset), f)
    print(
        f"train dataset's num={len(train_func)} , valid dataset's num={len(valid_func)} , test dataset's num ={len(test_func)}"
    )

def adjmat(gr):
    return nx.adjacency_matrix(gr).toarray().astype('float32')


def zero_padded_adjmat(graph, size):
    unpadded = adjmat(graph)
    padded = np.zeros((size, size))
    if len(graph) > size:
        padded = unpadded[0:size, 0:size]
    else:
        padded[0:unpadded.shape[0], 0:unpadded.shape[1]] = unpadded
    return padded


def feature_vector(graph, size):
    feature_mat = np.zeros((size, config.feature_size))
    for _node in graph.nodes:
        if _node == size:
            break
        feature = np.zeros((1, config.feature_size))
        vector = graph.nodes[_node]['v']
        num_const = vector[0]
        if len(num_const) == 1:
            feature[0, 0] = num_const[0]
        elif len(num_const) >= 2:
            feature[0, 0:2] = np.sort(num_const)[::-1][:2]
        feature[0, 2] = len(vector[1])
        feature[0, 3:] = vector[2:]
        feature_mat[_node, :] = feature
    return feature_mat

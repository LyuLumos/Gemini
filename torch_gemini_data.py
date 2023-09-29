import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import numpy as np
import networkx as nx
import config
import random


class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj

    def read(self, size):
        return self.fileobj.read(size).encode()

    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()


def read_cfg():
    all_function_dict = {}
    counts = []
    for a in config.arch:
        count = 0
        for v in config.version:
            for c in config.compiler:
                for o in config.optimizer:
                    filename = "_".join([v, a, c, o, "openssl"])
                    filepath = config.dir_name + filename+".cfg"
                    with open(filepath, "r") as f:
                        picklefile = pickle.load(StrToBytes(f))
                    for func in picklefile.raw_graph_list:
                        print(func.funcname)
                        print(func.g)
                        break
                        if len(func.g) < config.min_nodes_threshold:
                            continue
                        if all_function_dict.get(func.funcname) == None:
                            all_function_dict[func.funcname] = []
                        all_function_dict[func.funcname].append(func.g)
                        count += 1
        counts.append(count)
    print("for three arch:", counts)

    return all_function_dict


def dataset_split(all_function_dict):
    all_func_list = list(all_function_dict.items())
    train_size = int(len(all_func_list) * 0.8)
    test_size = int(len(all_func_list) * 0.1)
    valid_size = len(all_func_list) - train_size - test_size
    train_func, test_func, valid_func = random_split(all_func_list, [train_size, test_size, valid_size])
    print(
        f"train dataset's num ={len(train_func)} ,valid dataset's num={len(valid_func)} , test dataset's num ={len(test_func)}"
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
    feature_mat = np.zeros((size, 9))
    for _node in graph.nodes:
        if _node == size:
            break
        feature = np.zeros((1, 9))
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


class GeminiDataset(Dataset):
    def __init__(self, type):
        assert type in ["train", "test", "valid"], "dataset type error!"
        filepath = config.Gemini_dataset_dir + type
        self.func_dict = torch.load(filepath)
        self.funcname_list = list(self.func_dict.keys())

    def __len__(self):
        return len(self.funcname_list)

    def __getitem__(self, idx):
        funcname = self.funcname_list[idx]
        func_list = self.func_dict[funcname]
        if len(func_list) < 2:
            print(f'funcname: {funcname} has only one graph')
            return None
        for i, g in enumerate(func_list):
            g_adjmat = zero_padded_adjmat(g, config.max_nodes)
            g_featmat = feature_vector(g, config.max_nodes)
            for j in random.randint(0, 1):
                if j == 0:
                    g1_index = np.random.randint(
                        low=0, high=len(func_list))
                    while g1_index == i:
                        g1_index = np.random.randint(
                            low=0, high=len(func_list))
                    g1 = func_list[g1_index]
                    g1_adjmat = zero_padded_adjmat(g1, config.max_nodes)
                    g1_featmat = feature_vector(g1, config.max_nodes)
                    return g_adjmat, g_featmat, g1_adjmat, g1_featmat, 1
                else:
                    index = np.random.randint(low=0, high=len(self.funcname_list))
                    while self.funcname_list[index] == funcname:
                        index = np.random.randint(low=0, high=len(self.funcname_list))
                    g2_index = np.random.randint(
                        low=0, high=len(self.func_dict[self.funcname_list[index]]))
                    g2 = self.func_dict[self.funcname_list[index]][g2_index]
                    g2_adjmat = zero_padded_adjmat(g2, config.max_nodes)
                    g2_featmat = feature_vector(g2, config.max_nodes)
                    return g_adjmat, g_featmat, g2_adjmat, g2_featmat, -1


def dataloader_generate():
    train_dataset = GeminiDataset("train")
    test_dataset = GeminiDataset("test")
    valid_dataset = GeminiDataset("valid")
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.mini_batch, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.mini_batch, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config.mini_batch, shuffle=True)
    print(f'train dataset size: {len(train_dataset)}, test dataset size: {len(test_dataset)}, valid dataset size: {len(valid_dataset)}')
    print(f'train dataloader size: {len(train_dataloader)}, test dataloader size: {len(test_dataloader)}, valid dataloader size: {len(valid_dataloader)}')
    return train_dataloader, test_dataloader, valid_dataloader

if __name__ == '__main__':
    all_func_dict = read_cfg()
    dataset_split(all_func_dict)

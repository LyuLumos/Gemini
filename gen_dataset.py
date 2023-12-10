import os
import numpy as np
import pickle
from torch.utils.data import Dataset
from gensim.models import Word2Vec
import torch.nn as nn
import torch
from data_process import read_pkl, asmlist2tokens


def gen_pairs(x86_corpus, x64_corpus, save_path):
    x86_x64_pairs = []
    for x86_func_name in x86_corpus.keys():
        if x86_func_name in x64_corpus.keys():
            x86_data = asmlist2tokens(x86_corpus[x86_func_name])
            x64_data = asmlist2tokens(x64_corpus[x86_func_name])
            x86_x64_pairs.append((x86_func_name, x86_data, x86_func_name, x64_data, 1))

            random_x64_func_name = np.random.choice(list(x64_corpus.keys()))
            while random_x64_func_name == x86_func_name:
                random_x64_func_name = np.random.choice(list(x64_corpus.keys()))
            x64_data = asmlist2tokens(x64_corpus[random_x64_func_name])
            x86_x64_pairs.append((x86_func_name, x86_data, random_x64_func_name, x64_data, 0))
    print(f'x86_x64_pairs size: {len(x86_x64_pairs)}')

    if not os.path.exists(save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(x86_x64_pairs, f)


class BCSDDataset(Dataset):
    def __init__(self, x86_x64_pairs):
        self.x86_x64_pairs = x86_x64_pairs

    def __len__(self):
        return len(self.x86_x64_pairs)

    def __getitem__(self, index):
        # print(f'x86_x64_pairs: {self.x86_x64_pairs[index]}')
        x86_func_name, x86_data, x64_func_name, x64_data, label = self.x86_x64_pairs[index]
        return x86_func_name, x86_data, x64_func_name, x64_data, label


def collate_fn(batch):
    word2vec = Word2Vec.load('instruct2vec.model')
    token2id = word2vec.wv.key_to_index.copy()
    id2token = word2vec.wv.index_to_key.copy()

    x86_func_names, x86_datas, x64_func_names, x64_datas, labels = zip(*batch)
    # if unknown drop it !!! This is not a good way, i will fix it when possible.
    x86_data_ids = [[token2id[token] for token in x86_data if token in token2id] for x86_data in x86_datas]
    x64_data_ids = [[token2id[token] for token in x64_data if token in token2id] for x64_data in x64_datas]

    word2vec_embedding = nn.Embedding.from_pretrained(torch.from_numpy(word2vec.wv.vectors))
    x86_datas = [word2vec_embedding(torch.LongTensor(x86_data_id)) for x86_data_id in x86_data_ids]
    x64_datas = [word2vec_embedding(torch.LongTensor(x64_data_id)) for x64_data_id in x64_data_ids]
    
    max_len = 512
    pad_x86_datas = torch.zeros(len(x86_datas), max_len, x86_datas[0].shape[1])
    pad_x64_datas = torch.zeros(len(x64_datas), max_len, x64_datas[0].shape[1])
    for i, (x86_data, x64_data) in enumerate(zip(x86_datas, x64_datas)):
        if x86_data.shape[0] > max_len:
            pad_x86_datas[i] = x86_data[:max_len]
        else:
            pad_x86_datas[i, :x86_data.shape[0]] = x86_data

        if x64_data.shape[0] > max_len:
            pad_x64_datas[i] = x64_data[:max_len]
        else:
            pad_x64_datas[i, :x64_data.shape[0]] = x64_data

    labels = torch.LongTensor(labels)
    # x86_datas: [batch_size, max_len, emb_dim]
    # x64_datas: [batch_size, max_len, emb_dim]
    # labels: [batch_size]
    # import sys; sys.exit()
    return x86_func_names, pad_x86_datas, x64_func_names, pad_x64_datas, labels


def get_maxlen_and_minlen():
    dataset, _ = read_pkl('x86_x64_pairs.pkl')
    dataLoaders = get_dataloader(dataset, batch_size=4)
    maxlen = 0
    minlen = 100000
    for _, x86_datas, _, x64_datas, _ in dataLoaders:
        for x86_data in x86_datas:
            maxlen = max(maxlen, x86_data.shape[1])
            minlen = min(minlen, x86_data.shape[1])
        for x64_data in x64_datas:
            maxlen = max(maxlen, x64_data.shape[1])
            minlen = min(minlen, x64_data.shape[1])
    print(f'maxlen: {maxlen}, minlen: {minlen}')
    # maxlen: 18693, minlen: 14


if __name__ == '__main__':
    
    from train import get_dataloader
    # x86_corpus, x86_small_corpus = read_pkl('x86_corpus.pkl')
    # x64_corpus, x64_small_corpus = read_pkl('x64_corpus.pkl')
    # print(f'x86 corpus size: {len(x86_corpus)}')
    # print(f'x64 corpus size: {len(x64_corpus)}')
    # gen_pairs(x86_corpus, x64_corpus, 'x86_x64_pairs.pkl')
    dataset, _ = read_pkl('x86_x64_pairs.pkl')
    dataLoaders = get_dataloader(dataset, batch_size=4)
    print(next(iter(dataLoaders)))
    x86_func_names, x86_datas, x64_func_names, x64_datas, labels = next(iter(dataLoaders))
    # print shape
    print(f'x86_datas shape: {x86_datas.shape}')
    print(f'x64_datas shape: {x64_datas.shape}')


import os

from data_process import read_pkl, asmlist2tokens
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import torch.nn as nn

import torch

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
    
    x86_datas = nn.utils.rnn.pad_sequence(x86_datas, batch_first=True)
    x64_datas = nn.utils.rnn.pad_sequence(x64_datas, batch_first=True)

    labels = torch.LongTensor(labels)
    # x86_datas: [batch_size, seq_len, emb_dim]
    # x64_datas: [batch_size, seq_len, emb_dim]
    # labels: [batch_size]
    return x86_func_names, x86_datas, x64_func_names, x64_datas, labels


def get_dataloader(x86_x64_pairs, batch_size=32, shuffle=True):
    dataset = BCSDDataset(x86_x64_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
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


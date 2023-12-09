# import torch
# import torch.nn as nn
from gensim.models import Word2Vec
import pickle
import os


def instruct2vec():
    if os.path.exists('instruct2vec.model'):
        print('[INFO] instruct2vec.model already exists.')
        return
    with open('tokens_list.pkl', 'rb') as f:
        tokens_list = pickle.load(f)
    print(f'tokens list: {tokens_list}')
    sentences = [[token] for token in tokens_list]
    model = Word2Vec(sentences, vector_size=10, window=5, min_count=1, workers=4, sg=1)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('instruct2vec.model')


def test_instruct2vec():
    model = Word2Vec.load('instruct2vec.model')
    vector = model.wv['sahf']
    print(f'[EXAMPLE] sahf: {vector}, shape: {vector.shape}')


if __name__ == '__main__':
    instruct2vec()
    test_instruct2vec()
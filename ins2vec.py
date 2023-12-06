# import torch
# import torch.nn as nn
from gensim.models import Word2Vec
import pickle
import os


# class Instruction2Vec(nn.Module):
#     def __init__(self, word2vec_model_path, device, token_dim=10):
#         super(Instruction2Vec, self).__init__()

#         word2vec = Word2Vec.load(word2vec_model_path)
#         self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(word2vec.wv.vectors))
#         self.emb_dim = word2vec.wv.vector_size
#         self.token2id = word2vec.wv.key_to_index.copy()
#         self.id2token = word2vec.wv.index_to_key.copy()
#         self.device = device
#         del word2vec
    
#     def _to_tensors(self, keylist):
#         idlist = [self.token2id[k] for k in keylist]
#         return self.embedding(torch.LongTensor(idlist).to(self.device))
    
#     def instruction2vec(self, instruction):
#         pass

#     def forward(self, instructions):
#         embs = [self.instruction2vec(i) for i in instructions]
#         return torch.stack(embs, dim=0)

from gensim.models import Word2Vec


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


def inference():
    model = Word2Vec.load('instruct2vec.model')
    



if __name__ == '__main__':
    instruct2vec()
    test_instruct2vec()
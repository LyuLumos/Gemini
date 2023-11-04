import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import math
import numpy as np

import sys
sys.path.append("../")
from configs import gemini_config as config


class EmbeddingLayer(nn.Module):
    def __init__(self):
        super(EmbeddingLayer, self).__init__()
        self.P1 = nn.Parameter(torch.Tensor(
            config.embedding_size, config.embedding_size))
        self.P2 = nn.Parameter(torch.Tensor(
            config.embedding_size, config.embedding_size))
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.P1, a=math.sqrt(5))
        init.kaiming_uniform_(self.P2, a=math.sqrt(5))

    def forward(self, x):
        curr_embedding = torch.einsum('ik,akj->aij', self.P1, x)
        curr_embedding = self.relu(curr_embedding)
        curr_embedding = torch.einsum('ik,akj->aij', self.P2, curr_embedding)
        return curr_embedding


class GraphEmbedding(nn.Module):
    def __init__(self):
        super(GraphEmbedding, self).__init__()
        self.W1 = nn.Parameter(torch.Tensor(
            config.embedding_size, config.feature_size))
        self.W2 = nn.Parameter(torch.Tensor(
            config.embedding_size, config.embedding_size))
        self.init_embedding = nn.Parameter(torch.Tensor(
            config.max_nodes, config.embedding_size))
        self.embed_layer = EmbeddingLayer()
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        init.normal_(self.init_embedding)

    def forward(self, adjmat, feature_mat):
        adjmat, feature_mat = adjmat.to(torch.float32), feature_mat.to(torch.float32)
        feature_mat = torch.einsum('aij->aji', feature_mat)
        prev_embedding = torch.einsum('aik,kj->aij', adjmat, self.init_embedding)
        prev_embedding = torch.einsum('aij->aji', prev_embedding)
        for _ in range(config.T):
            neighbor_embedding = self.embed_layer(prev_embedding)
            term = torch.einsum('ik,akj->aij', self.W1, feature_mat)
            curr_embedding = torch.tanh(term + neighbor_embedding)
            prev_embedding = torch.einsum('aij->aji', curr_embedding)
            prev_embedding = torch.einsum('aik,akj->aij', adjmat, prev_embedding)
            prev_embedding = torch.einsum('aij->aji', prev_embedding)
        graph_embedding = torch.sum(curr_embedding, axis=2)
        graph_embedding = torch.einsum('ij->ji', graph_embedding)
        graph_embedding = torch.matmul(self.W2, graph_embedding) # [embedding_size, batch_size]
        return graph_embedding


class Gemini(nn.Module):
    def __init__(self):
        super(Gemini, self).__init__()
        self.graph_embed_layer = GraphEmbedding()
        self.cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, g1_adjmat, g1_feature_mat, g2_adjmat, g2_feature_mat):
        g1_embedding = self.graph_embed_layer(g1_adjmat, g1_feature_mat)
        g2_embedding = self.graph_embed_layer(g2_adjmat, g2_feature_mat)
        sim_score = self.cos_sim(g1_embedding, g2_embedding)
        sim_score = (sim_score + 1) / 2
        return sim_score, g1_embedding, g2_embedding

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import config
from torch_gemini_data import dataloader_generate

class EmbeddingLayer(nn.Module):
    def __init__(self):
        super(EmbeddingLayer, self).__init__()
        self.theta = nn.Parameter(torch.randn(config.embedding_size, config.embedding_size))
        self.theta1 = nn.Parameter(torch.randn(config.embedding_size, config.embedding_size))
        self.relu = nn.ReLU()

    def forward(self, x):
        curr_embedding = torch.einsum('ik,akj->aij', self.theta, x)
        curr_embedding = self.relu(curr_embedding)
        curr_embedding = torch.einsum('ik,akj->aij', self.theta1, curr_embedding)
        return curr_embedding


def compute_graph_embedding(adjmat, feature_mat, W1, W2, embed_layer):
    adjmat, feature_mat = adjmat.to(torch.float32), feature_mat.to(torch.float32)
    feature_mat = torch.einsum('aij->aji', feature_mat)
    init_embedding = torch.zeros(adjmat.shape[1], config.embedding_size).to(device)
    # print(adjmat.dtype, init_embedding.dtype, feature_mat.dtype, W1.dtype)
    prev_embedding = torch.einsum('aik,kj->aij', adjmat, init_embedding)
    prev_embedding = torch.einsum('aij->aji', prev_embedding)
    for _ in range(config.T):
        neighbor_embedding = embed_layer(prev_embedding)
        term = torch.einsum('ik,akj->aij', W1, feature_mat)
        curr_embedding = torch.tanh(term + neighbor_embedding)
        prev_embedding = torch.einsum('aij->aji', curr_embedding)
        prev_embedding = torch.einsum('aik,akj->aij', adjmat, prev_embedding)
        prev_embedding = torch.einsum('aij->aji', prev_embedding)
    graph_embedding = torch.sum(curr_embedding, axis=2)
    graph_embedding = torch.einsum('ij->ji', graph_embedding)
    graph_embedding = torch.matmul(W2, graph_embedding)
    return graph_embedding


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.embed_layer = EmbeddingLayer()
        self.W1 = nn.Parameter(torch.randn(config.embedding_size, config.Gemini_feature_size))
        self.W2 = nn.Parameter(torch.randn(config.embedding_size, config.embedding_size))
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, g1_adjmat, g1_feature_mat, g2_adjmat, g2_feature_mat):
        g1_embedding = compute_graph_embedding(g1_adjmat, g1_feature_mat, self.W1, self.W2, self.embed_layer)
        g2_embedding = compute_graph_embedding(g2_adjmat, g2_feature_mat, self.W1, self.W2, self.embed_layer)
        sim_score = self.cos(g1_embedding, g2_embedding)
        return sim_score, g1_embedding, g2_embedding

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dataloader, test_dataloader, valid_dataloader = dataloader_generate()

def train():
    model = MyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    model.train()
  
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        for g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y in train_dataloader:
            g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y = g1_adjmat.to(device), g1_featmat.to(device), g2_adjmat.to(device), g2_featmat.to(device), y.to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(g1_adjmat, g1_featmat, g2_adjmat, g2_featmat)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            predicted = [1 if i > 0 else -1 for i in outputs.data]
            predicted = torch.tensor(predicted).to(device)
            
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%' 
              %(epoch+1, config.epochs, epoch_loss, 100*correct/total))
    torch.save(model.state_dict(), config.Gemini_model_save_path)


if __name__ == "__main__":
    train()

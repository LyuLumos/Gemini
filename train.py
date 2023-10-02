import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import math
import numpy as np
import config
from data_processing import dataloader_generate
from sklearn.metrics import roc_auc_score


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


def compute_graph_embedding(adjmat, feature_mat, init_embedding, W1, W2, embed_layer):
    adjmat, feature_mat = adjmat.to(torch.float32), feature_mat.to(torch.float32)
    feature_mat = torch.einsum('aij->aji', feature_mat)
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


class Gemini(nn.Module):
    def __init__(self):
        super(Gemini, self).__init__()
        self.embed_layer = EmbeddingLayer()
        self.W1 = nn.Parameter(torch.Tensor(
            config.embedding_size, config.Gemini_feature_size))
        self.W2 = nn.Parameter(torch.Tensor(
            config.embedding_size, config.embedding_size))
        self.init_embedding = nn.Parameter(torch.Tensor(
            config.max_nodes, config.embedding_size))

        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        init.normal_(self.init_embedding)

    def forward(self, g1_adjmat, g1_feature_mat, g2_adjmat, g2_feature_mat):
        g1_embedding = compute_graph_embedding(
            g1_adjmat, g1_feature_mat, self.init_embedding, self.W1, self.W2, self.embed_layer)
        g2_embedding = compute_graph_embedding(
            g2_adjmat, g2_feature_mat, self.init_embedding, self.W1, self.W2, self.embed_layer)
        sim_score = self.cos(g1_embedding, g2_embedding)
        sim_score = (sim_score + 1) / 2
        return sim_score, g1_embedding, g2_embedding





def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataloader, test_dataloader, valid_dataloader = dataloader_generate()
    print(f'Using device: {device}')
    model = Gemini().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    # from utils import EarlyStopper
    # early_stopper = EarlyStopper(patience=2, min_delta=0)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        preds, gts, probs = [], [], []
        for g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y in train_dataloader:
            g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y = g1_adjmat.to(device), g1_featmat.to(
                device), g2_adjmat.to(device), g2_featmat.to(device), y.to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs, _, _ = model(g1_adjmat, g1_featmat, g2_adjmat, g2_featmat)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            probs.extend(outputs.cpu().data)
            gts.extend(y.cpu().data)

            predicted = [1 if i > 0.5 else 0 for i in outputs.data]
            preds.extend(predicted)

        epoch_loss /= len(train_dataloader)
        accuracy = (np.array(preds) == np.array(gts)).sum() / len(gts)
        auc = roc_auc_score(np.array(gts), np.array(probs))
        print("[Train] Epoch: %d, Loss: %f, Accuracy: %f, AUC: %f" %
              (epoch, epoch_loss, accuracy, auc))

        # valid
        model.eval()
        epoch_loss = 0.0
        preds, gts, probs = [], [], []
        with torch.no_grad():
            for g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y in valid_dataloader:
                g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y = g1_adjmat.to(device), g1_featmat.to(
                    device), g2_adjmat.to(device), g2_featmat.to(device), y.to(torch.float32).to(device)
                outputs, _, _ = model(g1_adjmat, g1_featmat, g2_adjmat, g2_featmat)
                loss = criterion(outputs, y)
                epoch_loss += loss.item()
                probs.extend(outputs.cpu().data)
                gts.extend(y.cpu().data)

                predicted = [1 if i > 0.5 else 0 for i in outputs.data]
                preds.extend(predicted)

            epoch_loss /= len(valid_dataloader)
            accuracy = (np.array(preds) == np.array(gts)).sum() / len(gts)
            auc = roc_auc_score(np.array(gts), np.array(probs))
            print("[Valid] Epoch: %d, Loss: %f, Accuracy: %f, AUC: %f" %
                (epoch, epoch_loss, accuracy, auc))
        

        # test
        model.eval()
        epoch_loss = 0.0
        preds, gts, probs = [], [], []
        with torch.no_grad():
            for g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y in test_dataloader:
                g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y = g1_adjmat.to(device), g1_featmat.to(
                    device), g2_adjmat.to(device), g2_featmat.to(device), y.to(torch.float32).to(device)
                outputs, _, _ = model(g1_adjmat, g1_featmat, g2_adjmat, g2_featmat)
                loss = criterion(outputs, y)
                epoch_loss += loss.item()
                probs.extend(outputs.cpu().data)
                gts.extend(y.cpu().data)

                predicted = [1 if i > 0.5 else 0 for i in outputs.data]
                preds.extend(predicted)

            epoch_loss /= len(test_dataloader)
            accuracy = (np.array(preds) == np.array(gts)).sum() / len(gts)
            auc = roc_auc_score(np.array(gts), np.array(probs))
            print("        [Test] Epoch: %d, Loss: %f, Accuracy: %f, AUC: %f" %
                (epoch, epoch_loss, accuracy, auc))
    
    torch.save(model.state_dict(), config.Gemini_model_save_path)


if __name__ == "__main__":
    train()

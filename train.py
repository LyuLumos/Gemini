from datasets.gemini_dataset import dataloader_generate
from sklearn.metrics import roc_auc_score
from models.gemini import Gemini
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from configs import gemini_config as config


def run_epoch(model, dataloader, optimizer, criterion, device, training=True):
    if training:
        model.train()
    else:
        model.eval()
    
    epoch_loss = 0.0
    preds, gts, probs = [], [], []
    
    with torch.set_grad_enabled(training):
        for g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y in dataloader:
            g1_adjmat, g1_featmat, g2_adjmat, g2_featmat, y = g1_adjmat.to(device), g1_featmat.to(
                device), g2_adjmat.to(device), g2_featmat.to(device), y.to(torch.float32).to(device)
            
            if training:
                optimizer.zero_grad()
            outputs, _, _ = model(g1_adjmat, g1_featmat, g2_adjmat, g2_featmat)
            loss = criterion(outputs, y)
            
            if training:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
            probs.extend(outputs.cpu().data)
            gts.extend(y.cpu().data)

            predicted = [1 if i > 0.5 else 0 for i in outputs.data]
            preds.extend(predicted)

    epoch_loss /= len(dataloader)
    accuracy = (np.array(preds) == np.array(gts)).sum() / len(gts)
    auc = roc_auc_score(np.array(gts), np.array(probs))
    return epoch_loss, accuracy, auc


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataloader, test_dataloader, valid_dataloader = dataloader_generate()
    print(f'Using device: {device}')
    model = Gemini().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(config.epochs):
        train_loss, train_accuracy, train_auc = run_epoch(model, train_dataloader, optimizer, criterion, device, training=True)
        print("[Train] Epoch: %d, Loss: %f, Accuracy: %f, AUC: %f" % (epoch, train_loss, train_accuracy, train_auc))

        valid_loss, valid_accuracy, valid_auc = run_epoch(model, valid_dataloader, None, criterion, device, training=False)
        print("[Valid] Epoch: %d, Loss: %f, Accuracy: %f, AUC: %f" % (epoch, valid_loss, valid_accuracy, valid_auc))

        test_loss, test_accuracy, test_auc = run_epoch(model, test_dataloader, None, criterion, device, training=False)
        print("    [Test] Epoch: %d, Loss: %f, Accuracy: %f, AUC: %f" % (epoch, test_loss, test_accuracy, test_auc))

    torch.save(model.state_dict(), config.Gemini_model_save_path)


if __name__ == "__main__":
    train()

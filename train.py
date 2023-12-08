import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from gen_dataset import BCSDDataset, get_dataloader
from data_process import read_pkl



class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        # out: [batch_size, seq_len, hidden_size * num_directions]
        # h_n: [num_layers * num_directions, batch_size, hidden_size]
        # c_n: [num_layers * num_directions, batch_size, hidden_size]
        out, (h_n, c_n) = self.lstm(x)
        return out
    

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.att = nn.Linear(hidden_size * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size * num_directions]
        # att: [batch_size, seq_len, 1]
        att = self.att(x)
        # att: [batch_size, seq_len]
        att = att.squeeze(2)
        # att: [batch_size, seq_len]
        att = self.softmax(att)
        # att: [batch_size, seq_len, 1]
        att = att.unsqueeze(2)
        # out: [batch_size, hidden_size * num_directions]
        out = torch.sum(x * att, dim=1)
        return out


class BiLSTMAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTMAttention, self).__init__()
        self.bilstm = BiLSTM(input_size, hidden_size, num_layers)
        self.att = Attention(hidden_size)
        self.linear = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        # out: [batch_size, hidden_size * num_directions]
        out = self.bilstm(x)
        # out: [batch_size, hidden_size * num_directions]
        out = self.att(out)
        # out: [batch_size, 1]
        out = self.linear(out)
        return out

input_size = 10
hidden_size = 64 
num_layers = 2


model = BiLSTMAttention(input_size, hidden_size, num_layers)


def train(model, dataLoaders, optimizer, criterion, device, num_epochs=10):
    model.train()
    model.to(device)
    for epoch in range(num_epochs):
        total_loss, total_cnt = 0, 0
        for x86_func_names, x86_datas, x64_func_names, x64_datas, labels in tqdm(dataLoaders):
            x86_datas = x86_datas.to(device)
            x64_datas = x64_datas.to(device)
            labels = labels.to(device).float()

            outputs = model(x86_datas).flatten()
            # print(f'dtype: {outputs.dtype}, dtype: {labels.dtype}')
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cnt += labels.size(0).item()

        print(f'Epoch: {epoch + 1}, Loss: {total_loss / total_cnt}')


def test(model, dataLoaders, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for x86_func_names, x86_datas, x64_func_names, x64_datas, labels in dataLoaders:
            x86_datas = x86_datas.to(device)
            x64_datas = x64_datas.to(device)
            labels = labels.to(device).float()

            outputs = model(x86_datas).flatten()
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total}')


def main():
    dataset, _ = read_pkl('x86_x64_pairs.pkl')
    dataLoaders = get_dataloader(dataset, batch_size=4)
    dataLoaders = post_process(dataLoaders)

    learning_rate = 0.001
    num_epochs = 10

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(model, dataLoaders, optimizer, criterion, device, num_epochs=num_epochs)
    test(model, dataLoaders, device)


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


def post_process(dataloader):
    # 对长度超过1024的数据进行截断，对长度不足1024的数据进行padding
    max_len = 1024

    for x86_func_names, x86_datas, x64_func_names, x64_datas, labels in dataloader:
        pad_x86_datas = torch.zeros(x86_datas.shape[0], max_len, x86_datas.shape[2])
        pad_x64_datas = torch.zeros(x64_datas.shape[0], max_len, x64_datas.shape[2])
        for i, (x86_data, x64_data) in enumerate(zip(x86_datas, x64_datas)):
            if x86_data.shape[0] > max_len:
                pad_x86_datas[i] = x86_data[:max_len]
            else:
                pad_x86_datas[i, :x86_data.shape[0]] = x86_data

            if x64_data.shape[0] > max_len:
                pad_x64_datas[i] = x64_data[:max_len]
            else:
                pad_x64_datas[i, :x64_data.shape[0]] = x64_data
        yield x86_func_names, pad_x86_datas, x64_func_names, pad_x64_datas, labels


if __name__ == '__main__':
    main()
    # getmaxlen_and_minlen()

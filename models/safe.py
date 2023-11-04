import torch
import torch.nn as nn
import sys
sys.path.append("../")
from configs import safe_config as config

# Not Tested
class SAFE(nn.module):
    def __init__(self):
        super(SAFE, self).__init__()

        self.instruc_emb = nn.Embedding(config.num_embeddings, config.embedding_size)
        self.bi_rnn = nn.GRU(input_size=config.embedding_size, hidden_size=config.rnn_state_size, num_layers=config.rnn_depth, bidirectional=True, batch_first=True)
        
        self.Ws1 = nn.Parameter(torch.Tensor(config.attention_depth, config.rnn_state_size*2))
        self.Ws2 = nn.Parameter(torch.Tensor(config.attention_hops, config.attention_depth))
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.mlp = nn.Sequential(
            nn.Linear(config.attention_hops * config.rnn_state_size * 2, config.dense_layer_size),
            nn.ReLU(),
            nn.Linear(config.dense_layer_size, config.embedding_size),
            nn.functional.normalize(dim=1)
        )


    def forward(self, instructions):
        instruction_embeddings = self.instruc_emb(instructions)
        rnn_output, rnn_hidden = self.bi_rnn(instruction_embeddings)

        padded_rnn_output = torch.zeros(config.batch_size, config.max_instructions, config.rnn_state_size*2)
        padded_rnn_output[:, :rnn_output.shape[1], :] = rnn_output

        A = torch.matmul(self.Ws1, padded_rnn_output.transpose(1,2))
        A = self.tanh(A)
        A = torch.matmul(self.Ws2, A)
        A = self.softmax(A)

        B = torch.matmul(A, rnn_output)
        B = B.view(config.batch_size, -1)

        output = self.mlp(B)
        return output







        





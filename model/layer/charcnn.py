import torch
from torch import nn
import numpy as np
from torch.nn import functional as F


class CharCNN(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, vocab_size, dropout=0.1):
        super(CharCNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Một model Embedding có chứa vocab_size tensor có kích thước embedding_dim
        self.embeddings.weight.data.copy_ = torch.from_numpy(self.random_embedding(vocab_size, hidden_dim))
        # Sử dụng random_embedding làm ma trận trọng số ban đầu cho model Embedding

        self.kernels = [3, 4]
        cnns = []

        for k in self.kernels:
            seq = nn.Sequential(
                nn.Conv1d(embedding_dim, self.hidden_dim, kernel_size=k, padding=1),
                nn.Tanh()
            )
            cnns.append(seq)
        self.cnns = nn.ModuleList(cnns)
        self.dropout = nn.Dropout(dropout)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim]) # Tạo một mảng ngẫu nhiên có kích thước (vocab_size, embedding_dim) 
        # pretrain_emb = [vocab_size, embedding_dim]
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])   
            # pretrain_emb[index, :].shape = [1, embedding_dim] 
            # Mỗi phần tử trong mảng là số thực ngẫu nhiên r trong dãy -scale>= r >scale
        # pretrain_emb = [vocab_size, embedding_dim]
        return pretrain_emb

    def forward(self, x):
        # x = [bs, max_seq, max_char]
        seq_len = x.size(1)             
        batch_size = x.size(0)
        x = x.view(batch_size * seq_len, x.size(2))
        # x = [bs*seq_len, max_char]
        x = self.embeddings(x)
        # x = [bs*seq_len, max_char, char_embedding_dim]
        x = torch.transpose(x, 2, 1)
        # x = [bs*seq_len, char_embedding_dim, max_char]
        tmp = [cnn(x) for cnn in self.cnns]
        # self.cnns[0](x) = [bs*seq_len, char_hidden_dim, max_char]
        char_cnn = []
        for item in tmp:
            char_cnn.append(F.max_pool1d(item, item.size(2)))
        # F.max_pool1d(self.cnns[0](x)) = [bs*seq_len, char_hidden_dim, 1]
        char_cnn = torch.cat(char_cnn, dim=1)
        # char_cnn = [bs*seq_len, char_hidden_dim*2, 1]
        char_out = char_cnn.view(batch_size, seq_len, -1)
        # char_out = [bs, seq_len, char_hidden_dim*2]
        char_out = self.dropout(char_out)
        # char_out = [bs, seq_len, char_hidden_dim*2]
        return char_out
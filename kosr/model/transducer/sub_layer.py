import math
import torch
import torch.nn as nn

from kosr.model.attention import *

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_length=2000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_length, dim_model, requires_grad=False)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        exp_term = torch.exp(torch.arange(0, dim_model, 2).float() * -(math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * exp_term)
        pe[:, 1::2] = torch.cos(position * exp_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        return self.pe[:, :input.size(1)]

class LayerNorm(nn.LayerNorm):
    def __init__(self, nout, eps=1e-6, dim=-1):
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)
    
class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, hidden_size, filter_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, filter_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(filter_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        return x
import torch
import torch.nn as nn
from kosr.model.attention import MultiHeadAttention, RelPositionMultiHeadAttention
from kosr.model.transformer.sub_layer import *

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, dropout_rate):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = RelPositionMultiHeadAttention(n_head, hidden_size, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, filter_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):  # pylint: disable=arguments-differ
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x, mask
    
class Encoder(nn.Module):
    def __init__(self, hidden_size, filter_size, n_head, dropout_rate, n_layers):
        super(Encoder, self).__init__()
        self.first_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, filter_size, n_head, dropout_rate)
                    for _ in range(n_layers)])

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, mask):
        encoder_output = self.first_norm(inputs + self.pe(inputs.size(1)))
        for enc_layer in self.layers:
            encoder_output, mask = enc_layer(encoder_output, mask)
        return self.last_norm(encoder_output), mask
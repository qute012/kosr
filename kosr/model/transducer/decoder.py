import math
import torch
import torch.nn as nn

from kosr.model.attention import MultiHeadAttention, RelPositionMultiHeadAttention
from kosr.model.transducer.sub_layer import PositionalEncoding, FeedForwardNetwork, LayerNorm
from kosr.model.mask import target_mask

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, filter_dim, n_head, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.att_norm = LayerNorm(hidden_dim, eps=1e-6)
        self.att = MultiHeadAttention(hidden_dim, n_head, dropout_rate=0.0)

        self.memory_att_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.memory_att = MultiHeadAttention(hidden_dim, n_head, dropout_rate=0.0)

        self.ffn_norm = LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_dim, filter_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, x_mask, memory, memory_mask):
        y = self.att_norm(x)
        y = self.att(y, y, y, x_mask)
        x = x + y

        if memory is not None:
            y = self.memory_att_norm(x)
            y = self.memory_att(y, memory, memory, memory_mask)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = x + self.dropout(y)
        return y

class Decoder(nn.Module):
    def __init__(self, out_dim, hidden_dim, filter_dim, n_head, dropout_rate, n_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(out_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.scale = math.sqrt(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, filter_dim, n_head, dropout_rate)
                    for _ in range(n_layers)])

        self.last_norm = LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, tgt, tgt_mask=None, memory=None, memory_mask=None):
        #tgt_mask = None
        decoder_output = self.dropout(self.embed(tgt)*self.scale + self.pos_enc(tgt))
        for i, dec_layer in enumerate(self.layers):
            decoder_output = dec_layer(decoder_output, tgt_mask, memory, memory_mask)
        decoder_output = self.last_norm(decoder_output)
        return decoder_output
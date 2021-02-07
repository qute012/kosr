import math
import torch
import torch.nn as nn

from kosr.model.attention import MultiHeadAttention, RelPositionMultiHeadAttention
from kosr.model.transducer.sub_layer import PositionalEncoding, FeedForwardNetwork, LayerNorm
from kosr.model.mask import make_non_pad_mask

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, filter_dim, n_head, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.att_norm = LayerNorm(hidden_dim, eps=1e-6)
        #self.rel_att = RelPositionMultiHeadAttention(hidden_dim, n_head, dropout_rate)
        self.att = MultiHeadAttention(hidden_dim, n_head, dropout_rate=0.0)

        self.ffn_norm = LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_dim, filter_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        y = self.att_norm(x)
        #y = self.rel_att(x, x, x, pos_enc, mask)
        y = self.att(y,y,y,mask)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = x + self.dropout(y)
        #x = self.dropout(x)
        return y, mask
    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, filter_dim, n_head, dropout_rate, n_layers):
        super(Encoder, self).__init__()
        self.scale = math.sqrt(hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, filter_dim, n_head, dropout_rate)
                    for _ in range(n_layers)])
        self.norm = LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, inputs, mask=None):
        #mask = None
        encoder_output = self.dropout(inputs*self.scale + self.pos_enc(inputs))
        for enc_layer in self.layers:
            encoder_output, mask = enc_layer(encoder_output, mask)
        encoder_output = self.norm(encoder_output)
        return encoder_output, mask
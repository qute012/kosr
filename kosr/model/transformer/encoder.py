import math
import torch
import torch.nn as nn

from kosr.model.attention import MultiHeadAttention, RelPositionMultiHeadAttention
from kosr.model.transformer.sub_layer import PositionalEncoding, FeedForwardNetwork
from kosr.model.mask import make_non_pad_mask

class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, filter_dim, n_head, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.pos_enc = PositionalEncoding(hidden_dim)
        
        self.att_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        #self.rel_att = RelPositionMultiHeadAttention(hidden_dim, n_head, dropout_rate)
        self.att = MultiHeadAttention(hidden_dim, n_head, dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_dim, filter_dim, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        x = self.att_norm(x)
        pos_enc = self.pos_enc(x)
        #y = self.rel_att(x, x, x, pos_enc, mask)
        y = self.att(x,x,x,mask)
        x = x + y

        x = self.ffn_norm(x)
        y = self.ffn(y)
        x = x + y
        x = self.dropout(x)
        return x, mask
    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, filter_dim, n_head, dropout_rate, n_layers):
        super(Encoder, self).__init__()
        self.scale = math.sqrt(hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([EncoderLayer(hidden_dim, filter_dim, n_head, dropout_rate)
                    for _ in range(n_layers)])

    def forward(self, inputs, input_length):
        mask = make_non_pad_mask(input_length)
        
        encoder_output = self.dropout(inputs*self.scale + self.pos_enc(inputs))
        for enc_layer in self.layers:
            encoder_output, mask = enc_layer(encoder_output, mask)
        return encoder_output, mask
import torch
import torch.nn as nn
from kosr.model.attention import MultiHeadAttention, RelPositionMultiHeadAttention
from kosr.model.transformer.sub_layer import *

class DecoderLayer(nn.Module):
    def __init__(self, hidden_dim, filter_dim, n_head, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = MultiHeadAttention(n_head, hidden_dim, dropout_rate)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(n_head, hidden_dim, dropout_rate)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_dim, filter_dim, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, tgt_mask, enc_mask, cache):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y = self.enc_dec_attention(y, enc_output, enc_output, i_mask,
                                       cache)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_dim, filter_dim, n_head, dropout_rate, n_layers):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([DecoderLayer(hidden_dim, filter_dim, n_head, dropout_rate)
                    for _ in range(n_layers)])

        self.last_norm = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, targets, enc_output, enc_mask, cache):
        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output = dec_layer(decoder_output, enc_output, tgt_mask, enc_mask, layer_cache)
        return self.last_norm(decoder_output)
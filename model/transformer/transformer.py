import torch
import torch.nn as nn

from encoder import *
from decoder import *

class Transformer(nn.Module):
    def __init__(self, vocab_size, enc_n_layers=16, dec_n_layer=1, hidden_size=512, filter_size=2048, dropout_rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(hidden_size, filter_size,
                               dropout_rate, n_layers)
        
        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, n_layers)
        
        self.initialize()

    def forward(self, padded_input, input_lengths, padded_target):
        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn':
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH

        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)
        hyp_best_scores, hyp_best_ids = torch.topk(pred, 1, dim=2)

        hyp_seq = hyp_best_ids.squeeze(2)
        gold_seq = gold

        return pred, gold, hyp_seq, gold_seq

    def initialize(self):
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if init_type == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(p.data)
                elif init_type == "xavier_normal":
                    torch.nn.init.xavier_normal_(p.data)
                elif init_type == "kaiming_uniform":
                    torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif init_type == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + init_type)
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (nn.Embedding, mm.LayerNorm)):
                m.reset_parameters()
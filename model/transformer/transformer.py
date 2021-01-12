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

    def forward(self, inputs, targets):
        enc_output, i_mask = None, None
        if self.has_inputs:
            i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)
            enc_output = self.encode(inputs, i_mask)

        t_mask = utils.create_pad_mask(targets, self.trg_pad_idx)
        target_size = targets.size()[1]
        t_self_mask = utils.create_trg_self_mask(target_size,
                                                 device=targets.device)
        return self.decode(targets, enc_output, i_mask, t_self_mask, t_mask)

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
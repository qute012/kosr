import torch
import torch.nn as nn

from kosr.model.feature_extractor import *
from kosr.model.transformer.encoder import Encoder
from kosr.model.transformer.decoder import Decoder

class Transformer(nn.Module):
    def __init__(
        self, 
        out_dim, 
        feat_extractor='vgg', 
        enc_n_layers=16, 
        dec_n_layers=1, 
        hidden_dim=512, 
        filter_dim=2048,
        n_head=8,
        dropout_rate=0.1, 
        pad_id=0, 
        sos_id=1, 
        eos_id=2,
        init_type="xavier_uniform"
    ):
        super(Transformer, self).__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.init_type = init_type
        
        self.feat_extractor = feat_extractor
        if feat_extractor=='vgg':
            self.conv = VGGExtractor()
        elif feat_extractor=='w2v':
            self.conv = W2VExtractor()
            
        self.encoder = Encoder(hidden_dim, filter_dim, n_head,
                               dropout_rate, enc_n_layers)
        
        self.decoder = Decoder(hidden_dim, filter_dim, n_head,
                               dropout_rate, dec_n_layers)
        
        self.fc = nn.Linear(hidden_dim, out_dim)
        
        self.initialize()

    def forward(self, inputs, input_length, tgt):
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
            padded_input = self.conv(padded_input)

        enc_out, enc_mask = self.encoder(inputs, input_length)
        pred = self.decoder(tgt, enc_out, enc_mask)
        
        return pred

    def initialize(self):
        # weight init
        for p in self.parameters():
            if p.dim() > 1:
                if self.init_type == "xavier_uniform":
                    nn.init.xavier_uniform_(p.data)
                elif self.init_type == "xavier_normal":
                    nn.init.xavier_normal_(p.data)
                elif self.init_type == "kaiming_uniform":
                    nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
                elif self.init_type == "kaiming_normal":
                    nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                else:
                    raise ValueError("Unknown initialization: " + self.init_type)
        # bias init
        for p in self.parameters():
            if p.dim() == 1:
                p.data.zero_()

        # reset some modules with default init
        for m in self.modules():
            if isinstance(m, (nn.Embedding, nn.LayerNorm)):
                m.reset_parameters()
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
        enc_n_layers=14, 
        dec_n_layers=6, 
        hidden_dim=512, 
        filter_dim=2048,
        n_head=8,
        dropout_rate=0.1,
        max_len=150,
        pad_id=0, 
        sos_id=1, 
        eos_id=2,
        init_type="xavier_uniform"
    ):
        super(Transformer, self).__init__()
        self.max_len = max_len
        self.out_dim = out_dim
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.init_type = init_type
        
        self.feat_extractor = feat_extractor
        if feat_extractor=='vgg':
            self.conv = VGGExtractor(hidden_dim=hidden_dim)
        elif feat_extractor=='w2v':
            self.conv = W2VExtractor(hidden_dim=hidden_dim)
            
        self.encoder = Encoder(hidden_dim, filter_dim, n_head,
                               dropout_rate, enc_n_layers)
        
        self.decoder = Decoder(out_dim, hidden_dim, filter_dim, 
                               n_head,dropout_rate, dec_n_layers, pad_id)
        
        self.initialize()

    def forward(self, inputs, input_length, tgt):
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
            inputs,input_length = self.conv(inputs), input_length>>2
        enc_out, enc_mask = self.encoder(inputs, input_length)
        
        tgt_in, golds = self.make_in_out(tgt)

        preds = self.decoder(tgt_in, enc_out, enc_mask)
        
        return preds, golds
    
    def recognize(self, inputs, input_length, tgt=None, mode='greedy'):
        if mode == 'greedy':
            preds, golds, y_hats = self.greedy_search(inputs, input_length, tgt)
        elif mode == 'beam':
            preds, golds, y_hats = self.beam_search(inputs, input_length, tgt)
        
        if golds is None:
            return preds, y_hats
        else:
            return preds, golds, y_hats
        
    def greedy_search(self, inputs, input_length, tgt=None):
        btz = inputs.size(0)
        device = inputs.device
        
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
            inputs,input_length = self.conv(inputs), input_length>>2

        enc_out, enc_mask = self.encoder(inputs, input_length)
        
        preds = torch.zeros(btz, self.max_len, self.out_dim, dtype=torch.float32).to(device)
        y_hats = torch.zeros(btz, self.max_len, dtype=torch.long).to(device)
        
        tgt_in = torch.zeros(btz,1, dtype=torch.long).fill_(self.sos_id).to(device)
        for step in range(self.max_len):
            pred = self.decoder(tgt_in, enc_out, enc_mask)
            preds[:,step,:] = pred.squeeze(-2)
            y_hat = pred.max(-1)[1]
            tgt_in = y_hat
            y_hats[:,step] = y_hat.squeeze(dim=-1)
        
        if tgt is None:
            """for testing"""
            golds = None
        else:
            """for validation"""
            golds = tgt[tgt!=self.sos_id].view(btz,-1)[:, :self.max_len].contiguous()
            preds = preds[:, :golds.size(1)].contiguous()
        
        return preds, golds, y_hats
        
    def beam_search(self):
        raise NotImplementedError
    
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
                
    def make_in_out(self, tgt):
        btz = tgt.size(0)
        
        tgt_in = torch.clone(tgt)
        tgt_in[tgt_in==self.pad_id] = self.eos_id
        tgt_in = tgt_in.view(btz,-1)[:, :-1]

        tgt_out = tgt[tgt!=self.sos_id].view(btz,-1)
        
        return tgt_in, tgt_out
import torch
import torch.nn as nn
import torch.nn.functional as F

from kosr.model.feature_extractor import *
from kosr.model.transducer.encoder import Encoder
from kosr.model.transducer.decoder import Decoder
from kosr.model.transducer.joint import JointNet

class Transducer(nn.Module):
    def __init__(
        self, 
        out_dim,
        feat_extractor='vgg', 
        enc_n_layers=16, 
        dec_n_layers=1, 
        hidden_dim=512, 
        filter_dim=2048,
        n_head=32,
        dropout_rate=0.1,
        max_len=300,
        pad_id=0, 
        sos_id=1, 
        eos_id=2,
        init_type="xavier_uniform"
    ):
        super(Transducer, self).__init__()
        self.max_len = max_len
        self.out_dim = out_dim
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
        
        self.decoder = Decoder(out_dim, hidden_dim, filter_dim, 
                               n_head, dropout_rate, dec_n_layers, pad_id)
        
        self.jointer = JointNet(out_dim, hidden_dim*2, hidden_dim, dropout_rate)
        self.initialize()

    def forward(self, inputs, input_length, tgt):
        btz = tgt.size(0)
        
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
            inputs,input_length = self.conv(inputs), input_length>>2
        enc_state, enc_mask = self.encoder(inputs, input_length)
        dec_state = self.decoder(tgt[tgt!=self.eos_id].view(btz,-1))
        
        preds = self.jointer(enc_state, dec_state)
        golds = tgt[tgt!=self.sos_id].view(btz,-1)
        
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
        
        if tgt is None:
            """for testing"""
            golds = None
            tgt = torch.zeros(btz,1, dtype=torch.long).fill_(self.sos_id).to(device)
        else:
            """for validation"""
            golds = tgt[tgt!=self.sos_id].view(btz,-1)
            tgt = tgt[:,:1]
        
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
            inputs,input_length = self.conv(inputs), input_length>>2

        enc_state, enc_mask = self.encoder(inputs, input_length)
        preds = torch.zeros(btz, self.max_len, self.out_dim, dtype=torch.float32).to(device)
        y_hats = torch.zeros(btz, self.max_len, dtype=torch.long).to(device)
        for step in range(self.max_len):
            dec_state = self.decoder(tgt)
            pred = self.jointer(enc_state, dec_state)
            preds[:,step,:] = pred.squeeze(-2)
            y_hat = pred.max(-1)[1]
            tgt = y_hat
            y_hats[:,step] = y_hat.squeeze(dim=-1)
        return preds, y_hats
        
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
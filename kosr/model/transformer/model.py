import torch
import torch.nn as nn


from kosr.feature_extractor import *

class Transformer(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        feat_extractor='vgg', 
        enc_n_layers=16, 
        dec_n_layers=1, 
        hidden_size=512, 
        filter_size=2048,
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
        
        if feat_extractor=='vgg':
            self.conv = VGGExtractor()
        elif feat_extractor=='w2v':
            self.conv = W2VExtractor()
            
        self.encoder = Encoder(hidden_size, filter_size, n_head,
                               dropout_rate, enc_n_layers)
        
        self.decoder = Decoder(hidden_size, filter_size,n_head,
                               dropout_rate, dec_n_layers)
        
        self.initialize()

    def forward(self, padded_input, input_lengths, padded_target):
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
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
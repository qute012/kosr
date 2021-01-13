import torch
import torch.nn as nn

from encoder import *
from decoder import *
from ..feature_extractor import *

class Transformer(nn.Module):
    def __init__(self, vocab_size, feat_extractor='vgg', enc_n_layers=16, dec_n_layer=1, hidden_size=512, filter_size=2048, dropout_rate=0.1):
        super(Transformer, self).__init__()
        if feat_extractor=='vgg':
            self.conv = VGGExtracter()
        elif feat_extractor=='w2v':
            self.conv = W2VExtracter()
            
        self.encoder = Encoder(hidden_size, filter_size,
                               dropout_rate, n_layers)
        
        self.decoder = Decoder(hidden_size, filter_size,
                               dropout_rate, n_layers)
        
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
    
    def recognize(self):
        btz = listener_features.size(0)
        y_hats = torch.zeros(btz, max_step).long().cuda()
        logit, hc, context = self.forward_step(inputs, hc, listener_features)
        output_words = logit.topk(beam_size)[1].squeeze(1)
        for bi in range(btz):
            b_output_words = output_words[bi,:].unsqueeze(0).transpose(1,0).contiguous()
            b_inputs = self.emb(b_output_words)
            b_listener_features = listener_features[bi,:,:].unsqueeze(0).expand((beam_size,-1,-1)).contiguous()
            if isinstance(hc, tuple):
                b_h = hc[0][:,bi,:].unsqueeze(1).expand((-1,beam_size,-1)).contiguous()
                b_c = hc[1][:,bi,:].unsqueeze(1).expand((-1,beam_size,-1)).contiguous()
                b_hc = (b_h, b_c)
            else:
                b_hc = hc[:,bi,:].unsqueeze(1).expand((-1,beam_size,-1)).contiguous()

            scores = torch.zeros(beam_size,1).cuda()
            ids = torch.zeros(beam_size, max_step, 1).long().cuda()
            for step in range(max_step):
                logit, b_hc, context = self.forward_step(b_inputs, b_hc, b_listener_features)
                score, id = logit.topk(1)
                scores += score.squeeze(1)
                ids[:,step,:] = id.squeeze(1)
                output_word = logit.topk(1)[1].squeeze(-1)
                b_inputs = self.emb(output_word)
            y_hats[bi,:] = ids[scores.squeeze(1).topk(1)[1],:].squeeze(2)
        return y_hats
    
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
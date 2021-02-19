import torch
import torch.nn as nn
import torch.nn.functional as F

from kosr.model.feature_extractor import *
from kosr.model.transformer.encoder import Encoder
from kosr.model.transformer.decoder import Decoder
from kosr.model.mask import target_mask, subsequent_mask, get_attn_pad_mask

class Transformer(nn.Module):
    def __init__(
        self, 
        out_dim,
        in_dim=80,
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
            self.conv = VGGExtractor(in_dim=in_dim, hidden_dim=hidden_dim)
        elif feat_extractor=='w2v':
            self.conv = W2VExtractor(in_dim=in_dim, hidden_dim=hidden_dim)
            
        self.encoder = Encoder(hidden_dim, filter_dim, n_head,
                               dropout_rate, enc_n_layers)
        
        self.decoder = Decoder(out_dim, hidden_dim, filter_dim, 
                               n_head,dropout_rate, dec_n_layers)
        
        self.initialize()

    def forward(self, inputs, input_length, tgt):
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
            inputs,input_length = self.conv(inputs), input_length>>2
            
        enc_mask = get_attn_pad_mask(input_length).to(inputs.device)
        enc_out, enc_mask = self.encoder(inputs, enc_mask)
        
        tgt_in, golds = self.make_in_out(tgt)
        tgt_mask = target_mask(tgt_in, ignore_id=self.pad_id).to(tgt.device)
        
        preds = self.decoder(tgt_in, tgt_mask, enc_out, enc_mask)
        
        return preds, golds
    
    def recognize(self, inputs, input_length, tgt=None, search='beam'):
        if search == 'greedy':
            preds, golds, y_hats = self.greedy_search(inputs, input_length, tgt)
        elif search == 'beam':
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
        
        enc_mask = get_attn_pad_mask(input_length).to(inputs.device)
        enc_out, enc_mask = self.encoder(inputs, enc_mask)

        preds = torch.zeros(btz, self.max_len, self.out_dim, dtype=torch.float32).to(device)
        y_hats = torch.zeros(btz, self.max_len, dtype=torch.long).to(device)
        
        tgt_in = torch.zeros(btz,1, dtype=torch.long).fill_(self.sos_id).to(device)
        for step in range(self.max_len):
            tgt_mask = subsequent_mask(step+1).to(tgt.device).eq(0).unsqueeze(0)
            pred = self.decoder(tgt_in, tgt_mask, enc_out, enc_mask)
            pred = pred[:, -1, :]
            y_hat = pred.max(-1)[1]
            tgt_in = torch.cat((tgt_in,y_hat.unsqueeze(1)), dim=1)
            preds[:,step,:] = pred.squeeze()
            y_hats[:,step] = y_hat.squeeze(dim=-1)
        if tgt is None:
            """for testing"""
            golds = None
        else:
            """for validation"""
            golds = tgt[tgt!=self.sos_id].view(btz,-1)[:, :self.max_len].contiguous()
            preds = preds[:, :golds.size(1)].contiguous()
        
        return preds, golds, y_hats
        
    def beam_search(self, inputs, input_length, tgt=None, K=16):        
        btz = inputs.size(0)
        device = inputs.device
        
        def get_length_penalty(length, alpha=1.2, min_length=1):
            p = (1 + length) ** alpha / (1 + min_length) ** alpha

            return p
        
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
            inputs,input_length = self.conv(inputs), input_length>>2
        
        enc_mask = get_attn_pad_mask(input_length).to(inputs.device)
        enc_out, enc_mask = self.encoder(inputs, enc_mask)
        
        y_hats = []
        
        for bi in range(btz):
            encoder_output = enc_out[bi].unsqueeze(0)
            encoder_mask = enc_mask[bi].unsqueeze(0)
            tgt_in = torch.zeros(1,1, dtype=torch.long).fill_(self.sos_id).to(device)
            
            hyp = {'score': 0.0, 'yseq':tgt_in}
            hyps = [hyp]
            ended_hyps = []
            
            for step in range(self.max_len):
                hyps_best_kept = []
                for hyp in hyps:
                    ys = hyp['yseq']
                    tgt_mask = subsequent_mask(step+1).to(tgt.device).eq(0).unsqueeze(0)
                    logits = self.decoder(ys, tgt_mask, encoder_output, encoder_mask)
                    logits = F.log_softmax(logits[:, -1, :], dim=-1)
                    local_best_scores, local_best_ids = torch.topk(logits.squeeze(1), K, dim=1)
                    
                    for j in range(K):
                        new_hyp = {}
                        new_hyp["score"] = hyp["score"] + local_best_scores[0, j]

                        new_hyp["yseq"] = torch.ones(1, (1+ys.size(1)), dtype=torch.long).to(device)
                        new_hyp["yseq"][:, :ys.size(1)] = hyp["yseq"].cpu()
                        new_hyp["yseq"][:, ys.size(1)] = int(local_best_ids[0, j]) # adding new word
                        
                        hyps_best_kept.append(new_hyp)

                    hyps_best_kept = sorted(hyps_best_kept, key=lambda x:x["score"], reverse=True)[:K]
                
                hyps = hyps_best_kept
                
                if step==self.max_len-1:
                    for hyp in hyps:
                        hyp["yseq"] = torch.cat([hyp["yseq"], torch.ones(1,1, dtype=torch.long).fill_(self.eos_id).to(device)], dim=1)

                unended_hyps = []
                for hyp in hyps:
                    if hyp["yseq"][0,-1] == self.eos_id:
                        seq = hyp["yseq"]
                        seq_len = seq[:torch.where(seq==self.eos_id)[0][0]].size(0)
                        hyp["final_score"] = hyp["score"] / get_length_penalty(seq_len)

                        ended_hyps.append(hyp)
                        
                    else:
                        unended_hyps.append(hyp)
                hyps = unended_hyps
                
            nbest_hyps = sorted(ended_hyps, key=lambda x:x["final_score"], reverse=True)[:1]

            for hyp in nbest_hyps:                
                hyp["yseq"] = hyp["yseq"][0].cpu().numpy().tolist()
                y_hats.append(hyp["yseq"])
        
        if tgt is None:
            """for testing"""
            golds = None
        else:
            """for validation"""
            golds = tgt[tgt!=self.sos_id].view(btz,-1)[:, :self.max_len].contiguous()
            #preds = preds[:, :golds.size(1)].contiguous()
        
        preds = None
        
        return preds, golds, y_hats
    
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
        
        #tgt_in = tgt[tgt!=self.eos_id].view(btz,-1)
        tgt_out = tgt[tgt!=self.sos_id].view(btz,-1)
        
        return tgt_in, tgt_out
    
class TransformerJointCTC(Transformer):
    def __init__(
        self, 
        out_dim,
        in_dim=80,
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
        super(TransformerJointCTC, self).__init__(
            out_dim, in_dim, feat_extractor, 
            enc_n_layers, dec_n_layers, 
            hidden_dim, filter_dim,
            n_head, dropout_rate, max_len, 
            pad_id, sos_id, eos_id
        )
        
        self.ctc_logistic = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        
    def forward(self, inputs, input_length, tgt):
        btz = inputs.size(0)
        
        if self.feat_extractor == 'vgg' or self.feat_extractor == 'w2v':
            inputs,input_length = self.conv(inputs), input_length>>2
        
        enc_mask = get_attn_pad_mask(input_length).to(inputs.device)
        enc_out, enc_mask = self.encoder(inputs, enc_mask)
        
        tgt_in, att_golds = self.make_in_out(tgt)
        tgt_mask = target_mask(tgt_in, ignore_id=self.pad_id).to(tgt.device)
        
        att_out = self.decoder(tgt_in, tgt_mask, enc_out, enc_mask)
        ctc_out = self.ctc_logistic(self.dropout(enc_out)).transpose(0,1)
        ctc_out = F.log_softmax(ctc_out, dim=-1)
        
        ctc_golds = att_golds[att_golds!=self.eos_id].view(btz,-1)
        golds_length = torch.LongTensor([x[x!=self.pad_id].size(0) for x in ctc_golds]).to(inputs.device)
        #ctc_golds = torch.cat([ctc_golds[i][:l] for i,l in enumerate(golds_length)]).to(inputs.device)
        #golds_length = torch.LongTensor(golds_length).to(inputs.device)
        
        return att_out, att_golds, ctc_out, ctc_golds, input_length, golds_length
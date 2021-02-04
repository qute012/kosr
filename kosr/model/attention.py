import math
import numpy
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, n_head, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = hidden_dim // n_head
        self.n_head = n_head
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        btz = query.size(0)
        q = self.w_q(query).view(btz, -1, self.n_head, self.d_k)
        k = self.w_k(key).view(btz, -1, self.n_head, self.d_k)
        v = self.w_v(value).view(btz, -1, self.n_head, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        
        if mask is not None:
            min_value = float(
                numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_head * self.d_k)) 

        return self.out_proj(x)

    def forward(self, query, key, value, mask=None):
        btz = query.size(0)
        
        q, k, v = self.forward_qkv(query, key, value)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadAttention(MultiHeadAttention):
    def __init__(self, hidden_dim, n_head, dropout_rate):
        super().__init__(hidden_dim, n_head, dropout_rate)
        self.w_pos = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.n_head, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu=False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.w_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)
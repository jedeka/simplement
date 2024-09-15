import random
from typing import Optional

import numpy as np
import torch
from torch import nn 
from torch.nn.init import xavier_uniform_import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        # TODO:Q: Why 2,3? Why apply temperature in query?
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) 

        if mask is not None:
            # TODO:Q: How does the mask work? 
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn 

class MultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head 
        self.d_k = d_k
        self.d_v = d_v 

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attn = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head 
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # pass through the pre-attention projection: b x lq x (n * dv)
        # seperate different heads: bx lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_qk(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_qv(v).view(sz_b, len_v, n_head, d_v)
        
        if mask is not None:
            mask  = mask.unsqueeze(1)
        
        q, attn = self.attn(q, k, v, mask=mask)

        # transpose to move the head dim back: b*lq*n*dv
        # combine the last 2 dims to concat all the heads together 
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual 

        q = self.layer_norm(q)

        return q, attn 
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in) 
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual 

        x = self.layer_norm(x)

        return x 
    


class EncoderLayer(nn.Module):
    """Transformer encoder layer"""
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, self_attn_mask=None):
        enc_output, enc_self_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=self_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_self_attn


class DecoderLayer(nn.Module):
    """Transformer decoder layer"""
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, 
                dec_input, 
                enc_output, 
                self_attn_mask=None, 
                dec_enc_attn_mask=None):
        
        dec_output, dec_self_attn = self.self_attn(
            dec_input, dec_input, dec_input, mask=self_attn_mask
        ) 
        dec_output, dec_enc_attn = self.self_attn(
            dec_input, enc_output, enc_output, mask=dec_enc_attn_mask
        ) 
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_self_attn, dec_enc_attn
    
